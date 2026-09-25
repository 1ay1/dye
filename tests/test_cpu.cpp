// test_cpu.cpp — the reference renderer, checked against arithmetic done by
// hand.
//
// This renderer is the oracle the GPU backend will be compared against, so
// it has to be right on its own terms first. "It looks fine" is not
// available here: every case computes the expected pixel independently and
// compares.
#include "check.hpp"

#include <dye/cpu.hpp>

#include <cstdio>
#include <utility>
#include <vector>

using namespace dye;
using namespace dye::cpu;

namespace {

// ---------------------------------------------------------------------------
// Arithmetic. These are the functions every other test depends on, so they
// are checked in isolation before anything draws.
// ---------------------------------------------------------------------------

// mul255 rounds; the naive (a*b)/255 truncates. One unit per blend does not
// sound like much until a stack of translucent windows has turned grey.
static_assert(mul255(255, 255) == 255);
static_assert(mul255(0, 255) == 0);
static_assert(mul255(255, 0) == 0);
static_assert(mul255(128, 255) == 128);   // truncating gives 127
static_assert(mul255(128, 128) == 64);    // and 64 here
static_assert(mul255(16, 16) == 1);

// Source-over with premultiplied inputs. Opaque source wins outright;
// transparent source changes nothing. Those two are most of a real frame.
static_assert(over(Pixel{10, 20, 30, 255}, Pixel{99, 99, 99, 255}) == Pixel{10, 20, 30, 255});
static_assert(over(Pixel{0, 0, 0, 0}, Pixel{99, 98, 97, 255}) == Pixel{99, 98, 97, 255});

// Half-transparent white over black: premultiplied, so the source is
// (128,128,128,128), and the result is 128 + 0 * (1 - 0.5) = 128.
static_assert(over(Pixel{128, 128, 128, 128}, Pixel{0, 0, 0, 255}).r == 128);
static_assert(over(Pixel{128, 128, 128, 128}, Pixel{0, 0, 0, 255}).a == 255);

// Packing round-trips.
static_assert(pack(Pixel{0x11, 0x22, 0x33, 0x44}) == 0x44332211u);
static_assert(unpack(0x44332211u) == Pixel{0x11, 0x22, 0x33, 0x44});

// Premultiplying a straight colour. Half-transparent pure red becomes
// (128, 0, 0) at alpha 128, not (255, 0, 0).
static_assert(premultiply(Colour{255, 0, 0, 128}).r == 128);
static_assert(premultiply(Colour{255, 0, 0, 128}).a == 128);
static_assert(premultiply(Colour{255, 255, 255, 255}) == Pixel{255, 255, 255, 255});
static_assert(premultiply(Colour{255, 255, 255, 0}) == Pixel{0, 0, 0, 0});

// ---------------------------------------------------------------------------
// A framebuffer to draw into, and a client buffer to draw from.
// ---------------------------------------------------------------------------

struct Canvas {
    std::vector<std::uint32_t> mem;
    Surface surface{};

    Canvas(std::int32_t w, std::int32_t h, std::uint32_t fill_with = 0xff000000u)
        : mem(static_cast<std::size_t>(w * h), fill_with) {
        surface.pixels = mem.data();
        surface.stride_bytes = w * 4;
        surface.size = {w, h};
    }

    std::uint32_t at(std::int32_t x, std::int32_t y) const {
        return mem[static_cast<std::size_t>(y) * static_cast<std::size_t>(surface.size.width) +
                   static_cast<std::size_t>(x)];
    }
};

/// A client buffer of one flat colour, in the given format.
struct Buf {
    std::vector<std::uint8_t> mem;
    Pixels pixels{};

    Buf(std::int32_t w, std::int32_t h, Format f, std::uint8_t c0, std::uint8_t c1,
        std::uint8_t c2, std::uint8_t c3)
        : mem(static_cast<std::size_t>(w * h * 4)) {
        for (std::size_t i = 0; i < mem.size(); i += 4) {
            mem[i] = c0;
            mem[i + 1] = c1;
            mem[i + 2] = c2;
            mem[i + 3] = c3;
        }
        pixels.data = mem.data();
        pixels.stride_bytes = w * 4;
        pixels.size = {w, h};
        pixels.format = f;
    }

    /// Set one pixel, in memory order.
    void put(std::int32_t x, std::int32_t y, std::uint8_t c0, std::uint8_t c1,
             std::uint8_t c2, std::uint8_t c3) {
        const std::size_t i =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(pixels.size.width) +
             static_cast<std::size_t>(x)) * 4;
        mem[i] = c0;
        mem[i + 1] = c1;
        mem[i + 2] = c2;
        mem[i + 3] = c3;
    }
};

Frame one_layer(Layer l, weft::Rect<weft::Output> damage) {
    Frame f;
    f.layers.push_back(std::move(l));
    f.damage.add(damage);
    return f;
}

// ---------------------------------------------------------------------------

void x_formats_are_opaque() {
    std::printf("an X format's fourth byte is not alpha\n");

    // The bug: a client attaches XRGB and leaves garbage in the top byte.
    // Read as ARGB, that byte becomes alpha, and the window renders with
    // holes in it. It usually LOOKS fine, because most allocators leave
    // 0xff there — so the test uses 0x00, the value that would make the
    // whole window vanish.
    Buf xrgb{4, 4, formats::xrgb8888, 0x00, 0x00, 0xff, 0x00};   // red, alpha byte 0
    Canvas c{4, 4, pack(Pixel{0, 0, 0, 255})};

    Layer l;
    l.source = Texture{TextureId{1}, {4, 4}, formats::xrgb8888};
    l.dst = {0, 0, 4, 4};
    draw_layer(c.surface, l, xrgb.pixels, {0, 0, 4, 4});

    // Fully opaque red, whatever the fourth byte said.
    CHECK(c.at(0, 0) == pack(Pixel{0, 0, 255, 255}));
    CHECK(c.at(3, 3) == pack(Pixel{0, 0, 255, 255}));

    // The same bytes as ARGB really are transparent, so the destination is
    // untouched. If both cases gave the same answer the test would prove
    // nothing about the format check.
    Buf argb{4, 4, formats::argb8888, 0x00, 0x00, 0xff, 0x00};
    Canvas c2{4, 4, pack(Pixel{0, 0, 0, 255})};
    Layer l2 = l;
    l2.source = Texture{TextureId{1}, {4, 4}, formats::argb8888};
    draw_layer(c2.surface, l2, argb.pixels, {0, 0, 4, 4});
    CHECK(c2.at(0, 0) == pack(Pixel{0, 0, 0, 255}));

    std::printf("  XRGB composited opaque, ARGB with the same bytes transparent\n");
}

void channel_order_is_respected() {
    std::printf("RGB and BGR are not the same buffer\n");

    // Pure red in ARGB memory order is B=0, G=0, R=255, A=255.
    Buf rgb{2, 2, formats::argb8888, 0x00, 0x00, 0xff, 0xff};
    Canvas c{2, 2};
    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 2, 2};
    draw_layer(c.surface, l, rgb.pixels, {0, 0, 2, 2});
    CHECK(c.at(0, 0) == pack(Pixel{0, 0, 255, 255}));   // red

    // The SAME bytes read as ABGR are blue. Swapping these is the bug that
    // looks like a colour-management problem and is a byte-order one.
    Buf bgr{2, 2, formats::abgr8888, 0x00, 0x00, 0xff, 0xff};
    Canvas c2{2, 2};
    Layer l2 = l;
    l2.source = Texture{TextureId{1}, {2, 2}, formats::abgr8888};
    draw_layer(c2.surface, l2, bgr.pixels, {0, 0, 2, 2});
    CHECK(c2.at(0, 0) == pack(Pixel{255, 0, 0, 255}));   // blue

    std::printf("  the same four bytes are red one way and blue the other\n");
}

void alpha_composites_correctly() {
    std::printf("translucent layers blend, and do not double-darken\n");

    // 50% white, premultiplied as the protocol requires: (128,128,128,128).
    Buf half{2, 2, formats::argb8888, 128, 128, 128, 128};
    Canvas c{2, 2, pack(Pixel{0, 0, 0, 255})};   // black

    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 2, 2};
    draw_layer(c.surface, l, half.pixels, {0, 0, 2, 2});

    // 128 + 0 * (1 - 128/255) = 128, exactly. A renderer that
    // premultiplies a second time gives 64 here, which is the "translucent
    // windows are too dark" bug.
    CHECK(c.at(0, 0) == pack(Pixel{128, 128, 128, 255}));

    // Drawing it twice: 128 + 128 * (1 - 128/255) = 191.75 exactly, and
    // mul255 rounds it to 192. Pinned to the rounded value rather than the
    // truncated 191, because rounding is the property being defended: the
    // truncating version loses a unit per blend, and a stack of translucent
    // windows goes grey.
    draw_layer(c.surface, l, half.pixels, {0, 0, 2, 2});
    const Pixel twice = unpack(c.at(0, 0));
    CHECK(twice.r == 192);

    std::printf("  one pass 128, two passes 192, as the arithmetic says\n");
}

void whole_surface_alpha() {
    std::printf("a layer's own opacity multiplies the source's\n");

    Buf opaque{2, 2, formats::argb8888, 0xff, 0xff, 0xff, 0xff};   // solid white
    Canvas c{2, 2, pack(Pixel{0, 0, 0, 255})};

    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 2, 2};
    l.alpha = 0.5f;   // wp_alpha_modifier
    draw_layer(c.surface, l, opaque.pixels, {0, 0, 2, 2});

    // 0.5 becomes 128/255; white premultiplied by it is 128.
    const Pixel p = unpack(c.at(0, 0));
    CHECK(p.r == 128);
    CHECK(p.a == 255);   // over black, the result is still opaque

    // Zero opacity draws nothing at all, rather than drawing black.
    Canvas c2{2, 2, pack(Pixel{0, 0, 255, 255})};
    Layer invisible = l;
    invisible.alpha = 0.0f;
    CHECK(invisible.invisible());
    draw_layer(c2.surface, invisible, opaque.pixels, {0, 0, 2, 2});
    CHECK(c2.at(0, 0) == pack(Pixel{0, 0, 255, 255}));

    std::printf("  half opacity halves it, zero draws nothing\n");
}

void scaling_samples_at_pixel_centres() {
    std::printf("scaling samples at pixel centres\n");

    // A 2x2 source, each pixel a different blue, scaled 2x to 4x4. Each
    // source pixel should become a clean 2x2 block. A renderer that samples
    // at pixel corners instead of centres shifts the whole image half a
    // pixel, which shows up as one wrong row and column at the edges.
    Buf src{2, 2, formats::argb8888, 0, 0, 0, 0xff};
    src.put(0, 0, 10, 0, 0, 0xff);
    src.put(1, 0, 20, 0, 0, 0xff);
    src.put(0, 1, 30, 0, 0, 0xff);
    src.put(1, 1, 40, 0, 0, 0xff);

    Canvas c{4, 4};
    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 4, 4};
    draw_layer(c.surface, l, src.pixels, {0, 0, 4, 4});

    CHECK(unpack(c.at(0, 0)).b == 10);
    CHECK(unpack(c.at(1, 0)).b == 10);
    CHECK(unpack(c.at(2, 0)).b == 20);
    CHECK(unpack(c.at(3, 0)).b == 20);
    CHECK(unpack(c.at(0, 2)).b == 30);
    CHECK(unpack(c.at(3, 3)).b == 40);

    std::printf("  a 2x2 source scaled 2x gives four clean blocks\n");
}

void transforms_move_the_right_corner() {
    std::printf("transforms put the right pixel in the right corner\n");

    // One marked corner, so a wrong rotation is unmistakable: top-left is
    // 10, everything else 99.
    Buf src{2, 2, formats::argb8888, 99, 0, 0, 0xff};
    src.put(0, 0, 10, 0, 0, 0xff);

    auto corner_of = [&](Transform t) {
        Canvas c{2, 2};
        Layer l;
        l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
        l.dst = {0, 0, 2, 2};
        l.transform = t;
        draw_layer(c.surface, l, src.pixels, {0, 0, 2, 2});
        // Where did the marked pixel end up?
        for (std::int32_t y = 0; y < 2; ++y)
            for (std::int32_t x = 0; x < 2; ++x)
                if (unpack(c.at(x, y)).b == 10) return std::pair<std::int32_t, std::int32_t>{x, y};
        return std::pair<std::int32_t, std::int32_t>{-1, -1};
    };

    // The buffer was drawn rotated, so rendering UNDOES the rotation: a
    // 90-degree buffer's top-left lands at the bottom-left of the output.
    const auto normal = corner_of(Transform::normal);
    const auto half = corner_of(Transform::rotate_180);
    const auto mirror = corner_of(Transform::flipped);
    CHECK(normal.first == 0 && normal.second == 0);
    CHECK(half.first == 1 && half.second == 1);
    CHECK(mirror.first == 1 && mirror.second == 0);

    // Every transform must place it SOMEWHERE: a case that falls through
    // and reads out of range would show up here as (-1,-1).
    for (std::uint8_t i = 0; i < 8; ++i) {
        const auto p = corner_of(static_cast<Transform>(i));
        CHECK(p.first >= 0 && p.second >= 0);
    }

    // And the axis-swapping ones are the four that rotate by 90 or 270.
    static_assert(swaps_axes(Transform::rotate_90));
    static_assert(swaps_axes(Transform::rotate_270));
    static_assert(swaps_axes(Transform::flipped_90));
    static_assert(!swaps_axes(Transform::rotate_180));
    static_assert(!swaps_axes(Transform::normal));

    std::printf("  all eight place the marked pixel, none read out of range\n");
}

void damage_bounds_everything() {
    std::printf("nothing outside the damage is touched\n");

    Canvas c{8, 8, pack(Pixel{0, 0, 0, 255})};
    Buf src{8, 8, formats::argb8888, 0xff, 0xff, 0xff, 0xff};

    Layer l;
    l.source = Texture{TextureId{1}, {8, 8}, formats::argb8888};
    l.dst = {0, 0, 8, 8};   // the layer covers the whole screen

    // ...but only a 2x2 corner is damaged.
    Frame f = one_layer(l, {1, 1, 2, 2});
    f.background = Colour{0, 0, 0, 255};
    const auto painted = render(c.surface, f, [&](const Texture&) { return src.pixels; });

    CHECK(unpack(c.at(1, 1)).r == 255);
    CHECK(unpack(c.at(2, 2)).r == 255);
    // One pixel outside in each direction: untouched. This is the property
    // the whole damage scheme rests on — a compositor redrawing a blinking
    // cursor must not repaint the screen.
    CHECK(c.at(0, 0) == pack(Pixel{0, 0, 0, 255}));
    CHECK(c.at(3, 3) == pack(Pixel{0, 0, 0, 255}));
    CHECK(c.at(7, 7) == pack(Pixel{0, 0, 0, 255}));

    CHECK(painted.area() == 4);

    // An empty damage region means there is nothing to do, and that is a
    // legitimate state: a still screen should cost nothing.
    Frame nothing;
    nothing.layers.push_back(l);
    CHECK(nothing.nothing_to_do());
    Canvas c2{8, 8, 0xdeadbeef};
    const auto none = render(c2.surface, nothing, [&](const Texture&) { return src.pixels; });
    CHECK(none.empty());
    CHECK(c2.at(4, 4) == 0xdeadbeef);

    std::printf("  a 2x2 damage repaints 4 pixels; empty damage repaints none\n");
}

void layers_stack_back_to_front() {
    std::printf("layers draw back to front\n");

    Canvas c{4, 4};
    Frame f;
    f.background = Colour{0, 0, 0, 255};
    f.damage.add({0, 0, 4, 4});

    // Two opaque colours over each other. The LAST one wins, everywhere
    // they overlap — if the order were reversed the first would show.
    Layer back;
    back.source = Colour{255, 0, 0, 255};
    back.dst = {0, 0, 4, 4};
    Layer front;
    front.source = Colour{0, 255, 0, 255};
    front.dst = {2, 2, 2, 2};
    f.layers = {back, front};

    render(c.surface, f, [](const Texture&) { return Pixels{}; });

    CHECK(c.at(0, 0) == pack(Pixel{0, 0, 255, 255}));   // red, only the back layer
    CHECK(c.at(3, 3) == pack(Pixel{0, 255, 0, 255}));   // green, the front one

    std::printf("  the later layer covers the earlier one\n");
}

void a_crop_reads_only_its_part() {
    std::printf("a viewporter crop reads only the part it names\n");

    // Left half red, right half green.
    Buf src{4, 2, formats::argb8888, 0, 0, 0xff, 0xff};
    for (std::int32_t y = 0; y < 2; ++y) {
        src.put(2, y, 0, 0xff, 0, 0xff);
        src.put(3, y, 0, 0xff, 0, 0xff);
    }

    Canvas c{2, 2};
    Layer l;
    l.source = Texture{TextureId{1}, {4, 2}, formats::argb8888};
    l.src = weft::Rect<weft::Buffer>{2, 0, 2, 2};   // the right half only
    l.dst = {0, 0, 2, 2};
    draw_layer(c.surface, l, src.pixels, {0, 0, 2, 2});

    CHECK(c.at(0, 0) == pack(Pixel{0, 255, 0, 255}));
    CHECK(c.at(1, 1) == pack(Pixel{0, 255, 0, 255}));

    // A crop that runs off the end of the buffer is clamped, not read.
    // Clients do send these, and the alternative is reading someone else's
    // memory.
    Canvas c2{2, 2, 0xdeadbeef};
    Layer bad = l;
    bad.src = weft::Rect<weft::Buffer>{100, 100, 4, 4};
    draw_layer(c2.surface, bad, src.pixels, {0, 0, 2, 2});
    CHECK(c2.at(0, 0) == 0xdeadbeef);   // nothing drawn, nothing read

    std::printf("  the named half is drawn; an out-of-range crop draws nothing\n");
}

void y_inverted_buffers() {
    std::printf("a bottom-up buffer is sampled the other way\n");

    // GL clients draw with the origin at the bottom left. Row 0 red,
    // row 1 green.
    Buf src{2, 2, formats::argb8888, 0, 0, 0xff, 0xff};
    src.put(0, 1, 0, 0xff, 0, 0xff);
    src.put(1, 1, 0, 0xff, 0, 0xff);

    Canvas c{2, 2};
    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 2, 2};

    draw_layer(c.surface, l, src.pixels, {0, 0, 2, 2});
    CHECK(unpack(c.at(0, 0)).r == 255);   // top row red

    Pixels flipped = src.pixels;
    flipped.y_invert = true;
    Canvas c2{2, 2};
    draw_layer(c2.surface, l, flipped, {0, 0, 2, 2});
    CHECK(unpack(c2.at(0, 0)).g == 255);   // top row now green

    std::printf("  inverting swaps the rows, with no copy\n");
}

void degenerate_inputs_do_nothing() {
    std::printf("nonsense draws nothing rather than crashing\n");

    Buf src{2, 2, formats::argb8888, 0xff, 0xff, 0xff, 0xff};
    Layer l;
    l.source = Texture{TextureId{1}, {2, 2}, formats::argb8888};
    l.dst = {0, 0, 2, 2};

    // A surface with no memory. A compositor hits this between allocating a
    // buffer and the allocation succeeding.
    Surface none;
    CHECK(!none.valid());
    draw_layer(none, l, src.pixels, {0, 0, 2, 2});

    // A stride too small for the width: the description is inconsistent, so
    // reading it would walk diagonally through memory.
    Canvas c{4, 4, 0xdeadbeef};
    Pixels bad = src.pixels;
    bad.stride_bytes = 1;
    CHECK(!bad.valid());
    draw_layer(c.surface, l, bad, {0, 0, 4, 4});
    CHECK(c.at(0, 0) == 0xdeadbeef);

    // A destination entirely off-screen.
    Layer far = l;
    far.dst = {1000, 1000, 10, 10};
    draw_layer(c.surface, far, src.pixels, {0, 0, 4, 4});
    CHECK(c.at(0, 0) == 0xdeadbeef);

    // An empty destination rectangle.
    Layer empty = l;
    empty.dst = {0, 0, 0, 0};
    CHECK(empty.invisible());

    std::printf("  no memory, bad stride, off-screen: all no-ops\n");
}

}  // namespace

void run_cpu_tests() {
    x_formats_are_opaque();
    channel_order_is_respected();
    alpha_composites_correctly();
    whole_surface_alpha();
    scaling_samples_at_pixel_centres();
    transforms_move_the_right_corner();
    damage_bounds_everything();
    layers_stack_back_to_front();
    a_crop_reads_only_its_part();
    y_inverted_buffers();
    degenerate_inputs_do_nothing();
}
