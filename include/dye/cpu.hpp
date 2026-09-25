#pragma once
// dye/cpu.hpp — the reference renderer.
//
// Two jobs, and the second is the important one.
//
// 1. It is the fallback. A compositor in a VM, in CI, or on a machine whose
//    driver refused to load still has to put pixels on a screen.
//
// 2. It is the ORACLE. Everything a GPU renderer does — sampling, transforms,
//    premultiplied blending, damage clipping — is done here in a form that
//    can be read and reasoned about, so the GL backend can be checked against
//    it pixel for pixel (tests/test_parity.cpp). Without that, a GPU renderer
//    is tested by looking at it, which finds the bugs that are obvious and
//    none of the ones that matter: a half-pixel sampling offset, an edge
//    row blended twice, alpha applied in the wrong space.
//
// So this is written for clarity over speed. It is a straightforward
// per-pixel loop, and the comments explain the arithmetic rather than the
// optimisations, because when the two renderers disagree this is the one
// that has to be believed.
//
// No GPU headers, no allocation per frame, and pure apart from the
// destination: testable anywhere.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>

#include "draw.hpp"

namespace dye::cpu {

// ---------------------------------------------------------------------------
// Pixels.
//
// One internal representation: premultiplied ARGB32, host byte order. Every
// source format converts to it on the way in and the destination converts on
// the way out, so the blend itself is written once instead of once per format
// pair — which is where format-handling bugs actually come from.
// ---------------------------------------------------------------------------

/// Premultiplied ARGB, one pixel. `a` is the alpha, and r/g/b have ALREADY
/// been multiplied by it.
///
/// Premultiplied because that is what makes compositing associative: you can
/// blend A over B, then that over C, and get the same answer as blending B
/// over C first. With straight alpha you cannot, and a compositor blends in
/// whatever order the window stack happens to be.
struct Pixel {
    std::uint8_t b = 0, g = 0, r = 0, a = 0;

    friend constexpr bool operator==(Pixel, Pixel) = default;
};

static_assert(sizeof(Pixel) == 4);

/// Pack to the 0xAARRGGBB an ARGB32 framebuffer wants.
[[nodiscard]] constexpr std::uint32_t pack(Pixel p) noexcept {
    return static_cast<std::uint32_t>(p.a) << 24 | static_cast<std::uint32_t>(p.r) << 16 |
           static_cast<std::uint32_t>(p.g) << 8 | static_cast<std::uint32_t>(p.b);
}

[[nodiscard]] constexpr Pixel unpack(std::uint32_t v) noexcept {
    return Pixel{static_cast<std::uint8_t>(v),
                 static_cast<std::uint8_t>(v >> 8),
                 static_cast<std::uint8_t>(v >> 16),
                 static_cast<std::uint8_t>(v >> 24)};
}

/// Multiply two 0..255 values as if they were 0..1, rounding correctly.
///
/// The naive `(a * b) / 255` is wrong at the top: 255 * 255 / 255 is 255 by
/// luck, but 128 * 255 / 255 truncates elsewhere in the range. This is the
/// standard rounding trick, and it matters because a renderer that darkens
/// by one unit per blend turns a stack of translucent windows grey.
[[nodiscard]] constexpr std::uint8_t mul255(std::uint32_t a, std::uint32_t b) noexcept {
    const std::uint32_t t = a * b + 128;
    return static_cast<std::uint8_t>((t + (t >> 8)) >> 8);
}

/// A straight (non-premultiplied) colour, premultiplied.
[[nodiscard]] constexpr Pixel premultiply(Colour c) noexcept {
    return Pixel{mul255(c.b, c.a), mul255(c.g, c.a), mul255(c.r, c.a), c.a};
}

/// Read one pixel out of a client's buffer, in that buffer's format, and
/// return it premultiplied.
///
/// This is where an X format's fourth byte is thrown away. A client may
/// leave anything there, and treating it as alpha renders the window full of
/// holes — the bug that hides because most allocators happen to leave 0xff.
[[nodiscard]] inline Pixel read_pixel(const std::uint8_t* src, Format f) noexcept {
    // Wayland's formats are named in big-endian-ish channel order but stored
    // little-endian, so ARGB8888 is B, G, R, A in memory.
    const std::uint8_t c0 = src[0], c1 = src[1], c2 = src[2], c3 = src[3];
    const bool rgb = is_rgb(f.order());
    const std::uint8_t r = rgb ? c2 : c0;
    const std::uint8_t g = c1;
    const std::uint8_t b = rgb ? c0 : c2;
    if (!f.has_alpha()) return Pixel{b, g, r, 255};
    // A client's ARGB buffer is already premultiplied: that is what the
    // Wayland protocol says wl_shm's argb8888 means.
    return Pixel{b, g, r, c3};
}

/// Source-over, both sides premultiplied.
///
///     out = src + dst * (1 - src.a)
///
/// The opaque and fully-transparent cases are separated out because they are
/// the overwhelming majority of pixels in a real frame and the arithmetic is
/// identical either way.
[[nodiscard]] constexpr Pixel over(Pixel src, Pixel dst) noexcept {
    if (src.a == 255) return src;
    if (src.a == 0) return dst;
    const std::uint32_t inv = 255u - src.a;
    return Pixel{static_cast<std::uint8_t>(src.b + mul255(dst.b, inv)),
                 static_cast<std::uint8_t>(src.g + mul255(dst.g, inv)),
                 static_cast<std::uint8_t>(src.r + mul255(dst.r, inv)),
                 static_cast<std::uint8_t>(src.a + mul255(dst.a, inv))};
}

/// Scale a premultiplied pixel by a whole-surface opacity.
[[nodiscard]] constexpr Pixel scale_alpha(Pixel p, std::uint8_t alpha) noexcept {
    if (alpha == 255) return p;
    return Pixel{mul255(p.b, alpha), mul255(p.g, alpha), mul255(p.r, alpha),
                 mul255(p.a, alpha)};
}

// ---------------------------------------------------------------------------
// Sampling: which source pixel does a destination pixel come from?
//
// One function, used by both renderers, because this is precisely where they
// drift apart. Nearest-neighbour with a half-pixel offset: destination pixel
// centres map into the source, which is what avoids the systematic
// half-pixel shift that makes a GPU and a CPU renderer disagree along every
// edge.
// ---------------------------------------------------------------------------

/// Where destination pixel (dx, dy) reads from, in the source rectangle's
/// own coordinates, before any transform.
///
/// Returns buffer-space coordinates, already offset by the source rect's
/// origin. `dx`/`dy` are relative to the destination rect's origin.
struct SamplePoint {
    std::int32_t x = 0;
    std::int32_t y = 0;
};

[[nodiscard]] constexpr SamplePoint sample_point(std::int32_t dx, std::int32_t dy,
                                                 std::int32_t dst_w, std::int32_t dst_h,
                                                 weft::Rect<weft::Buffer> src,
                                                 Transform t) noexcept {
    if (dst_w <= 0 || dst_h <= 0) return {src.origin.x, src.origin.y};

    // The transform is applied FIRST: the client drew rotated, so undo that
    // to get back into the buffer's own axes. Then scale. Doing it the other
    // way round scales along the wrong axis whenever the transform swaps
    // them, which is the "rotated window is squashed" bug.
    std::int32_t ux = dx, uy = dy;
    std::int32_t uw = dst_w, uh = dst_h;

    if (swaps_axes(t)) {
        std::swap(ux, uy);
        std::swap(uw, uh);
    }

    switch (t) {
        case Transform::normal:
        case Transform::flipped:
            break;
        case Transform::rotate_90:
        case Transform::flipped_90:
            uy = uh - 1 - uy;
            break;
        case Transform::rotate_180:
        case Transform::flipped_180:
            ux = uw - 1 - ux;
            uy = uh - 1 - uy;
            break;
        case Transform::rotate_270:
        case Transform::flipped_270:
            ux = uw - 1 - ux;
            break;
    }
    if (is_flipped(t)) ux = uw - 1 - ux;

    // Nearest neighbour, sampling at destination pixel CENTRES:
    //   sx = (ux + 0.5) * src.w / uw
    // in integer arithmetic. The +0.5 is what keeps a 2x upscale from
    // shifting the whole image half a pixel left.
    const std::int64_t sx =
        (static_cast<std::int64_t>(ux) * 2 + 1) * src.size.width / (static_cast<std::int64_t>(uw) * 2);
    const std::int64_t sy =
        (static_cast<std::int64_t>(uy) * 2 + 1) * src.size.height / (static_cast<std::int64_t>(uh) * 2);

    return {src.origin.x + static_cast<std::int32_t>(sx),
            src.origin.y + static_cast<std::int32_t>(sy)};
}

// ---------------------------------------------------------------------------
// The surface being drawn into.
// ---------------------------------------------------------------------------

/// Somebody else's memory: a dumb buffer, a wl_shm pool, a test's array.
/// Borrowed, never owned, because the thing that owns a framebuffer is the
/// display and it outlives any one frame.
struct Surface {
    std::uint32_t* pixels = nullptr;
    /// Bytes per row. NOT pixels: every framebuffer API reports it in bytes
    /// and a renderer that assumes otherwise draws a diagonal.
    std::int32_t stride_bytes = 0;
    weft::Size<weft::Output> size{};

    [[nodiscard]] bool valid() const noexcept {
        return pixels != nullptr && !size.empty() &&
               stride_bytes >= size.width * 4;
    }

    [[nodiscard]] std::uint32_t* row(std::int32_t y) const noexcept {
        return reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(pixels) +
                                                static_cast<std::ptrdiff_t>(y) * stride_bytes);
    }
};

/// A client's pixels in main memory, for the CPU renderer to sample.
///
/// The GPU renderer takes a TextureId instead; this is the same picture,
/// read back or never uploaded. `data` is borrowed.
struct Pixels {
    const std::uint8_t* data = nullptr;
    std::int32_t stride_bytes = 0;
    weft::Size<weft::Buffer> size{};
    Format format{};
    bool y_invert = false;

    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && !size.empty() && format.valid() &&
               stride_bytes >= size.width * static_cast<std::int32_t>(format.bytes_per_pixel());
    }

    [[nodiscard]] const std::uint8_t* row(std::int32_t y) const noexcept {
        const std::int32_t yy = y_invert ? size.height - 1 - y : y;
        return data + static_cast<std::ptrdiff_t>(yy) * stride_bytes;
    }
};

// ---------------------------------------------------------------------------
// Drawing.
// ---------------------------------------------------------------------------

/// Fill a rectangle with a premultiplied colour, blending.
inline void fill(const Surface& dst, weft::Rect<weft::Output> area, Pixel colour) {
    if (!dst.valid()) return;
    const auto clipped =
        area.intersected(weft::Rect<weft::Output>{0, 0, dst.size.width, dst.size.height});
    if (clipped.empty()) return;

    for (std::int32_t y = clipped.top(); y < clipped.bottom(); ++y) {
        std::uint32_t* out = dst.row(y);
        if (colour.a == 255) {
            // Opaque: no read, no blend. Worth separating because a
            // compositor's background is most of the screen most of the time.
            const std::uint32_t v = pack(colour);
            std::fill(out + clipped.left(), out + clipped.right(), v);
        } else {
            for (std::int32_t x = clipped.left(); x < clipped.right(); ++x)
                out[x] = pack(over(colour, unpack(out[x])));
        }
    }
}

/// Draw one textured layer, clipped to `clip`.
///
/// `src_pixels` is what the layer's Texture refers to; the CPU renderer
/// cannot read a GPU texture, so the caller supplies the memory. That is
/// the seam: a GPU backend binds a texture here, and everything else about
/// the frame is identical.
inline void draw_layer(const Surface& dst, const Layer& layer, const Pixels& src_pixels,
                       weft::Rect<weft::Output> clip) {
    if (!dst.valid() || !src_pixels.valid() || layer.invisible()) return;

    const auto area = layer.dst.intersected(clip).intersected(
        weft::Rect<weft::Output>{0, 0, dst.size.width, dst.size.height});
    if (area.empty()) return;

    // The part of the source to read. Unset means all of it, which is the
    // common case; a set one comes from wp_viewporter's crop.
    weft::Rect<weft::Buffer> src =
        layer.src.value_or(weft::Rect<weft::Buffer>{0, 0, src_pixels.size.width,
                                                    src_pixels.size.height});
    // A crop the client got wrong must not read outside the buffer.
    src = src.intersected(
        weft::Rect<weft::Buffer>{0, 0, src_pixels.size.width, src_pixels.size.height});
    if (src.empty()) return;

    const auto alpha = static_cast<std::uint8_t>(
        std::clamp(layer.alpha, 0.0f, 1.0f) * 255.0f + 0.5f);
    const std::int32_t dst_w = layer.dst.size.width;
    const std::int32_t dst_h = layer.dst.size.height;
    const std::uint32_t bpp = src_pixels.format.bytes_per_pixel();

    for (std::int32_t y = area.top(); y < area.bottom(); ++y) {
        std::uint32_t* out = dst.row(y);
        const std::int32_t dy = y - layer.dst.origin.y;
        for (std::int32_t x = area.left(); x < area.right(); ++x) {
            const std::int32_t dx = x - layer.dst.origin.x;
            const auto s = sample_point(dx, dy, dst_w, dst_h, src, layer.transform);
            // Clamped rather than wrapped: rounding at the last row must not
            // read the first one, which shows as a one-pixel stripe of the
            // wrong colour along an edge.
            const std::int32_t sx = std::clamp(s.x, 0, src_pixels.size.width - 1);
            const std::int32_t sy = std::clamp(s.y, 0, src_pixels.size.height - 1);

            const std::uint8_t* p = src_pixels.row(sy) + static_cast<std::ptrdiff_t>(sx) * bpp;
            const Pixel texel = scale_alpha(read_pixel(p, src_pixels.format), alpha);
            out[x] = pack(over(texel, unpack(out[x])));
        }
    }
}

/// Draw a whole frame.
///
/// `pixels_for` maps a layer's texture to its memory: the caller owns the
/// buffers, and the renderer does not care where they came from. Returns the
/// region it actually wrote, which is what a backend hands to the display as
/// damage.
template <class PixelsFor>
weft::Region<weft::Output> render(const Surface& dst, const Frame& frame,
                                  PixelsFor&& pixels_for) {
    weft::Region<weft::Output> painted;
    if (!dst.valid() || frame.nothing_to_do()) return painted;

    const Pixel background = premultiply(frame.background);

    // Once per damage rectangle rather than once per layer: a layer that
    // touches three damage rectangles is drawn three times, each clipped,
    // and no pixel outside the damage is written. That is the property the
    // whole scheme rests on — a compositor redrawing one blinking cursor
    // must not repaint the screen.
    for (const auto& rect : frame.damage.rects()) {
        const auto area =
            rect.intersected(weft::Rect<weft::Output>{0, 0, dst.size.width, dst.size.height});
        if (area.empty()) continue;

        fill(dst, area, background);

        for (const Layer& layer : frame.layers) {
            if (layer.invisible() || !layer.dst.intersects(area)) continue;
            if (const auto* c = std::get_if<Colour>(&layer.source)) {
                const auto alpha = static_cast<std::uint8_t>(
                    std::clamp(layer.alpha, 0.0f, 1.0f) * 255.0f + 0.5f);
                fill(dst, layer.dst.intersected(area), scale_alpha(premultiply(*c), alpha));
            } else {
                const auto& tex = std::get<Texture>(layer.source);
                const Pixels src = pixels_for(tex);
                if (src.valid()) draw_layer(dst, layer, src, area);
            }
        }
        painted.add(area);
    }
    return painted;
}

}  // namespace dye::cpu
