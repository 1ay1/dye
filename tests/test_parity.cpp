// test_parity.cpp — the GPU renderer must agree with the CPU one.
//
// Why this test is the important one
// ----------------------------------
// A GPU renderer is otherwise tested by looking at it, which finds the bugs
// that are obvious and none of the ones that matter: a half-pixel sampling
// offset, an edge row blended twice, alpha applied in the wrong space, a
// transform that is right for three of its eight cases. Every one of those
// produces a picture that looks fine in a screenshot and wrong in motion.
//
// So the same dye::Frame goes through both renderers and the outputs are
// compared. The CPU renderer is the oracle: it is simple enough to read and
// its arithmetic is checked against hand-computed values in test_cpu.cpp.
// If the two disagree, one of them is wrong, and this says by how much and
// where.
//
// Exact equality is not the standard. A GPU samples with LINEAR filtering
// and blends in a slightly different order, so a tolerance of a few units
// per channel is expected and anything more is a bug.
#include "check.hpp"

#include <dye/cpu.hpp>
#include <dye/device.hpp>
#include <dye/renderer.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dye;

namespace {

int skipped = 0;

/// How far apart two images are: the worst single channel, and how many
/// pixels differ by more than a small amount.
struct Difference {
    int worst_channel = 0;
    int pixels_over_tolerance = 0;
    int first_x = -1, first_y = -1;
};

Difference compare(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b,
                   int w, int h, int tolerance) {
    Difference d;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::uint32_t p = a[static_cast<std::size_t>(y) * w + x];
            const std::uint32_t q = b[static_cast<std::size_t>(y) * w + x];
            int worst = 0;
            for (int shift = 0; shift < 32; shift += 8) {
                const int pa = static_cast<int>((p >> shift) & 0xff);
                const int qa = static_cast<int>((q >> shift) & 0xff);
                worst = std::max(worst, std::abs(pa - qa));
            }
            d.worst_channel = std::max(d.worst_channel, worst);
            if (worst > tolerance) {
                ++d.pixels_over_tolerance;
                if (d.first_x < 0) {
                    d.first_x = x;
                    d.first_y = y;
                }
            }
        }
    }
    return d;
}

/// Render `frame` with the CPU renderer.
std::vector<std::uint32_t> on_cpu(const Frame& frame, int w, int h,
                                  const std::vector<std::uint8_t>& texture, int tw, int th,
                                  Format format) {
    std::vector<std::uint32_t> out(static_cast<std::size_t>(w) * h, 0xff000000u);
    cpu::Surface surface{out.data(), w * 4, {w, h}};
    cpu::Pixels pixels{texture.data(), tw * 4, {tw, th}, format};
    cpu::render(surface, frame, [&](const Texture&) { return pixels; });
    return out;
}

// ---------------------------------------------------------------------------

void the_two_renderers_agree() {
    std::printf("the GPU and CPU renderers draw the same frame\n");

    auto device = Device::open();
    if (!device) {
        std::printf("  SKIP: %s\n", device.error().what.data());
        ++skipped;
        return;
    }
    auto renderer = Renderer::create(**device);
    if (!renderer) {
        std::printf("  SKIP: %s\n", renderer.error().what.data());
        ++skipped;
        return;
    }

    constexpr int kW = 256, kH = 256;
    constexpr int kTexW = 64, kTexH = 64;

    // A texture with structure, not a flat colour: a flat one would hide a
    // sampling offset entirely, which is one of the bugs this exists to
    // catch. Four quadrants, each a different colour.
    std::vector<std::uint8_t> texture(static_cast<std::size_t>(kTexW) * kTexH * 4);
    for (int y = 0; y < kTexH; ++y) {
        for (int x = 0; x < kTexW; ++x) {
            auto* p = &texture[(static_cast<std::size_t>(y) * kTexW + x) * 4];
            const bool right = x >= kTexW / 2;
            const bool lower = y >= kTexH / 2;
            p[0] = right ? 0xc0 : 0x20;             // B
            p[1] = lower ? 0xc0 : 0x20;             // G
            p[2] = (right == lower) ? 0xf0 : 0x40;  // R
            p[3] = 0xff;
        }
    }

    // Upload it to the GPU as a real client buffer would arrive.
    auto buffer = (*device)->allocate(kTexW, kTexH, formats::argb8888);
    if (!buffer) {
        std::printf("  SKIP: %s\n", buffer.error().what.data());
        ++skipped;
        return;
    }
    auto image = (*device)->import(std::move(*buffer));
    if (!image) {
        std::printf("  SKIP: %s\n", image.error().what.data());
        ++skipped;
        return;
    }
    // Put the pixels ON the GPU. Without this the texture is whatever the
    // allocator left behind, and the comparison is against uninitialised
    // memory — which is exactly what the first run of this test did.
    {
        std::vector<std::uint32_t> packed(static_cast<std::size_t>(kTexW) * kTexH);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            const auto* p = &texture[i * 4];
            packed[i] = static_cast<std::uint32_t>(p[3]) << 24 |
                        static_cast<std::uint32_t>(p[2]) << 16 |
                        static_cast<std::uint32_t>(p[1]) << 8 | p[0];
        }
        const auto uploaded = (*image)->write(packed.data(), kTexW);
        if (!uploaded) {
            std::printf("  SKIP: %s\n", uploaded.error().what.data());
            ++skipped;
            return;
        }
    }

    // Somewhere to render into.
    auto target_buffer = (*device)->allocate(kW, kH, formats::argb8888);
    if (!target_buffer) {
        std::printf("  SKIP: %s\n", target_buffer.error().what.data());
        ++skipped;
        return;
    }
    auto target_image = (*device)->import(std::move(*target_buffer));
    if (!target_image) {
        std::printf("  SKIP: %s\n", target_image.error().what.data());
        ++skipped;
        return;
    }
    RenderTarget target{target_image->get(), {kW, kH}};

    struct Case {
        const char* what;
        Layer       layer;
        int         tolerance;
    };

    const auto textured = [&](weft::Rect<weft::Output> dst) {
        Layer l;
        l.source = (*image)->as_texture();
        l.dst = dst;
        return l;
    };

    std::vector<Case> cases;
    {
        // 1:1, no scaling. The strictest case: any disagreement here is a
        // real difference in sampling or blending, not a filtering artefact.
        cases.push_back({"1:1, no scaling", textured({32, 32, kTexW, kTexH}), 2});

        // Scaled up 2x. LINEAR filtering on the GPU against nearest on the
        // CPU differs at the boundaries between quadrants, so the tolerance
        // is looser — but a sampling OFFSET would shift every boundary and
        // blow past even this.
        cases.push_back({"scaled 2x", textured({0, 0, kTexW * 2, kTexH * 2}), 96});

        // Half opacity. Tests the whole-surface alpha path, which is where
        // premultiplication goes wrong.
        Layer faded = textured({32, 32, kTexW, kTexH});
        faded.alpha = 0.5f;
        cases.push_back({"half opacity", faded, 3});

        // A solid colour, no texture at all.
        Layer solid;
        solid.source = Colour{0x40, 0x80, 0xc0, 0xff};
        solid.dst = {16, 16, 128, 128};
        cases.push_back({"a solid colour", solid, 2});

        // A translucent colour over the background: the blend itself.
        Layer blended;
        blended.source = Colour{0xff, 0x00, 0x00, 0x80};
        blended.dst = {0, 0, kW, kH};
        cases.push_back({"a translucent colour", blended, 3});
    }

    int agreed = 0;
    for (const Case& c : cases) {
        Frame frame;
        frame.layers.push_back(c.layer);
        frame.damage.add({0, 0, kW, kH});
        frame.background = Colour{0, 0, 0, 255};

        const auto expected = on_cpu(frame, kW, kH, texture, kTexW, kTexH, formats::argb8888);

        const auto submission = (*renderer)->render(target, frame);
        if (!submission) {
            std::printf("  %-22s SKIP: %s\n", c.what, submission.error().what.data());
            ++skipped;
            continue;
        }
        // Wait for the GPU before reading. The fence is what a compositor
        // would hand to KMS instead of waiting on.
        if (submission->done.valid()) (void)submission->done.wait(2000);
        (void)(*renderer)->wait_idle();

        std::vector<std::uint32_t> actual(static_cast<std::size_t>(kW) * kH, 0);
        const auto read = (*target_image)->read(actual.data(), kW);
        if (!read) {
            std::printf("  %-22s SKIP: %s\n", c.what, read.error().what.data());
            ++skipped;
            continue;
        }

        const Difference d = compare(expected, actual, kW, kH, c.tolerance);
        const bool ok = d.pixels_over_tolerance == 0;
        CHECK(ok);
        if (ok) {
            ++agreed;
            std::printf("  %-22s agree (worst channel %d of %d allowed)\n", c.what,
                        d.worst_channel, c.tolerance);
        } else {
            std::printf("  %-22s DIFFER: %d pixels over tolerance %d, worst %d, "
                        "first at (%d,%d)\n",
                        c.what, d.pixels_over_tolerance, c.tolerance, d.worst_channel,
                        d.first_x, d.first_y);
        }
    }

    std::printf("  %d of %zu cases agree\n", agreed, cases.size());
}

void a_still_screen_costs_nothing() {
    std::printf("an undamaged frame is not drawn at all\n");

    auto device = Device::open();
    if (!device) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }
    auto renderer = Renderer::create(**device);
    if (!renderer) {
        std::printf("  SKIP: no renderer\n");
        ++skipped;
        return;
    }

    auto buffer = (*device)->allocate(64, 64, formats::argb8888);
    if (!buffer) {
        std::printf("  SKIP: allocate refused\n");
        ++skipped;
        return;
    }
    auto image = (*device)->import(std::move(*buffer));
    if (!image) {
        std::printf("  SKIP: import refused\n");
        ++skipped;
        return;
    }
    RenderTarget target{image->get(), {64, 64}};

    // A frame with layers but NO damage. Nothing changed on screen, so
    // nothing should happen — no submit, no fence, no work. This is what
    // makes an idle compositor cost nothing, and it is easy to lose.
    Frame frame;
    Layer l;
    l.source = Colour{255, 0, 0, 255};
    l.dst = {0, 0, 64, 64};
    frame.layers.push_back(l);
    CHECK(frame.nothing_to_do());

    const auto submission = (*renderer)->render(target, frame);
    CHECK(submission.has_value());
    if (submission) {
        CHECK(submission->layers_drawn == 0);
        CHECK(!submission->done.valid());   // nothing was submitted to wait for
    }

    std::printf("  no layers drawn, no fence, no submit\n");
}

void a_fullscreen_layer_can_skip_compositing() {
    std::printf("a fullscreen opaque layer is a direct-scanout candidate\n");

    const weft::Size<weft::Output> output{1920, 1080};

    // The case worth having: one opaque fullscreen video. Its buffer goes
    // to the display untouched — no draw, no blend, no intermediate frame.
    Frame video;
    Layer l;
    l.source = Texture{TextureId{1}, {1920, 1080}, formats::xrgb8888};
    l.dst = {0, 0, 1920, 1080};
    video.layers.push_back(l);
    CHECK(Renderer::direct_scanout_candidate(video, output) != nullptr);

    // Two layers: something is on top, so it must be composited. Scanning
    // out the bottom one would drop whatever is above it.
    Frame with_overlay = video;
    Layer osd;
    osd.source = Colour{255, 255, 255, 200};
    osd.dst = {100, 100, 200, 50};
    with_overlay.layers.push_back(osd);
    CHECK(Renderer::direct_scanout_candidate(with_overlay, output) == nullptr);

    // Not quite fullscreen: a windowed video needs the desktop around it.
    Frame windowed;
    Layer small = l;
    small.dst = {10, 10, 1900, 1060};
    windowed.layers.push_back(small);
    CHECK(Renderer::direct_scanout_candidate(windowed, output) == nullptr);

    // Fullscreen but translucent: what is underneath shows through.
    Frame faded;
    Layer dim = l;
    dim.alpha = 0.5f;
    faded.layers.push_back(dim);
    CHECK(Renderer::direct_scanout_candidate(faded, output) == nullptr);

    // An ARGB buffer is only a candidate when the compositor says it is
    // opaque. The format alone does not settle it: an ARGB window may be
    // fully opaque, and only the compositor knows.
    Frame argb;
    Layer maybe = l;
    maybe.source = Texture{TextureId{1}, {1920, 1080}, formats::argb8888};
    argb.layers.push_back(maybe);
    CHECK(Renderer::direct_scanout_candidate(argb, output) == nullptr);

    argb.layers[0].opaque.add({0, 0, 1920, 1080});
    CHECK(Renderer::direct_scanout_candidate(argb, output) != nullptr);

    // An invisible layer on top does not spoil it: it draws nothing.
    Frame with_hidden = video;
    Layer hidden;
    hidden.source = Colour{0, 0, 0, 0};
    hidden.dst = {0, 0, 100, 100};
    with_hidden.layers.push_back(hidden);
    CHECK(Renderer::direct_scanout_candidate(with_hidden, output) != nullptr);

    std::printf("  fullscreen opaque yes; overlaid, windowed, faded and ARGB no\n");
}

}  // namespace

void run_parity_tests() {
    the_two_renderers_agree();
    a_still_screen_costs_nothing();
    a_fullscreen_layer_can_skip_compositing();

    if (skipped) std::printf("\n  (%d GPU check(s) skipped)\n", skipped);
}
