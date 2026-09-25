// bench_upload.cpp — what a shared-memory client costs.
//
// The counterpart to bench_readback.cpp, and for the same reason: the cost
// is invisible to every correctness test. write() produces perfect pixels
// whether it takes 0.5 ms or 15.
//
// This is the hot path, whatever device.hpp's comment says. A compositor
// uploads a texture for every shared-memory surface that changed, EVERY
// COMMIT — and most clients are shared-memory clients: foot, and every GTK
// app. A terminal scrolling at 60 Hz calls this 60 times a second; three of
// them, three times that.
//
// So the number that matters is not "is it fast enough once", it is "how
// many of these fit in a frame". At 60 Hz there are 16.7 ms for compositing
// AND uploading AND flipping. If one 1440p upload eats 12 of them, then a
// second busy window cannot fit at all, and the compositor visibly stalls
// exactly when the screen is changing most — which is the worst possible
// time, and the one a benchmark with one surface never shows.
#include <dye/device.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using namespace std::chrono;

namespace {

double ms_per(int n, const std::function<void()>& f) {
    f();   // warm: the first call allocates the staging buffer, later ones reuse it
    const auto t0 = steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    const auto t1 = steady_clock::now();
    return duration<double, std::milli>(t1 - t0).count() / n;
}

}  // namespace

int main() {
    std::printf("dye upload cost (the shared-memory client path)\n\n");

    auto device = dye::Device::open();
    if (!device) {
        std::printf("  SKIP: %s\n", device.error().what.data());
        return 0;
    }
    std::printf("  %s\n\n", (*device)->info().name.c_str());

    struct Case {
        const char* what;
        std::int32_t w, h;
    };
    const Case cases[] = {
        {"a terminal  (800x600)", 800, 600},
        {"720p  (a window)", 1280, 720},
        {"1080p (a big window)", 1920, 1080},
        {"1440p (fullscreen)", 3440, 1440},
    };

    // 60 Hz. Everything — compositing, uploading, flipping — has to fit.
    constexpr double kFrameMs = 16.7;

    for (const Case& c : cases) {
        const std::size_t px = static_cast<std::size_t>(c.w) * c.h;
        std::vector<std::uint32_t> src(px, 0xff336699);
        std::vector<std::uint32_t> dst(px);

        // The floor: what it costs to touch the bytes at all.
        const double floor_ms = ms_per(20, [&] { std::memcpy(dst.data(), src.data(), px * 4); });

        auto buf = (*device)->allocate(c.w, c.h, dye::formats::argb8888);
        if (!buf) {
            std::printf("  %-24s SKIP: %s\n", c.what, buf.error().what.data());
            continue;
        }
        auto image = (*device)->import(std::move(*buf));
        if (!image) {
            std::printf("  %-24s SKIP: %s\n", c.what, image.error().what.data());
            continue;
        }

        const double up_ms = ms_per(20, [&] { (void)(*image)->write(src.data(), c.w); });

        // The question a compositor actually asks: how many busy windows fit?
        const double fit = up_ms > 0 ? kFrameMs / up_ms : 0;
        std::printf("  %-24s %6.2f ms  (memcpy floor %.2f, %4.1f%% of a frame, ~%.0f fit)%s\n",
                    c.what, up_ms, floor_ms, up_ms / kFrameMs * 100.0, fit,
                    up_ms > kFrameMs / 4 ? "  <-- too slow" : "");

        // And the case that is actually typical: a terminal scrolls, so one
        // band of rows changed and the rest did not. A compositor that
        // ignores wl_surface.damage pays the full number above for this.
        const std::int32_t band = c.h / 10;
        const double dmg_ms = ms_per(20, [&] { (void)(*image)->write_rows(src.data(), c.w, 0, band); });
        std::printf("  %-24s %6.2f ms  (10%% damaged: %.1fx cheaper)\n", "  ^ 10% of it",
                    dmg_ms, dmg_ms > 0 ? up_ms / dmg_ms : 0);
    }

    std::printf("\n  Every number above is paid on the compositor's LOOP THREAD,\n");
    std::printf("  because write() submits and then waits for the GPU. While it\n");
    std::printf("  waits, no client request is read and no key is delivered.\n");
    return 0;
}
