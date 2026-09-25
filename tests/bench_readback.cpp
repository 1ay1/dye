// bench_readback.cpp — what a frame costs.
//
// This is not a test of correctness. It exists because the most expensive
// mistake in this library was invisible to every correctness test: reading
// back a 1080p frame took 637 ms, and the pixels were perfectly correct.
//
// The cause was one memory property. Host-visible memory on a discrete GPU
// is write-combined by default — fine to write, catastrophic to READ,
// because every access crosses PCIe uncached. Asking for HOST_CACHED as
// well took the same operation to 5.6 ms. Both produce identical output, so
// only a number on a clock could tell them apart.
//
// Hence a benchmark that prints a budget. A compositor at 60 Hz has 16.7 ms
// for everything; if one buffer read eats 637 of them, a video plays at two
// frames a second and the reason is nowhere in the code.
#include <dye/device.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using namespace std::chrono;

namespace {

double ms_per(int n, const std::function<void()>& f) {
    f();   // warm: the first call allocates, later ones should not
    const auto t0 = steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    const auto t1 = steady_clock::now();
    return duration<double, std::milli>(t1 - t0).count() / n;
}

}  // namespace

int main() {
    std::printf("dye read-back cost\n\n");

    auto device = dye::Device::open();
    if (!device) {
        std::printf("  SKIP: %s\n", device.error().what.data());
        return 0;
    }
    std::printf("  %s\n\n", (*device)->info().name.c_str());

    struct Case {
        const char* what;
        std::int32_t w, h;
        double budget_ms;   // the frame time this size is usually drawn at
    };
    const Case cases[] = {
        {"720p  (a window)", 1280, 720, 16.7},
        {"1080p (a video)", 1920, 1080, 41.7},   // 23.976 fps
        {"1440p (this screen)", 3440, 1440, 16.7},
    };

    // The floor: what it costs the CPU just to touch that many bytes. A
    // read-back can never be faster, and how close it gets is the real
    // measure — "5.6 ms" means nothing without "0.5 ms is the floor".
    for (const Case& c : cases) {
        const std::size_t px = static_cast<std::size_t>(c.w) * c.h;
        std::vector<std::uint32_t> a(px), b(px);
        const double floor_ms = ms_per(20, [&] { std::memcpy(b.data(), a.data(), px * 4); });

        auto buf = (*device)->allocate(c.w, c.h, dye::formats::argb8888);
        if (!buf) {
            std::printf("  %-22s SKIP: %s\n", c.what, buf.error().what.data());
            continue;
        }
        auto image = (*device)->import(std::move(*buf));
        if (!image) {
            std::printf("  %-22s SKIP: %s\n", c.what, image.error().what.data());
            continue;
        }

        const double read_ms =
            ms_per(20, [&] { (void)(*image)->read(b.data(), c.w); });

        const double pct = read_ms / c.budget_ms * 100.0;
        std::printf("  %-22s %6.2f ms  (memcpy floor %.2f, %.0f%% of a %.1f ms frame)%s\n",
                    c.what, read_ms, floor_ms, pct, c.budget_ms,
                    pct > 50 ? "  <-- too slow" : "");
    }

    std::printf("\n  Read-back is the SLOW path: it exists for the CPU renderer,\n");
    std::printf("  for screen capture and for the parity test. Compositing samples\n");
    std::printf("  the imported image directly and never copies it.\n");
    return 0;
}
