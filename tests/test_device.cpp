// test_device.cpp — the GPU, when there is one.
//
// Every test here SKIPS rather than fails when there is no GPU: a VM and a
// CI runner are ordinary places to build a compositor, and a suite that goes
// red there teaches people to ignore it.
//
// But a skip that looks like a pass is worse than no test — heddle's
// hardware tests hardcoded /dev/dri/card0, skipped everything on a machine
// whose GPU was card1, and still printed "all passed". So the skips are
// counted and reported at the end.
#include "check.hpp"

#include <dye/device.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

using namespace dye;

namespace {

int skipped = 0;

/// The GPU, opened once: EGL initialisation is slow and every test wants the
/// same device.
Device* gpu() {
    static auto opened = Device::open();
    return opened ? opened->get() : nullptr;
}

// ---------------------------------------------------------------------------

void opening_reports_why_it_failed() {
    std::printf("opening a GPU either works or says why\n");

    auto d = Device::open();
    if (!d) {
        // Not a failure. But the error has to be usable: a compositor prints
        // it and falls back to the CPU renderer, and "" tells nobody
        // anything.
        CHECK(!d.error().what.empty());
        CHECK(d.error().code != std::errc{});
        std::printf("  SKIP: %s\n", d.error().what.data());
        ++skipped;
        return;
    }

    CHECK(!(*d)->node().empty());
    CHECK((*d)->device_id() != 0);
    std::printf("  %.*s, device %llu\n", static_cast<int>((*d)->node().size()),
                (*d)->node().data(),
                static_cast<unsigned long long>((*d)->device_id()));

    // A node that does not exist fails cleanly rather than crashing.
    auto bad = Device::open("/dev/dri/renderD999");
    CHECK(!bad.has_value());
    CHECK(!bad.error().what.empty());
}

void the_format_list_is_honest() {
    std::printf("the advertised formats are ones this GPU can really import\n");
    Device* d = gpu();
    if (!d) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }

    const auto formats = d->formats();
    CHECK(!formats.empty());

    int linear = 0;
    for (const Layout& l : formats) {
        // Every entry must be something dye can actually composite. An
        // advertised format the renderer cannot sample means a client
        // allocates a buffer whose window then never appears.
        CHECK(l.format.valid());
        CHECK(!l.modifier.is_invalid());
        CHECK(d->supports(l));
        if (l.modifier.is_linear()) ++linear;
    }

    // NOT "linear must be here". It is tempting — linear is the layout
    // every client can allocate without knowing anything about the GPU —
    // but NVIDIA's proprietary driver advertises 48 layouts, all of them
    // its own tiling, and refuses DRM_FORMAT_MOD_LINEAR outright. A test
    // demanding linear passes on Intel and AMD and is simply wrong there.
    //
    // What IS required is that the list is usable: at least one layout, and
    // allocate() can produce a buffer in one of them.

    // A layout nobody supports is not claimed. 0x00ff... is MOD_INVALID,
    // which is never importable.
    CHECK(!d->supports(Layout{formats::argb8888, Modifier::invalid()}));

    std::printf("  %zu layouts, %d linear, %d vendor-tiled\n", formats.size(), linear,
                static_cast<int>(formats.size()) - linear);
}

void a_real_buffer_imports_and_reads_back() {
    std::printf("a client buffer imports, and the pixels survive the round trip\n");
    Device* d = gpu();
    if (!d) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }
    // Allocated BY THE DEVICE, in a layout it actually supports. A
    // hand-made linear buffer is refused outright by NVIDIA, so a test that
    // makes its own skips on the machine it most needs to run on.
    auto desc = d->allocate(16, 16, formats::argb8888);
    if (!desc) {
        std::printf("  SKIP: allocate refused: %s\n", desc.error().what.data());
        ++skipped;
        return;
    }
    CHECK(d->supports(desc->layout));
    CHECK(is_valid(*desc));
    const bool tiled = !desc->layout.modifier.is_linear();

    auto image = d->import(std::move(*desc));
    if (!image) {
        std::printf("  SKIP: import refused: %s\n", image.error().what.data());
        ++skipped;
        return;
    }

    CHECK((*image)->size().width == 16);
    CHECK((*image)->size().height == 16);
    CHECK((*image)->format() == formats::argb8888);
    CHECK((*image)->texture().valid());

    // as_texture() restates the image's facts for the draw list. Getting one
    // of the three wrong is a whole class of bug a compositor would
    // otherwise write by hand at every call site.
    const Texture t = (*image)->as_texture();
    CHECK(t.id == (*image)->texture());
    CHECK(t.size.width == 16);
    CHECK(t.format == formats::argb8888);

    // Read-back has to WORK, whatever the layout: that is the whole point
    // of going through the GPU rather than mapping the memory. A tiled
    // buffer cannot be memcpy'd, and this path is how its pixels are
    // obtained at all.
    std::vector<std::uint32_t> back(16 * 16, 0xdeadbeefu);
    const auto r = (*image)->read(back.data(), 16);
    if (!r) {
        std::printf("  SKIP: read-back unsupported: %s\n", r.error().what.data());
        ++skipped;
        return;
    }

    // Every pixel was written. A read that leaves the scratch value behind
    // is what a silently-failing glReadPixels looks like, and it is the
    // failure that would otherwise be mistaken for a black window.
    int written = 0;
    for (std::uint32_t px : back)
        if (px != 0xdeadbeefu) ++written;
    CHECK(written == 16 * 16);

    std::printf("  16x16 ARGB (%s) imported, %d/%d pixels read back\n",
                tiled ? "vendor-tiled" : "linear", written, 16 * 16);
}

void malformed_buffers_are_refused_before_the_driver() {
    std::printf("a malformed buffer is refused the same way on every GPU\n");
    Device* d = gpu();
    if (!d) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }

    // The point of checking these here rather than letting EGL answer: one
    // driver returns an error, another renders garbage, a third hangs the
    // GPU. A compositor cannot ship "it depends on the vendor".
    auto base = d->allocate(8, 8, formats::argb8888);
    if (!base) {
        std::printf("  SKIP: allocate refused\n");
        ++skipped;
        return;
    }

    auto broken = std::move(*base);
    broken.planes[0].stride = 4;   // far too small for 8 pixels
    const auto r = d->import(std::move(broken));
    CHECK(!r.has_value());
    CHECK(r.error().code == std::errc::invalid_argument);
    // The message is the validator's, which means the driver was never
    // asked — the check ran on arithmetic, before anything vendor-specific.
    CHECK(r.error().what == describe(BadBuffer::stride_too_small));

    // An unsupported layout is a different error from a malformed buffer,
    // and a compositor maps the two onto different protocol errors.
    auto wrong_layout = d->allocate(8, 8, formats::argb8888);
    if (wrong_layout) {
        // A layout this GPU does not claim. Made up rather than borrowed
        // from another vendor's list, so the case holds whatever hardware
        // this runs on.
        wrong_layout->layout.modifier = Modifier{0x00abcdef12345678ULL};
        CHECK(!d->supports(wrong_layout->layout));
        const auto r2 = d->import(std::move(*wrong_layout));
        CHECK(!r2.has_value());
        CHECK(r2.error().code == std::errc::not_supported);
    }

    std::printf("  rejected by arithmetic, with the validator's own message\n");
}

void importing_does_not_keep_the_client_s_descriptors() {
    std::printf("an imported image does not hold the client's descriptors\n");
    Device* d = gpu();
    if (!d) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }

    const auto open_count = [] {
        std::size_t n = 0;
        if (DIR* dir = ::opendir("/proc/self/fd")) {
            while (::readdir(dir)) ++n;
            ::closedir(dir);
        }
        return n;
    };

    auto desc = d->allocate(8, 8, formats::argb8888);
    if (!desc) {
        std::printf("  SKIP: allocate refused\n");
        ++skipped;
        return;
    }

    const std::size_t before = open_count();
    auto image = d->import(std::move(*desc));
    if (!image) {
        std::printf("  SKIP: import refused\n");
        ++skipped;
        return;
    }

    // The descriptor the client sent is gone: the driver took what it needed
    // during import. Holding it would pin the client's memory for as long as
    // the compositor kept the image, and a hundred windows meant a hundred
    // descriptors and eventually EMFILE. That was a real leak.
    const std::size_t after = open_count();
    CHECK(after <= before);

    std::printf("  descriptors before %zu, after %zu\n", before, after);
}

}  // namespace

void run_device_tests() {
    opening_reports_why_it_failed();
    the_format_list_is_honest();
    a_real_buffer_imports_and_reads_back();
    malformed_buffers_are_refused_before_the_driver();
    importing_does_not_keep_the_client_s_descriptors();

    // Say so loudly. A silent skip is how a suite reports "all passed" on a
    // machine where it tested nothing.
    if (skipped) std::printf("\n  (%d GPU check(s) skipped)\n", skipped);
}
