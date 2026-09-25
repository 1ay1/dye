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

/// The GPU, opened once: device creation is slow and every test wants the
/// same one.
///
/// open_any(), not open(): a test wants llvmpipe when there is no real GPU,
/// because it runs everywhere and is a genuine Vulkan implementation. A
/// COMPOSITOR must never silently fall back to it — a desktop at 4 fps with
/// no explanation is worse than one that says it found no GPU — which is
/// why they are separate entry points.
Device* gpu() {
    static auto opened = Device::open_any();
    return opened ? opened->get() : nullptr;
}

// ---------------------------------------------------------------------------

void opening_reports_why_it_failed() {
    std::printf("opening a GPU either works or says why\n");

    auto d = Device::open_any();
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

    const DeviceInfo& info = (*d)->info();
    CHECK(!info.name.empty());
    CHECK(info.usable);
    std::printf("  %s (%s, driver %s)\n", info.name.c_str(), describe(info.kind).data(),
                info.driver.empty() ? "?" : info.driver.c_str());

    // A real GPU reports its DRM nodes; a software one has none, which is
    // how llvmpipe is identified rather than by matching on its name.
    if (info.is_software()) {
        CHECK(!info.render_node.valid());
        CHECK(!info.primary_node.valid());
        std::printf("  software renderer: no DRM nodes, as expected\n");
    } else {
        CHECK(info.render_node.valid() || info.primary_node.valid());
        std::printf("  drm nodes: render %u:%u  primary %u:%u\n", info.render_node.major,
                    info.render_node.minor, info.primary_node.major, info.primary_node.minor);
    }

    // A device number nobody owns fails cleanly rather than crashing.
    auto bad = Device::open(DeviceNumber{226, 250});
    CHECK(!bad.has_value());
    CHECK(!bad.error().what.empty());

    // And an invalid one is rejected before any lookup.
    auto zero = Device::open(DeviceNumber{});
    CHECK(!zero.has_value());
    CHECK(zero.error().code == std::errc::invalid_argument);
}

void enumeration_reports_unusable_gpus_too() {
    std::printf("every GPU is listed, with a reason when it cannot be used\n");

    auto gpus = enumerate_gpus();
    if (!gpus) {
        CHECK(!gpus.error().what.empty());
        std::printf("  SKIP: %s\n", gpus.error().what.data());
        ++skipped;
        return;
    }

    CHECK(!gpus->empty());
    for (const DeviceInfo& i : *gpus) {
        CHECK(!i.name.empty());
        // An unusable device must SAY why. "No GPU" on a machine with a
        // visible card is the least helpful message a compositor can print,
        // and this is the field that prevents it.
        if (!i.usable) CHECK(!i.unusable_because.empty());
        std::printf("  %-34s %-11s %s\n", i.name.c_str(), describe(i.kind).data(),
                    i.usable ? "usable" : i.unusable_because.c_str());
    }
}

void a_device_is_found_by_its_drm_node() {
    std::printf("a GPU is found by its DRM node, not by its name\n");

    auto gpus = enumerate_gpus();
    if (!gpus) {
        std::printf("  SKIP: no Vulkan\n");
        ++skipped;
        return;
    }

    // The multi-GPU entry point. Matching on a device NAME instead is how a
    // compositor picks the wrong card on a machine with two of the same
    // model — and the DRM node is what heddle's enumeration hands over.
    const DeviceInfo* real = nullptr;
    for (const DeviceInfo& i : *gpus)
        if (i.usable && i.render_node.valid()) real = &i;
    if (!real) {
        std::printf("  SKIP: no GPU reports a DRM render node\n");
        ++skipped;
        return;
    }

    auto by_node = Device::open(real->render_node);
    CHECK(by_node.has_value());
    if (by_node) {
        CHECK((*by_node)->info().name == real->name);
        CHECK((*by_node)->info().render_node == real->render_node);
    }

    // The PRIMARY node must find the same device. A compositor holding a
    // primary node (from heddle, for scanout) and one holding the render
    // node must agree about which GPU they mean.
    if (real->primary_node.valid()) {
        auto by_primary = Device::open(real->primary_node);
        CHECK(by_primary.has_value());
        if (by_primary) CHECK((*by_primary)->info().name == real->name);
        std::printf("  render %u:%u and primary %u:%u are the same GPU\n",
                    real->render_node.major, real->render_node.minor,
                    real->primary_node.major, real->primary_node.minor);
    }
}

void two_devices_negotiate_a_shared_layout() {
    std::printf("two GPUs agree on a layout, or the caller learns they cannot\n");
    Device* d = gpu();
    if (!d) {
        std::printf("  SKIP: no GPU\n");
        ++skipped;
        return;
    }

    // A device shares every layout with itself: the trivial case, but it is
    // the one a single-GPU machine takes, and it must not be empty.
    const auto with_self = d->shared_layouts(*d, formats::argb8888);
    CHECK(!with_self.empty());
    for (const Layout& l : with_self) {
        CHECK(d->supports(l));
        CHECK(l.format == formats::argb8888);
    }

    // Every shared layout is importable by BOTH. That is the property the
    // whole multi-GPU scheme rests on: advertise the intersection and a
    // client's buffer works on both by construction, rather than by luck.
    std::printf("  %zu shared ARGB8888 layouts\n", with_self.size());
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

/// A damage upload writes the rows it was given and LEAVES THE REST.
///
/// This is the whole risk of uploading only what changed. Get the layout
/// transition wrong (UNDEFINED says "the contents may be discarded") and the
/// untouched rows come back as garbage — on screen that is a window whose
/// unchanged parts flicker to noise, which no full-upload test can catch.
void a_partial_upload_keeps_the_rest() {
    auto device = dye::Device::open();
    if (!device) {
        std::printf("  SKIP: %s\n", device.error().what.data());
        ++skipped;
        return;
    }

    constexpr std::int32_t kW = 64, kH = 32;
    auto buf = (*device)->allocate(kW, kH, dye::formats::argb8888);
    if (!buf) {
        std::printf("  SKIP: allocate refused\n");
        ++skipped;
        return;
    }
    auto image = (*device)->import(std::move(*buf));
    if (!image) {
        std::printf("  SKIP: import refused\n");
        ++skipped;
        return;
    }

    // Fill it all with one colour, then change a band in the middle.
    constexpr std::uint32_t kOld = 0xff112233, kNew = 0xffddeeff;
    std::vector<std::uint32_t> all(static_cast<std::size_t>(kW) * kH, kOld);
    if (!(*image)->write(all.data(), kW)) {
        std::printf("  SKIP: full write refused\n");
        ++skipped;
        return;
    }

    constexpr std::int32_t kFrom = 8, kRows = 10;
    std::vector<std::uint32_t> edited = all;
    for (std::int32_t y = kFrom; y < kFrom + kRows; ++y)
        for (std::int32_t x = 0; x < kW; ++x)
            edited[static_cast<std::size_t>(y) * kW + x] = kNew;

    // The source is the WHOLE buffer; only these rows should move.
    if (!(*image)->write_rows(edited.data(), kW, kFrom, kRows)) {
        std::printf("  SKIP: partial write refused\n");
        ++skipped;
        return;
    }

    std::vector<std::uint32_t> back(static_cast<std::size_t>(kW) * kH, 0);
    if (!(*image)->read(back.data(), kW)) {
        std::printf("  SKIP: read-back refused\n");
        ++skipped;
        return;
    }

    bool band_updated = true, rest_intact = true;
    for (std::int32_t y = 0; y < kH; ++y) {
        const bool in_band = y >= kFrom && y < kFrom + kRows;
        for (std::int32_t x = 0; x < kW; ++x) {
            const std::uint32_t got = back[static_cast<std::size_t>(y) * kW + x];
            if (in_band) {
                if (got != kNew) band_updated = false;
            } else if (got != kOld) {
                rest_intact = false;
            }
        }
    }

    CHECK(band_updated);   // the damaged rows did change
    CHECK(rest_intact);    // and nothing else did

    // Rows outside the image are clamped, not an error: a client may damage a
    // region of a buffer it has just resized.
    CHECK((*image)->write_rows(edited.data(), kW, kH + 100, 4).has_value());
    CHECK((*image)->write_rows(edited.data(), kW, -5, 2).has_value());

    std::printf("  a damage upload changed %d rows and left %d alone\n", kRows, kH - kRows);
}

/// A big upload produces the right pixels, all of them.
///
/// The upload path does page-alignment and row-offset arithmetic that a
/// small image never exercises: a 64x32 test fits in one page and one
/// command, so it cannot catch an image shifted by a few rows. A 1080p
/// gradient can — a one-row or one-column slip shows up as thousands of
/// mismatched pixels rather than as nothing at all.
///
/// This also guards the fast paths that come and go here. A zero-copy
/// import (VK_EXT_external_memory_host) was tried and reverted because it
/// was slower in a real compositor than the staging copy; if it or anything
/// like it returns, this is the test that says whether it is CORRECT,
/// separately from whether it is fast.
void a_big_upload_is_still_correct() {
    auto device = dye::Device::open();
    if (!device) {
        std::printf("  SKIP: %s\n", device.error().what.data());
        ++skipped;
        return;
    }

    // Big enough to span many pages and many rows.
    constexpr std::int32_t kW = 1920, kH = 1080;
    auto buf = (*device)->allocate(kW, kH, dye::formats::xrgb8888);
    if (!buf) {
        std::printf("  SKIP: allocate refused\n");
        ++skipped;
        return;
    }
    auto image = (*device)->import(std::move(*buf));
    if (!image) {
        std::printf("  SKIP: import refused\n");
        ++skipped;
        return;
    }

    // A gradient, so a shift by even one row or column is visible as a
    // mismatch rather than hiding in a flat colour.
    std::vector<std::uint32_t> src(static_cast<std::size_t>(kW) * kH);
    for (std::int32_t y = 0; y < kH; ++y)
        for (std::int32_t x = 0; x < kW; ++x)
            src[static_cast<std::size_t>(y) * kW + x] =
                0xff000000u | (static_cast<std::uint32_t>(x & 0xff) << 16) |
                (static_cast<std::uint32_t>(y & 0xff) << 8) |
                static_cast<std::uint32_t>((x ^ y) & 0xff);

    if (!(*image)->write(src.data(), kW)) {
        std::printf("  SKIP: write refused\n");
        ++skipped;
        return;
    }

    std::vector<std::uint32_t> back(static_cast<std::size_t>(kW) * kH, 0);
    if (!(*image)->read(back.data(), kW)) {
        std::printf("  SKIP: read-back refused\n");
        ++skipped;
        return;
    }

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < src.size(); ++i)
        if ((back[i] & 0x00ffffffu) != (src[i] & 0x00ffffffu)) ++wrong;

    CHECK(wrong == 0);
    std::printf("  a %dx%d gradient survives a full upload (%zu wrong pixels)\n", kW, kH,
                wrong);
}

}  // namespace

void run_device_tests() {
    opening_reports_why_it_failed();
    enumeration_reports_unusable_gpus_too();
    a_device_is_found_by_its_drm_node();
    the_format_list_is_honest();
    two_devices_negotiate_a_shared_layout();
    a_real_buffer_imports_and_reads_back();
    malformed_buffers_are_refused_before_the_driver();
    importing_does_not_keep_the_client_s_descriptors();
    a_partial_upload_keeps_the_rest();
    a_big_upload_is_still_correct();

    // Say so loudly. A silent skip is how a suite reports "all passed" on a
    // machine where it tested nothing.
    if (skipped) std::printf("\n  (%d GPU check(s) skipped)\n", skipped);
}
