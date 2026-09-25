// test_buffer.cpp — the malformed buffers a driver cannot be trusted to
// reject.
//
// Every case here is something a client can send. The reason they are
// checked in dye rather than left to EGL is that drivers disagree about
// them: one returns an error, another renders garbage, a third hangs the
// GPU. A compositor needs one answer, before the driver is involved, and it
// needs to be able to test that answer on a machine with no GPU — which is
// what this file is.
#include "check.hpp"

#include <dye/buffer.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <utility>

using namespace dye;

namespace {

/// A real descriptor, so validate() sees what it will see in life.
owned_handle fresh_fd() {
    int fds[2];
    if (::pipe(fds) != 0) return owned_handle{};
    ::close(fds[1]);
    return owned_handle{fds[0]};
}

/// A description that is correct in every way, as the starting point each
/// case then breaks in exactly one place. Building the broken ones from a
/// known-good base is what keeps a case honest: if it fails, it fails for
/// the reason in its name.
BufferDescription good(std::int32_t w = 64, std::int32_t h = 64) {
    BufferDescription d;
    d.width = w;
    d.height = h;
    d.layout = Layout{formats::argb8888, Modifier::linear()};
    Plane p;
    p.fd = fresh_fd();
    p.offset = 0;
    p.stride = static_cast<std::uint32_t>(w) * 4;
    d.planes.push_back(std::move(p));
    return d;
}

void the_good_case_is_good() {
    std::printf("a well-formed buffer passes\n");
    const auto d = good();
    CHECK(validate(d) == BadBuffer::ok);
    CHECK(is_valid(d));
    CHECK(d.pixels() == 64 * 64);
    std::printf("  and everything below breaks exactly one thing about it\n");
}

void sizes() {
    std::printf("sizes that are not sizes\n");

    auto zero = good();
    zero.width = 0;
    CHECK(validate(zero) == BadBuffer::bad_size);

    auto neg = good();
    neg.height = -1;
    CHECK(validate(neg) == BadBuffer::bad_size);
    // A negative height passed to a driver becomes a huge unsigned. This is
    // the check that stops that conversion ever happening.
    CHECK(neg.pixels() == 0);

    auto huge = good();
    huge.width = kMaxDimension + 1;
    CHECK(validate(huge) == BadBuffer::huge);

    // Right at the limit is still fine: an off-by-one here would reject a
    // legitimate 16K buffer.
    auto edge = good(kMaxDimension, 4);
    CHECK(validate(edge) == BadBuffer::ok);

    std::printf("  zero, negative and absurd all rejected; the limit itself is not\n");
}

void formats_and_layouts() {
    std::printf("formats and layouts that cannot be rendered\n");

    auto unknown = good();
    unknown.layout.format = Format{};
    CHECK(validate(unknown) == BadBuffer::unknown_format);

    // MOD_INVALID means "the driver picks", which is a legitimate request
    // when ALLOCATING and meaningless when importing: nobody knows how the
    // bytes are arranged. Clients do send it.
    auto invalid_mod = good();
    invalid_mod.layout.modifier = Modifier::invalid();
    CHECK(validate(invalid_mod) == BadBuffer::invalid_modifier);

    // A tiled buffer is perfectly valid. It just cannot be read by the CPU,
    // which is a different question and must not be conflated with this one.
    auto tiled = good();
    tiled.layout.modifier = Modifier{0x0300000000000015ULL};
    CHECK(validate(tiled) == BadBuffer::ok);
    CHECK(!tiled.layout.cpu_readable());

    std::printf("  unknown format and MOD_INVALID rejected; tiled is valid but not readable\n");
}

void planes() {
    std::printf("planes\n");

    auto none = good();
    none.planes.clear();
    CHECK(validate(none) == BadBuffer::no_planes);

    auto many = good();
    for (int i = 0; i < 5; ++i) {
        Plane p;
        p.fd = fresh_fd();
        p.stride = 256;
        many.planes.push_back(std::move(p));
    }
    CHECK(validate(many) == BadBuffer::too_many_planes);

    // A plane with no descriptor: the protocol allows the message through,
    // and the driver dereferences it.
    auto no_fd = good();
    no_fd.planes[0].fd = owned_handle{};
    CHECK(validate(no_fd) == BadBuffer::bad_fd);

    std::printf("  none, too many, and one with no descriptor\n");
}

void strides() {
    std::printf("strides that cannot hold the pixels they claim\n");

    // The one that renders a diagonally-sheared window: a stride too small
    // for the width, so every row starts part-way into the previous one.
    auto small = good(64, 64);
    small.planes[0].stride = 63 * 4;
    CHECK(validate(small) == BadBuffer::stride_too_small);

    // Exactly enough is enough.
    auto exact = good(64, 64);
    exact.planes[0].stride = 64 * 4;
    CHECK(validate(exact) == BadBuffer::ok);

    // More than enough is normal: drivers pad rows for alignment, and a
    // compositor that demanded stride == width * bpp would reject most real
    // buffers.
    auto padded = good(64, 64);
    padded.planes[0].stride = 64 * 4 + 64;
    CHECK(validate(padded) == BadBuffer::ok);

    // A zero stride passes "is it a number" and fails "can it hold a row".
    auto zero = good();
    zero.planes[0].stride = 0;
    CHECK(validate(zero) == BadBuffer::stride_too_small);

    std::printf("  too small rejected, exact and padded accepted\n");
}

void arithmetic_that_would_overflow() {
    std::printf("offsets and strides that run off the end of the arithmetic\n");

    // stride * height is where the multiplication happens, and both are
    // uint32: in 32-bit arithmetic this wraps to a small number, and the
    // driver reads far outside the mapping.
    auto d = good(4096, 4096);
    d.planes[0].stride = 0xffffff00u;
    CHECK(validate(d) == BadBuffer::out_of_range);

    // An offset near the top of the range, likewise.
    auto off = good();
    off.planes[0].offset = 0xffffffffu;
    CHECK(validate(off) == BadBuffer::out_of_range);

    std::printf("  both caught in 64-bit arithmetic, before any driver sees them\n");
}

void every_reason_has_words() {
    std::printf("every rejection can be explained\n");
    // A caller has to turn these into a protocol error message. A reason
    // with no text, or two reasons with the same text, makes that report
    // useless — and this is the enum a new case gets added to.
    const BadBuffer all[] = {
        BadBuffer::ok,           BadBuffer::no_planes,      BadBuffer::too_many_planes,
        BadBuffer::bad_size,     BadBuffer::huge,           BadBuffer::unknown_format,
        BadBuffer::invalid_modifier, BadBuffer::bad_fd,     BadBuffer::stride_too_small,
        BadBuffer::out_of_range,
    };
    for (BadBuffer b : all) {
        CHECK(!describe(b).empty());
        CHECK(describe(b) != "unknown");
    }
    std::printf("  %zu reasons, each with its own message\n", std::size(all));
}

void descriptors_are_owned() {
    std::printf("a description owns its descriptors\n");

    // How many descriptors this process holds, counted by asking the kernel
    // rather than by probing a number we believe is closed. Probing is what
    // the obvious version of this test does — fcntl(raw, F_GETFD) after the
    // scope ends — and valgrind's --track-fds reports THAT as a
    // use-after-close, correctly: the descriptor may have been handed to
    // something else in between.
    const auto open_count = [] {
        std::size_t n = 0;
        if (DIR* d = ::opendir("/proc/self/fd")) {
            while (::readdir(d)) ++n;
            ::closedir(d);
        }
        return n;
    };

    const std::size_t before = open_count();
    {
        auto d = good();
        CHECK(d.planes[0].fd.valid());
        CHECK(open_count() > before);
    }
    // The description went out of scope and took the descriptor with it. A
    // compositor handles thousands of buffers a second; a leak here is a
    // client that dies on EMFILE after a few minutes of scrolling.
    CHECK(open_count() == before);

    // And it is move-only, so there is no path where two descriptions close
    // the same descriptor.
    static_assert(!std::is_copy_constructible_v<Plane>);
    static_assert(std::is_move_constructible_v<Plane>);

    std::printf("  closed when the description goes, and never copied\n");
}

}  // namespace

void run_buffer_tests() {
    the_good_case_is_good();
    sizes();
    formats_and_layouts();
    planes();
    strides();
    arithmetic_that_would_overflow();
    every_reason_has_words();
    descriptors_are_owned();
}
