// test_format.cpp — formats and layouts.
//
// Most of this is static_assert, because the whole point of the type is that
// it answers at compile time: a format's alpha-ness, a modifier's
// readability, a layout's validity. A runtime test of a constexpr fact is
// weaker than the compiler refusing to build.
#include "check.hpp"

#include <dye/format.hpp>

#include <cstdio>
#include <set>

using namespace dye;

namespace {

// ---------------------------------------------------------------------------
// The X formats' fourth byte is NOT alpha.
//
// This is the bug the Order enum exists for. A client attaches an XRGB
// buffer, leaves the top byte as whatever its allocator returned, and a
// compositor that treats the buffer as ARGB renders the window with random
// holes in it. Worse, on most allocators that byte happens to be 0xff, so it
// looks right until it doesn't.
// ---------------------------------------------------------------------------
static_assert(formats::argb8888.has_alpha());
static_assert(formats::abgr8888.has_alpha());
static_assert(!formats::xrgb8888.has_alpha());
static_assert(!formats::xbgr8888.has_alpha());

// The X and A pair differ ONLY in alpha: same size, same channel order.
static_assert(formats::argb8888.bytes_per_pixel() == formats::xrgb8888.bytes_per_pixel());
static_assert(is_rgb(formats::argb8888.order()) == is_rgb(formats::xrgb8888.order()));
static_assert(formats::argb8888 != formats::xrgb8888);

// RGB and BGR are different formats. Confusing them swaps red and blue,
// which is the other classic: it looks like a colour-management bug and is
// actually a byte-order one.
static_assert(is_rgb(formats::argb8888.order()));
static_assert(!is_rgb(formats::abgr8888.order()));
static_assert(formats::argb8888 != formats::abgr8888);

// A default Format is not a format. It cannot be mistaken for ARGB8888 just
// because its fourcc is 0.
static_assert(!Format{}.valid());
static_assert(!Format{});
static_assert(Format{}.bytes_per_pixel() == 0);

// The fourcc values are the kernel's, checked against the literal bytes
// rather than against themselves — a table that agrees with itself proves
// nothing.
static_assert(formats::argb8888.fourcc() == 0x34325241u);   // 'AR24'
static_assert(formats::xrgb8888.fourcc() == 0x34325258u);   // 'XR24'
static_assert(formats::abgr8888.fourcc() == 0x34324241u);   // 'AB24'
static_assert(formats::xbgr8888.fourcc() == 0x34324258u);   // 'XB24'

// Parsing a fourcc from outside is a total function: known ones come back,
// everything else is nullopt rather than a guess.
static_assert(Format::from_fourcc(0x34325241u) == formats::argb8888);
static_assert(!Format::from_fourcc(0).has_value());
static_assert(!Format::from_fourcc(0x33324452u).has_value());   // 'RD23', nonsense

// Strides: what a row needs, before any driver padding.
static_assert(formats::argb8888.min_stride(100) == 400);
static_assert(formats::argb8888.min_stride(0) == 0);
static_assert(formats::argb8888.min_stride(-5) == 0);   // not a huge unsigned

// ---------------------------------------------------------------------------
// Modifiers.
//
// The trap here is that 0 is a REAL value (linear), so a modifier that was
// never set looks like a perfectly good CPU-readable layout. That is why
// "unset" is DRM_FORMAT_MOD_INVALID and why is_invalid() exists separately
// from is_linear().
// ---------------------------------------------------------------------------
static_assert(Modifier::linear().raw() == 0);
static_assert(Modifier::linear().is_linear());
static_assert(Modifier::linear().cpu_readable());
static_assert(!Modifier::linear().is_invalid());

static_assert(Modifier::invalid().raw() == 0x00ffffffffffffffULL);
static_assert(Modifier::invalid().is_invalid());
static_assert(!Modifier::invalid().is_linear());
static_assert(!Modifier::invalid().cpu_readable());

// A default Modifier IS linear, which is the kernel's meaning of 0 and worth
// pinning: if that ever changes, a great deal of read-back code is wrong.
static_assert(Modifier{}.is_linear());

// A tiled layout is not CPU-readable. This one is NVIDIA's 16Bx2 block
// layout, the case that actually broke: it imports fine, samples fine on the
// GPU, and produces garbage the moment anything memcpys it.
constexpr Modifier nvidia_16bx2{0x0300000000000015ULL};
static_assert(!nvidia_16bx2.is_linear());
static_assert(!nvidia_16bx2.cpu_readable());
static_assert(nvidia_16bx2.vendor() == 0x03);
static_assert(nvidia_16bx2 != Modifier::linear());

// ---------------------------------------------------------------------------
// Layout: the pair, as one thing.
// ---------------------------------------------------------------------------
static_assert(Layout{formats::argb8888, Modifier::linear()}.valid());
static_assert(Layout{formats::argb8888, Modifier::linear()}.cpu_readable());

// A good format in a tiled layout is valid but NOT readable. Conflating
// those two questions is how a compositor ends up memcpying tiled memory.
static_assert(Layout{formats::argb8888, nvidia_16bx2}.valid());
static_assert(!Layout{formats::argb8888, nvidia_16bx2}.cpu_readable());

// MOD_INVALID is not a layout at all: it means "you pick", which is a
// request when allocating and a bug when importing.
static_assert(!Layout{formats::argb8888, Modifier::invalid()}.valid());
static_assert(!Layout{formats::argb8888, Modifier::invalid()}.cpu_readable());

// No format, no layout.
static_assert(!Layout{}.valid());
static_assert(!Layout{Format{}, Modifier::linear()}.valid());

// Same format, different modifier: different layout. This comparison is what
// a format-negotiation table is keyed on, so getting it wrong means offering
// a client a layout the GPU cannot actually import.
static_assert(Layout{formats::argb8888, Modifier::linear()} !=
              Layout{formats::argb8888, nvidia_16bx2});

// Zero cost: a Format is its fourcc plus a name, a Modifier is its integer.
static_assert(sizeof(Modifier) == sizeof(std::uint64_t));

}  // namespace

// ---------------------------------------------------------------------------
// The handful of facts that need a loop rather than an assertion.
// ---------------------------------------------------------------------------
void run_format_tests() {
    std::printf("formats know what they are\n");

    // Every format in the table round-trips through its fourcc. A table
    // entry with a typo'd fourcc would otherwise sit there unnoticed until a
    // client sent that exact format.
    for (const Format& f : formats::all) {
        const auto back = Format::from_fourcc(f.fourcc());
        CHECK(back.has_value());
        CHECK(back && *back == f);
        CHECK(f.valid());
        CHECK(!f.name().empty());
        CHECK(f.bytes_per_pixel() == 4);
    }

    // No two entries share a fourcc: a copy-paste in the table would make
    // from_fourcc return the wrong format, silently.
    std::set<std::uint32_t> seen;
    for (const Format& f : formats::all) CHECK(seen.insert(f.fourcc()).second);
    CHECK(seen.size() == std::size(formats::all));

    std::printf("  %zu formats, each distinct, each round-tripping\n", seen.size());
}
