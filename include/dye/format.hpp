#pragma once
// dye/format.hpp — pixel formats, and the layouts a GPU stores them in.
//
// The bug this exists to prevent
// -----------------------------
// A DRM fourcc is a uint32_t. So is a modifier's low half, a stride, an
// offset, a width, a plane count and a GL enum. Every graphics API in this
// space takes a row of them:
//
//   eglCreateImage(dpy, ctx, target, buf, attrs);   // fourcc, stride,
//                                                   // offset, modifier...
//
// and every one of those is the same type to the compiler. Passing a stride
// where an offset goes compiles, and produces a black window or a torn one,
// on a machine you do not have.
//
// So a format here is a TYPE that knows what it is: how many bytes a pixel
// takes, whether those bytes carry alpha, and which byte lands where. A
// modifier is a separate type again, because "ARGB8888" and "ARGB8888 in
// NVIDIA's 16Bx2 block layout" are not interchangeable — the second cannot
// be read by the CPU at all, and mixing them up is how you get a compositor
// that works on Intel and garbles on NVIDIA.
//
// No GPU headers: a fourcc is four bytes of ASCII from the kernel's uapi
// (drm_fourcc.h), spelled out here so this header costs nothing to include
// and can be used by code with no EGL, no GL and no libdrm.

#include <compare>
#include <cstdint>
#include <optional>
#include <string_view>

namespace dye {

// ---------------------------------------------------------------------------
// Channel order.
//
// "RGBA" names four things at once: which channels exist, what order they sit
// in memory, whether there is an alpha channel, and whether that alpha means
// anything. Spelling the first two as an enum means a conversion between two
// formats is a total function over a small set rather than a chain of ifs
// that quietly lacks a case.
// ---------------------------------------------------------------------------
enum class Order : std::uint8_t {
    argb,   ///< byte 0 = B, 1 = G, 2 = R, 3 = A on little-endian
    xrgb,   ///< same bytes; the fourth is padding and means nothing
    abgr,
    xbgr,
};

/// Does this order carry a meaningful alpha channel?
///
/// The X formats have a fourth byte, and it is NOT alpha: a client may leave
/// anything in it. Compositing it as alpha is the classic "why is this window
/// transparent in places" bug, so the answer lives with the format rather
/// than at each use.
[[nodiscard]] constexpr bool has_alpha(Order o) noexcept {
    return o == Order::argb || o == Order::abgr;
}

/// Is red before blue in memory?
[[nodiscard]] constexpr bool is_rgb(Order o) noexcept {
    return o == Order::argb || o == Order::xrgb;
}

// ---------------------------------------------------------------------------
// Format — one pixel layout, as a value.
// ---------------------------------------------------------------------------

/// A pixel format the whole stack agrees on.
///
/// Constructed only from the known set below: there is no Format::from_raw
/// taking any uint32_t, because "whatever four bytes the client sent" is
/// exactly the thing that must be validated at the protocol edge and then
/// never doubted again. `Format::from_fourcc` is the edge, and it returns
/// an optional.
class Format {
public:
    constexpr Format() = default;

    /// The DRM fourcc, for handing to the kernel or to EGL.
    [[nodiscard]] constexpr std::uint32_t fourcc() const noexcept { return fourcc_; }

    [[nodiscard]] constexpr Order order() const noexcept { return order_; }
    [[nodiscard]] constexpr std::uint32_t bytes_per_pixel() const noexcept { return bpp_; }
    [[nodiscard]] constexpr bool has_alpha() const noexcept { return dye::has_alpha(order_); }
    [[nodiscard]] constexpr bool valid() const noexcept { return fourcc_ != 0; }
    constexpr explicit operator bool() const noexcept { return valid(); }

    /// The name the kernel uses, for logs and for test failures that would
    /// otherwise print a meaningless number.
    [[nodiscard]] constexpr std::string_view name() const noexcept { return name_; }

    /// Bytes one row of `width` pixels occupies, ignoring padding. The
    /// actual stride of a buffer is a separate fact (Plane::stride) and is
    /// usually larger.
    [[nodiscard]] constexpr std::uint64_t min_stride(std::int32_t width) const noexcept {
        return width <= 0 ? 0 : static_cast<std::uint64_t>(width) * bpp_;
    }

    friend constexpr bool operator==(Format, Format) = default;

    /// Parse a fourcc that came from outside: a client's dmabuf, a KMS
    /// plane's format list. Nothing if it is not one this library handles,
    /// which the caller turns into a protocol error rather than a guess.
    [[nodiscard]] static constexpr std::optional<Format> from_fourcc(std::uint32_t f) noexcept;

private:
    constexpr Format(std::uint32_t f, Order o, std::uint32_t bpp, std::string_view n) noexcept
        : fourcc_(f), name_(n), order_(o), bpp_(bpp) {}

    friend struct formats;

    std::uint32_t    fourcc_ = 0;
    std::string_view name_{};
    Order            order_ = Order::argb;
    std::uint32_t    bpp_ = 0;
};

namespace detail {
/// drm_fourcc.h's macro, as a constexpr function.
constexpr std::uint32_t fourcc(char a, char b, char c, char d) noexcept {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8 |
           static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16 |
           static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24;
}
}  // namespace detail

/// The formats this library knows how to sample, blend and scan out.
///
/// Deliberately few. Every one here is 8 bits per channel and one plane,
/// which is what a compositor actually composites; YUV and 10-bit belong
/// with video and HDR, and adding them means adding conversion code, not
/// just a table row.
struct formats {
    static constexpr Format argb8888{detail::fourcc('A', 'R', '2', '4'), Order::argb, 4,
                                     "ARGB8888"};
    static constexpr Format xrgb8888{detail::fourcc('X', 'R', '2', '4'), Order::xrgb, 4,
                                     "XRGB8888"};
    static constexpr Format abgr8888{detail::fourcc('A', 'B', '2', '4'), Order::abgr, 4,
                                     "ABGR8888"};
    static constexpr Format xbgr8888{detail::fourcc('X', 'B', '2', '4'), Order::xbgr, 4,
                                     "XBGR8888"};

    /// Everything above, in one place, so a loop over "the formats we take"
    /// cannot forget one.
    static constexpr Format all[] = {argb8888, xrgb8888, abgr8888, xbgr8888};
};

constexpr std::optional<Format> Format::from_fourcc(std::uint32_t f) noexcept {
    for (const Format& k : formats::all)
        if (k.fourcc_ == f) return k;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Modifier — how those pixels are arranged in memory.
//
// A modifier is the driver's tiling and compression scheme. It is the
// difference between "these bytes are rows of pixels" and "these bytes are
// 16x2 blocks, swizzled, with a separate compression plane", and it is NOT
// part of the format: the same ARGB8888 buffer can be linear on one GPU and
// tiled on another.
//
// The distinction matters because linear is the only layout the CPU can read.
// A modifier kept as a bare uint64_t next to a fourcc gets swapped with it,
// or defaulted to 0 — and 0 is a REAL modifier (linear), so that mistake
// silently claims a tiled buffer is readable and renders garbage.
// ---------------------------------------------------------------------------
class Modifier {
public:
    constexpr Modifier() = default;
    constexpr explicit Modifier(std::uint64_t raw) noexcept : raw_(raw) {}

    [[nodiscard]] constexpr std::uint64_t raw() const noexcept { return raw_; }

    /// Rows of pixels, no tiling: the only layout the CPU may read directly.
    [[nodiscard]] static constexpr Modifier linear() noexcept { return Modifier{0}; }

    /// "The driver picks." Not a layout — a request, valid only when
    /// ALLOCATING. Importing with this is a bug: it means nobody knows how
    /// the bytes are arranged.
    [[nodiscard]] static constexpr Modifier invalid() noexcept {
        return Modifier{0x00ffffffffffffffULL};
    }

    [[nodiscard]] constexpr bool is_linear() const noexcept { return raw_ == 0; }
    [[nodiscard]] constexpr bool is_invalid() const noexcept {
        return raw_ == invalid().raw_;
    }

    /// Can the CPU read a buffer in this layout? Only linear.
    [[nodiscard]] constexpr bool cpu_readable() const noexcept { return is_linear(); }

    /// Which vendor defined it, for logs: the top 8 bits.
    [[nodiscard]] constexpr std::uint8_t vendor() const noexcept {
        return static_cast<std::uint8_t>(raw_ >> 56);
    }

    friend constexpr auto operator<=>(Modifier, Modifier) = default;
    friend constexpr bool operator==(Modifier, Modifier) = default;

private:
    std::uint64_t raw_ = 0;
};

// ---------------------------------------------------------------------------
// Layout — a format AND the arrangement of its bytes.
//
// These two always travel together: a buffer is not "ARGB8888", it is
// "ARGB8888, linear" or "ARGB8888, in this vendor's tiling". Every API in
// the stack takes both, next to each other, as a uint32 and a uint64. Pairing
// them in one type means the pair is passed as one thing and cannot be
// half-updated.
// ---------------------------------------------------------------------------
struct Layout {
    Format   format{};
    Modifier modifier{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return format.valid() && !modifier.is_invalid();
    }

    /// Can the CPU map and read this buffer? Needs a real format and a
    /// linear layout: the check every read-back path must make, in one
    /// place, so no caller reasons it out again and gets it wrong.
    [[nodiscard]] constexpr bool cpu_readable() const noexcept {
        return format.valid() && modifier.cpu_readable();
    }

    friend constexpr bool operator==(Layout, Layout) = default;
};

}  // namespace dye
