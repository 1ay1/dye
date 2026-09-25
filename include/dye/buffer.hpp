#pragma once
// dye/buffer.hpp — what a client's GPU buffer actually is.
//
// A dmabuf arrives as a pile of loose integers: some descriptors, a width, a
// height, a fourcc, a modifier, and per plane an offset and a stride. They
// are all the same type to the compiler, they arrive in an order chosen by a
// protocol rather than by sense, and one of them (the descriptor) is an owned
// resource that must be closed exactly once.
//
// Getting that wrong does not crash. It renders a window shifted by a few
// pixels, or torn, or black — on someone else's GPU. So the description is a
// type: the fds are jaal::owned_handle (closed once, never copied), the
// format and modifier are dye::Layout (one thing, not two), and the
// per-plane numbers are named rather than positional.
//
// Nothing here touches the GPU. This is the description a client sent, and
// validating it is a separate step (`validate`) that runs BEFORE any driver
// call — because the driver's answer to a malformed buffer ranges from an
// error code to a hang, and the protocol wants a specific error message
// anyway.

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <jaal/platform/handle.hpp>

#include "format.hpp"

namespace dye {

using jaal::platform::borrowed_handle;
using jaal::platform::owned_handle;

/// The largest number of planes any format here uses. Single-plane RGB is
/// one; the constant exists so a loop bound is never a literal 4 that
/// outlives the reason for it.
inline constexpr std::size_t kMaxPlanes = 4;

// ---------------------------------------------------------------------------
// Plane — one chunk of memory inside a buffer.
// ---------------------------------------------------------------------------

/// Where one plane's pixels live.
///
/// `offset` and `stride` are both byte counts and both uint32, adjacent in
/// every API that takes them, and swapping them is undetectable at the call
/// site. Named fields in a struct make the swap visible; taking the pair by
/// this type makes it impossible.
struct Plane {
    /// The memory. Owned: this closes exactly once, whatever path unwinds.
    owned_handle fd{};

    /// Bytes from the start of the mapping to this plane's first pixel.
    std::uint32_t offset = 0;

    /// Bytes from one row of pixels to the next. At least the format's
    /// minimum; usually more, because drivers pad rows for alignment.
    std::uint32_t stride = 0;
};

// ---------------------------------------------------------------------------
// BufferDescription — a client's buffer, before anyone has looked at it.
// ---------------------------------------------------------------------------

/// Move-only, because it owns descriptors.
struct BufferDescription {
    std::int32_t width = 0;
    std::int32_t height = 0;

    /// The format and the memory arrangement, as one fact.
    Layout layout{};

    /// The planes, in order. One for every format this library handles.
    std::vector<Plane> planes{};

    /// The client drew this upside down (GL's origin is bottom-left).
    /// Carried rather than corrected: the renderer flips a texture for free
    /// by sampling differently, while flipping in memory costs a copy.
    bool y_invert = false;

    [[nodiscard]] std::uint64_t pixels() const noexcept {
        if (width <= 0 || height <= 0) return 0;
        return static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    }
};

// ---------------------------------------------------------------------------
// Validation.
//
// Every one of these is a real bug a real client has shipped, and each one
// produces a different wrong picture rather than an error. They are checked
// here, once, before the driver sees anything.
// ---------------------------------------------------------------------------

/// Why a buffer description is unusable.
///
/// An enum rather than a string because the caller has to MAP these onto its
/// own protocol errors (zwp_linux_buffer_params_v1 has its own codes), and
/// matching on a message is how that goes wrong.
enum class BadBuffer : std::uint8_t {
    ok,
    no_planes,          ///< nothing to read
    too_many_planes,    ///< more than the format can use
    bad_size,           ///< zero or negative width/height
    huge,               ///< width * height * bpp overflows, or is absurd
    unknown_format,     ///< a fourcc this library does not handle
    invalid_modifier,   ///< DRM_FORMAT_MOD_INVALID: nobody knows the layout
    bad_fd,             ///< a plane with no descriptor
    stride_too_small,   ///< a row cannot hold width pixels
    out_of_range,       ///< offset + stride * height overflows 64 bits
};

[[nodiscard]] constexpr std::string_view describe(BadBuffer b) noexcept {
    switch (b) {
        case BadBuffer::ok:               return "ok";
        case BadBuffer::no_planes:        return "the buffer has no planes";
        case BadBuffer::too_many_planes:  return "more planes than the format uses";
        case BadBuffer::bad_size:         return "width and height must both be positive";
        case BadBuffer::huge:             return "the buffer is larger than makes sense";
        case BadBuffer::unknown_format:   return "unsupported pixel format";
        case BadBuffer::invalid_modifier: return "DRM_FORMAT_MOD_INVALID is not a layout";
        case BadBuffer::bad_fd:           return "a plane has no file descriptor";
        case BadBuffer::stride_too_small: return "the stride is too small for the width";
        case BadBuffer::out_of_range:     return "offset and stride run past 64 bits";
    }
    return "unknown";
}

/// The largest buffer worth believing: 16384 on a side.
///
/// Not a hardware limit but a sanity one. A client that asks for 2^30 pixels
/// is confused or hostile, and the arithmetic downstream (stride * height,
/// allocations) is much easier to reason about with a bound in front of it.
inline constexpr std::int32_t kMaxDimension = 16384;

/// The furthest into a mapping any plane may reach, in bytes.
///
/// Derived, not picked: the biggest image kMaxDimension allows is
/// 16384 x 16384 x 4 = 1 GiB. Four times that leaves room for a plane living
/// inside a shared pool with generous padding, and still catches the case
/// this bound is really for — an offset or stride near the top of its uint32
/// range, which wraps to a small number in 32-bit arithmetic and sends the
/// driver reading outside the mapping entirely.
inline constexpr std::uint64_t kMaxExtent = std::uint64_t{4} << 30;   // 4 GiB

/// Check a description without touching the GPU.
///
/// Pure, so it is testable without a GPU — and it is where every test of the
/// "malformed buffer" family lives, because a driver cannot be relied on to
/// reject these consistently (some return an error, some produce garbage,
/// some hang).
[[nodiscard]] inline BadBuffer validate(const BufferDescription& d) noexcept {
    if (d.width <= 0 || d.height <= 0) return BadBuffer::bad_size;
    if (d.width > kMaxDimension || d.height > kMaxDimension) return BadBuffer::huge;
    if (!d.layout.format.valid()) return BadBuffer::unknown_format;
    if (d.layout.modifier.is_invalid()) return BadBuffer::invalid_modifier;
    if (d.planes.empty()) return BadBuffer::no_planes;
    if (d.planes.size() > kMaxPlanes) return BadBuffer::too_many_planes;

    const std::uint64_t min_stride = d.layout.format.min_stride(d.width);
    for (const Plane& p : d.planes) {
        if (!p.fd.valid()) return BadBuffer::bad_fd;
        if (p.stride < min_stride) return BadBuffer::stride_too_small;
        // The real extent the driver will read: the last row's start, plus
        // one row. Computed in 64 bits from values already bounded above, so
        // this arithmetic cannot itself wrap — which is the whole point,
        // because in the uint32s these arrive as, it does.
        const std::uint64_t last_row =
            static_cast<std::uint64_t>(p.stride) * static_cast<std::uint64_t>(d.height - 1);
        const std::uint64_t extent = static_cast<std::uint64_t>(p.offset) + last_row + min_stride;
        if (extent > kMaxExtent) return BadBuffer::out_of_range;
    }
    return BadBuffer::ok;
}

[[nodiscard]] inline bool is_valid(const BufferDescription& d) noexcept {
    return validate(d) == BadBuffer::ok;
}

}  // namespace dye
