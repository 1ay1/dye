#pragma once
// dye/device.hpp — the GPU, as a resource with a lifetime.
//
// What this is for
// ----------------
// A compositor needs four things from a GPU, and only four:
//
//   * which formats and layouts it can actually import (formats())
//   * turning a client's dmabuf into something samplable (import())
//   * reading pixels back into memory (Image::read)
//   * knowing when the GPU has finished (fences, fence.hpp)
//
// Everything else EGL offers is for drawing to a window, and a compositor
// has no window.
//
// Why it is optional
// ------------------
// There may be no GPU: a VM, a CI runner, a machine whose driver refused to
// load. `Device::open()` then returns an error and the compositor uses the
// CPU renderer, which is why dye's other layers know nothing about this one.
// A library that made EGL mandatory would be untestable on exactly the
// machines tests run on.
//
// What it refuses to do
// ---------------------
// It does not validate buffers. That is buffer.hpp's job, it is pure, and it
// runs BEFORE any driver call — because a driver's response to a malformed
// buffer ranges from a clean error to a GPU hang, and no compositor can ship
// "it depends on the vendor" as its error handling.
//
// No EGL headers here. They live in src/device.cpp, so including this costs
// nothing and code that only wants the types (a test, the protocol layer)
// does not drag in a GL stack.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <jaal/core/error.hpp>
#include <jaal/platform/handle.hpp>
#include <weft/space.hpp>

#include "buffer.hpp"
#include "draw.hpp"
#include "format.hpp"

namespace dye {

using Error = jaal::error;
template <class T>
using Result = jaal::result<T>;
using Status = Result<void>;

// ---------------------------------------------------------------------------
// Image — a client's buffer, now something the GPU can sample.
// ---------------------------------------------------------------------------

/// An imported buffer.
///
/// Move-only and owning: destroying it releases the driver's handles. The
/// descriptors the client sent are NOT kept — the driver has taken what it
/// needs by the time import() returns, and holding them would pin a client's
/// memory for as long as the compositor happened to keep the image. That was
/// a real leak: a hundred imported buffers meant a hundred descriptors, and
/// the compositor hit EMFILE.
class Image {
public:
    virtual ~Image() = default;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    [[nodiscard]] weft::Size<weft::Buffer> size() const noexcept { return size_; }
    [[nodiscard]] Format format() const noexcept { return format_; }
    [[nodiscard]] bool y_invert() const noexcept { return y_invert_; }

    /// The texture, for a GL renderer to bind.
    [[nodiscard]] TextureId texture() const noexcept { return texture_; }

    /// As a draw-list source, so a compositor does not restate the three
    /// facts that must agree with the image (size, format, orientation) and
    /// get one of them wrong.
    [[nodiscard]] Texture as_texture() const noexcept {
        return Texture{texture_, size_, format_, y_invert_};
    }

    /// Copy the pixels into main memory as premultiplied ARGB32.
    ///
    /// This is the slow path, and deliberately so: it exists for the CPU
    /// renderer and for screen capture, not for compositing. `dst_stride_px`
    /// is in PIXELS, unlike every framebuffer API's byte stride — which is
    /// why it is in the name.
    virtual Status read(std::uint32_t* dst, std::int32_t dst_stride_px) = 0;

protected:
    Image() = default;

    weft::Size<weft::Buffer> size_{};
    Format                   format_{};
    TextureId                texture_{};
    bool                     y_invert_ = false;
};

// ---------------------------------------------------------------------------
// Device — one GPU.
// ---------------------------------------------------------------------------

/// A render node, opened for import and drawing.
///
/// Abstract so the GL implementation can live in one .cpp with the EGL
/// headers, and so a test can substitute a device that imports nothing. The
/// alternative — a concrete class with void* members — is what tapestry had,
/// and it made every test of the import path need a GPU.
class Device {
public:
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    /// Open the first usable render node (/dev/dri/renderD*).
    ///
    /// An error rather than an exception or a null: "there is no GPU here"
    /// is an ordinary state on a VM or a CI runner, and the caller's
    /// response is to use the CPU renderer, not to abort.
    [[nodiscard]] static Result<std::unique_ptr<Device>> open();

    /// Open one specific node, for a machine with two GPUs.
    [[nodiscard]] static Result<std::unique_ptr<Device>> open(const char* node);

    /// The node this is, for logs: "/dev/dri/renderD128".
    [[nodiscard]] virtual std::string_view node() const noexcept = 0;

    /// The device id (st_rdev), which is what a compositor advertises to
    /// clients so they allocate on the same GPU (linux-dmabuf's main_device).
    [[nodiscard]] virtual std::uint64_t device_id() const noexcept = 0;

    /// Every format and layout this GPU can import.
    ///
    /// The list a compositor advertises. Advertising one that is not here
    /// means a client allocates a buffer the compositor then cannot import,
    /// and the client sees its window never appear.
    [[nodiscard]] virtual std::span<const Layout> formats() const noexcept = 0;

    /// Can this exact format-and-layout pair be imported?
    [[nodiscard]] bool supports(Layout l) const noexcept {
        for (const Layout& k : formats())
            if (k == l) return true;
        return false;
    }

    /// Import a client's buffer.
    ///
    /// `d` must already have passed dye::validate: this checks what only the
    /// driver knows (can it take this layout), not what arithmetic can
    /// answer. Takes the description BY VALUE because importing consumes the
    /// descriptors.
    [[nodiscard]] virtual Result<std::unique_ptr<Image>> import(BufferDescription d) = 0;

    /// Allocate a buffer this GPU can definitely import.
    ///
    /// A compositor needs this for its own surfaces (a cursor, a scratch
    /// target), and a test needs it because a hand-made buffer may be in a
    /// layout the driver refuses. NVIDIA's proprietary driver is the case
    /// that forced this: it advertises 48 layouts, every one of them its own
    /// tiling, and refuses DRM_FORMAT_MOD_LINEAR outright — so "allocate a
    /// linear buffer and import it" works on Intel and AMD and cannot work
    /// there at all.
    ///
    /// The layout is the driver's choice from what it supports. The returned
    /// description says which, and carries the descriptors.
    [[nodiscard]] virtual Result<BufferDescription> allocate(std::int32_t width,
                                                             std::int32_t height,
                                                             Format format) = 0;

protected:
    Device() = default;
};

/// Is a GPU backend compiled in at all?
///
/// False on a build without EGL. A compositor checks this rather than
/// calling open() and interpreting the error, so "no EGL in this build" and
/// "no GPU in this machine" stay distinguishable.
[[nodiscard]] bool gpu_available() noexcept;

}  // namespace dye
