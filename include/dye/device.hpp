#pragma once
// dye/device.hpp — the GPU, as a resource with a lifetime.
//
// What this is for
// ----------------
// A compositor needs five things from a GPU, and only five:
//
//   * which formats and layouts it can import          (formats())
//   * turning a client's dmabuf into something samplable (import())
//   * allocating a buffer it can definitely import      (allocate())
//   * reading pixels back into memory                   (Image::read)
//   * knowing when the GPU has finished                 (fence.hpp)
//
// Everything else a graphics API offers is for drawing to a window, and a
// compositor has no window.
//
// Why Vulkan, not EGL
// -------------------
// Measured on this hardware (NVIDIA RTX 4060, proprietary driver), asked for
// ARGB8888:
//
//                       EGL/GLES        Vulkan
//   layouts offered     48              7
//   of which LINEAR     0               1
//   import a linear     REFUSED         works
//   dmabuf              (BAD_PARAMETER)
//
// LINEAR is the one layout every client can produce without knowing anything
// about the GPU. A compositor on EGL+NVIDIA cannot accept it. That single
// result decided the API; the rest (native fd fences, per-modifier feature
// flags, errors as return values) is corroborating. See DESIGN.md §1.
//
// Why it is optional
// ------------------
// There may be no GPU: a VM, a CI runner, a driver that refused to load.
// `Device::open()` returns an error and the compositor uses the CPU
// renderer, which is why dye's other layers know nothing about this one. A
// library that made Vulkan mandatory would be untestable on exactly the
// machines tests run on.
//
// What it refuses to do
// ---------------------
// It does not validate buffers. That is buffer.hpp's job, it is pure, and it
// runs BEFORE any driver call — because a driver's response to a malformed
// buffer ranges from a clean error to a GPU hang, and no compositor can ship
// "it depends on the vendor" as its error handling.
//
// It does not find devices either. That is heddle's job (heddle/discover.hpp):
// it is the DRM library, it already owns device nodes, and rendering does not
// need to know what a CRTC is. dye is handed a DeviceNumber.
//
// No Vulkan headers here. They live in src/device.cpp, so including this
// costs nothing and code that only wants the types (a test, the protocol
// layer) does not drag in a GPU stack.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
// DeviceNumber — which kernel device this is.
//
// Deliberately a plain pair of integers rather than a heddle type: it is the
// entire vocabulary dye and heddle share, and keeping it this small is what
// lets neither library include the other's headers.
//
// It is also exactly what VK_EXT_physical_device_drm reports, which is the
// authoritative link between "this Vulkan device" and "this /dev/dri node".
// Matching on a device NAME instead is how a compositor picks the wrong GPU
// on a machine with two of the same model.
// ---------------------------------------------------------------------------
struct DeviceNumber {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return major != 0; }
    friend constexpr bool operator==(DeviceNumber, DeviceNumber) = default;
};

/// What a GPU is, roughly. For choosing between several, and for logs.
enum class DeviceKind : std::uint8_t {
    discrete,     ///< a separate card, its own memory
    integrated,   ///< shares memory with the CPU
    software,     ///< llvmpipe, lavapipe: correct and slow
    other,
};

[[nodiscard]] constexpr std::string_view describe(DeviceKind k) noexcept {
    switch (k) {
        case DeviceKind::discrete:   return "discrete";
        case DeviceKind::integrated: return "integrated";
        case DeviceKind::software:   return "software";
        case DeviceKind::other:      return "other";
    }
    return "unknown";
}

/// One GPU Vulkan can see, before it is opened.
///
/// Enumerating separately from opening is what makes multi-GPU tractable: a
/// compositor can look at what exists, match it against the DRM device it
/// wants, and only then pay the cost of creating a logical device.
struct DeviceInfo {
    std::string   name;            ///< "NVIDIA GeForce RTX 4060"
    std::string   driver;          ///< "NVIDIA", "radv", "llvmpipe"
    DeviceKind    kind = DeviceKind::other;

    /// The DRM nodes this GPU owns, from VK_EXT_physical_device_drm.
    /// Both are invalid for a software device — which is how llvmpipe is
    /// identified, rather than by matching on its name.
    DeviceNumber  render_node{};
    DeviceNumber  primary_node{};

    /// Can it do what a compositor needs: import dmabufs, export fences?
    /// False means it is a perfectly good GPU for something else.
    bool          usable = false;

    /// Why not, when usable is false. A compositor prints this rather than
    /// saying "no GPU" on a machine that visibly has one.
    std::string   unusable_because{};

    [[nodiscard]] bool is_software() const noexcept { return kind == DeviceKind::software; }
};

/// Every GPU Vulkan can see, whether usable or not.
///
/// Returns the unusable ones too, with a reason. A compositor that silently
/// skipped them would tell a user with a working card that there is no GPU.
[[nodiscard]] Result<std::vector<DeviceInfo>> enumerate_gpus();

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

    /// The texture, for the renderer to bind.
    [[nodiscard]] TextureId texture() const noexcept { return texture_; }

    /// As a draw-list source, so a compositor does not restate the three
    /// facts that must agree with the image (size, format, orientation) and
    /// get one of them wrong.
    [[nodiscard]] Texture as_texture() const noexcept {
        return Texture{texture_, size_, format_, y_invert_};
    }

    /// Copy the pixels into main memory as premultiplied ARGB32.
    ///
    /// The slow path, and deliberately so: it exists for the CPU renderer,
    /// for screen capture and for the parity test, not for compositing.
    /// `dst_stride_px` is in PIXELS, unlike every framebuffer API's byte
    /// stride — which is why it is in the name.
    ///
    /// Works whatever the layout. That is the point of going through the GPU
    /// rather than mapping the memory: a tiled buffer cannot be memcpy'd,
    /// and this is how its pixels are obtained at all.
    virtual Status read(std::uint32_t* dst, std::int32_t dst_stride_px) = 0;

    /// Put pixels INTO the image, from premultiplied ARGB32.
    ///
    /// This IS the hot path, unlike read() above. Every shared-memory client
    /// becomes a texture this way — foot, and every GTK app — once per
    /// commit, on the compositor's loop thread. A fullscreen terminal
    /// redrawing at 60 Hz calls it 60 times a second, so its cost is a
    /// direct tax on how many busy windows a compositor can show. See
    /// tests/bench_upload.cpp; the numbers are measured, not assumed.
    ///
    /// A client's GPU buffer is never written this way; it is imported and
    /// sampled where it already is.
    ///
    /// `src_stride_px` is in PIXELS, for the same reason as above.
    virtual Status write(const std::uint32_t* src, std::int32_t src_stride_px) = 0;

    /// The same, but only the rows in [y, y + height).
    ///
    /// A client that tells us what it changed should not have the whole
    /// surface re-uploaded: wl_surface.damage on a terminal is usually one
    /// line of it. `src` still points at the START of the buffer and
    /// `src_stride_px` still describes the whole thing, so the caller does no
    /// pointer arithmetic and cannot get the offset wrong.
    ///
    /// Rows, not a rectangle, because a partial-width copy costs one command
    /// per row while a row range is one: the win is in not touching the
    /// untouched 90% of a scrolling terminal, and that is already rows.
    ///
    /// The default does a full write, so an implementation may ignore this.
    virtual Status write_rows(const std::uint32_t* src, std::int32_t src_stride_px,
                              std::int32_t y, std::int32_t height) {
        (void)y;
        (void)height;
        return write(src, src_stride_px);
    }

protected:
    Image() = default;

    weft::Size<weft::Buffer> size_{};
    Format                   format_{};
    TextureId                texture_{};
    bool                     y_invert_ = false;
};

// ---------------------------------------------------------------------------
// Device — one GPU, opened.
// ---------------------------------------------------------------------------

/// A GPU, ready to import and draw.
///
/// Abstract so the Vulkan implementation lives in one .cpp with the Vulkan
/// headers, and so a test can substitute a device that imports nothing. The
/// alternative — a concrete class full of opaque handles — is what the EGL
/// backend was, and it made every test of the import path need a GPU.
class Device {
public:
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    /// Open the best usable GPU.
    ///
    /// "Best" is discrete, then integrated, then software. A compositor that
    /// wants a specific one uses the DeviceNumber overload; this is for the
    /// ordinary single-GPU case, which must not require a machine's worth of
    /// enumeration code in every caller.
    [[nodiscard]] static Result<std::unique_ptr<Device>> open();

    /// Open the GPU that owns this DRM node.
    ///
    /// The multi-GPU entry point, and the one a compositor should use: the
    /// number comes from heddle's enumeration, and matching on it rather
    /// than on a device name is what picks the right card on a machine with
    /// two of the same model.
    [[nodiscard]] static Result<std::unique_ptr<Device>> open(DeviceNumber drm_node);

    /// Open a device for tests, allowing a software one.
    ///
    /// Separate because a compositor must never silently fall back to
    /// llvmpipe — a desktop rendering at 4 fps with no explanation is worse
    /// than one that says it found no GPU. A test, meanwhile, wants
    /// llvmpipe: it runs everywhere and it is a real Vulkan implementation.
    [[nodiscard]] static Result<std::unique_ptr<Device>> open_any();

    /// What this device is, and which kernel nodes it owns.
    [[nodiscard]] virtual const DeviceInfo& info() const noexcept = 0;

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

    /// The layouts BOTH devices can handle, for a format.
    ///
    /// The multi-GPU negotiation, in one place. A client's buffer must be
    /// importable by the GPU that composites AND by the GPU that scans out;
    /// advertising the intersection means that holds by construction rather
    /// than by luck.
    ///
    /// An empty result is meaningful: the two GPUs share no layout, so every
    /// frame needs a copy. A compositor should say so once, in a log line —
    /// one that silently copies every frame is slow for reasons nobody can
    /// find, and one that does not notice renders black.
    [[nodiscard]] std::vector<Layout> shared_layouts(const Device& other,
                                                     Format format) const;

    /// Import a client's buffer.
    ///
    /// `d` is checked by dye::validate first: this reports only what the
    /// driver knows (can it take this layout), not what arithmetic can
    /// answer. Takes the description BY VALUE because importing consumes the
    /// descriptors.
    [[nodiscard]] virtual Result<std::unique_ptr<Image>> import(BufferDescription d) = 0;

    /// Allocate a buffer this GPU can definitely import.
    ///
    /// A compositor needs this for its own surfaces (a cursor, a scratch
    /// target), and a test needs it because a hand-made buffer may be in a
    /// layout the driver refuses.
    ///
    /// The layout is the driver's choice from what it supports, and the
    /// returned description says which. Asking for LINEAR specifically is
    /// what does not work on NVIDIA through EGL; through Vulkan it does, but
    /// letting the driver choose is still right — it picks something it can
    /// render to efficiently.
    [[nodiscard]] virtual Result<BufferDescription> allocate(std::int32_t width,
                                                             std::int32_t height,
                                                             Format format) = 0;

    // -- the seam with the renderer -----------------------------------------
    //
    // The renderer needs this device's actual Vulkan objects, and nothing
    // else does. Rather than make Device concrete (which would drag the
    // Vulkan headers into every file that mentions a GPU) or make the
    // renderer a friend (which would tie their lifetimes together), the
    // handles are exposed through one opaque struct.
    //
    // `Handles` is declared but not defined here: only src/*.cpp ever sees
    // its members, so this header still costs nothing to include.

    /// This device's Vulkan objects.
    ///
    /// Deliberately opaque POINTERS rather than Vulkan types: this header
    /// must not include vulkan.h, or every file that mentions a GPU pays
    /// for it. src/renderer.cpp casts them back, and it is the only file
    /// that may.
    struct Handles {
        void*         device = nullptr;           // VkDevice
        void*         graphics_queue = nullptr;   // VkQueue
        std::uint32_t graphics_family = 0;

        /// The image view a TextureId names, or null if this device never
        /// issued it. An id from ANOTHER device must not resolve: rendering
        /// with it would read whatever that number happens to hit.
        std::function<void*(TextureId)> view_of;
        /// The image behind one of our own Images, for use as a target.
        std::function<void*(Image&)> image_of;
    };

    /// This device's Vulkan objects, or null on a build without Vulkan.
    [[nodiscard]] virtual Handles* vulkan_handles() noexcept = 0;

    /// A memory type satisfying `bits` with all of `properties`.
    ///
    /// Exposed because the renderer allocates its own instance buffer and
    /// must make the same choice this device would. The bug it exists to
    /// avoid: host-visible memory on a discrete GPU is uncached by default,
    /// and reading it is 100x slower than reading cached memory.
    [[nodiscard]] virtual std::optional<std::uint32_t> memory_type_index(
        std::uint32_t bits, std::uint32_t properties) const noexcept = 0;

protected:
    Device() = default;
};

/// Is a GPU backend compiled in at all?
///
/// False in a build without Vulkan. A compositor checks this rather than
/// calling open() and interpreting the error, so "no Vulkan in this build"
/// and "no GPU in this machine" stay distinguishable.
[[nodiscard]] bool gpu_available() noexcept;

}  // namespace dye
