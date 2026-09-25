// device.cpp — EGL on a render node.
//
// The only file in dye that includes a GPU header. Everything above it works
// in terms of Device and Image, which is what lets the rest of the library
// compile and be tested on a machine with no EGL at all.
//
// A render node (/dev/dri/renderD128) rather than a card node: no display, no
// master, no permission beyond the `render` group. A compositor that only
// wants to import and draw does not need to own a screen, and asking for one
// it does not need is how a nested compositor fails to start.
#include "dye/device.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace dye {

namespace {

auto fail(std::errc code, std::string_view what) {
    return std::unexpected(Error::make(code, what));
}
auto fail_errno(std::string_view what) {
    return std::unexpected(Error::from_errno(errno, what));
}

bool has_extension(const char* list, const char* name) {
    if (!list) return false;
    const std::size_t n = std::strlen(name);
    for (const char* p = list; (p = std::strstr(p, name)) != nullptr; p += n) {
        // A substring match is not enough: "EGL_KHR_image" matches inside
        // "EGL_KHR_image_base", and a compositor that believes it has an
        // extension it does not have crashes on the first call.
        const bool starts = p == list || p[-1] == ' ';
        const bool ends = p[n] == ' ' || p[n] == '\0';
        if (starts && ends) return true;
    }
    return false;
}

template <class F>
F proc(const char* name) {
    return reinterpret_cast<F>(eglGetProcAddress(name));
}

/// Render nodes, in order. renderD128 is the first; a machine with two GPUs
/// has 129 as well, and which one is wanted is the caller's business.
std::vector<std::string> render_nodes() {
    std::vector<std::string> nodes;
    if (DIR* d = ::opendir("/dev/dri")) {
        while (auto* e = ::readdir(d))
            if (std::strncmp(e->d_name, "renderD", 7) == 0)
                nodes.emplace_back(std::string("/dev/dri/") + e->d_name);
        ::closedir(d);
    }
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

// ---------------------------------------------------------------------------
// GlImage
// ---------------------------------------------------------------------------

class GlDevice;

class GlImage final : public Image {
public:
    GlImage(GlDevice& device, EGLDisplay display, EGLImage image, GLuint texture)
        : device_(device), display_(display), image_(image) {
        texture_ = TextureId{texture};
    }

    ~GlImage() override;

    Status read(std::uint32_t* dst, std::int32_t dst_stride_px) override;

private:
    friend class GlDevice;

    GlDevice&  device_;
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLImage   image_ = EGL_NO_IMAGE;
    GLuint     fbo_ = 0;
    /// Scratch for read(): GL gives back RGBA bytes and the renderer wants
    /// premultiplied ARGB32. Kept per image rather than per call so a
    /// compositor reading every frame does not allocate every frame.
    std::vector<std::uint8_t> scratch_;
};

// ---------------------------------------------------------------------------
// GlDevice
// ---------------------------------------------------------------------------

class GlDevice final : public Device {
public:
    ~GlDevice() override {
        if (display_ != EGL_NO_DISPLAY) {
            eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
            eglTerminate(display_);
        }
        if (gbm_) gbm_device_destroy(gbm_);
        // The descriptor last: gbm and EGL both reference it, and closing it
        // first leaves them talking to whatever opens next.
        fd_.reset();
    }

    std::string_view node() const noexcept override { return node_; }
    std::uint64_t device_id() const noexcept override { return device_; }
    std::span<const Layout> formats() const noexcept override { return formats_; }

    Result<std::unique_ptr<Image>> import(BufferDescription d) override {
        // Arithmetic first, driver second. A malformed buffer must produce
        // the same error on every vendor, and only this order gives that.
        if (const BadBuffer bad = validate(d); bad != BadBuffer::ok)
            return fail(std::errc::invalid_argument, describe(bad));

        if (!supports(d.layout))
            return fail(std::errc::not_supported,
                        "this GPU cannot import that format and layout");

        if (!make_current())
            return fail(std::errc::io_error, "eglMakeCurrent() failed");

        // Single-plane only, which every format dye handles is. A multi-plane
        // YUV import needs the other plane attributes, and adding the
        // attributes without the sampling code would import buffers that
        // then render as garbage.
        const Plane& p = d.planes.front();
        const std::uint64_t mod = d.layout.modifier.raw();
        const EGLAttrib attrs[] = {
            EGL_WIDTH,                          d.width,
            EGL_HEIGHT,                         d.height,
            EGL_LINUX_DRM_FOURCC_EXT,           static_cast<EGLAttrib>(d.layout.format.fourcc()),
            EGL_DMA_BUF_PLANE0_FD_EXT,          p.fd.get(),
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,      static_cast<EGLAttrib>(p.offset),
            EGL_DMA_BUF_PLANE0_PITCH_EXT,       static_cast<EGLAttrib>(p.stride),
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, static_cast<EGLAttrib>(mod & 0xffffffffu),
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, static_cast<EGLAttrib>(mod >> 32),
            EGL_NONE};

        EGLImage egl_image =
            eglCreateImage(display_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attrs);
        if (egl_image == EGL_NO_IMAGE)
            return fail(std::errc::invalid_argument, "the driver refused this dmabuf");

        GLuint texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        // Clamp, not repeat: a sample that rounds past the last row must not
        // wrap to the first, which shows as a stripe of the wrong colour
        // along one edge of every window.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        image_target_texture_(GL_TEXTURE_2D, static_cast<GLeglImageOES>(egl_image));
        const GLenum err = glGetError();
        glBindTexture(GL_TEXTURE_2D, 0);

        if (err != GL_NO_ERROR) {
            glDeleteTextures(1, &texture);
            eglDestroyImage(display_, egl_image);
            return fail(std::errc::io_error, "binding the imported image to a texture failed");
        }

        auto image = std::make_unique<GlImage>(*this, display_, egl_image, texture);
        image->size_ = {d.width, d.height};
        image->format_ = d.layout.format;
        image->y_invert_ = d.y_invert;
        // `d` dies here, and with it the client's descriptors. The driver
        // has what it needs; keeping them would pin the client's memory for
        // as long as this image lived, and a hundred windows meant a hundred
        // descriptors and eventually EMFILE.
        return image;
    }

    Result<BufferDescription> allocate(std::int32_t width, std::int32_t height,
                                       Format format) override {
        if (width <= 0 || height <= 0 || width > kMaxDimension || height > kMaxDimension)
            return fail(std::errc::invalid_argument, "that is not a buffer size");
        if (!format.valid())
            return fail(std::errc::invalid_argument, "unsupported pixel format");

        // Every layout this GPU takes for this format, handed to gbm so the
        // driver picks its own best. Asking for LINEAR specifically is what
        // does not work on NVIDIA — it supports none.
        std::vector<std::uint64_t> mods;
        for (const Layout& l : formats_)
            if (l.format == format) mods.push_back(l.modifier.raw());
        if (mods.empty())
            return fail(std::errc::not_supported, "this GPU does not handle that format");

        gbm_bo* bo = gbm_bo_create_with_modifiers(gbm_, static_cast<std::uint32_t>(width),
                                                  static_cast<std::uint32_t>(height),
                                                  format.fourcc(), mods.data(),
                                                  static_cast<unsigned>(mods.size()));
        if (!bo) return fail(std::errc::io_error, "gbm could not allocate that buffer");

        BufferDescription d;
        d.width = width;
        d.height = height;
        d.layout = Layout{format, Modifier{gbm_bo_get_modifier(bo)}};

        const int plane_count = gbm_bo_get_plane_count(bo);
        for (int i = 0; i < plane_count; ++i) {
            const int raw = gbm_bo_get_fd_for_plane(bo, i);
            if (raw < 0) {
                gbm_bo_destroy(bo);
                return fail_errno("gbm_bo_get_fd_for_plane() failed");
            }
            Plane p;
            p.fd = jaal::platform::owned_handle{raw};
            p.offset = gbm_bo_get_offset(bo, i);
            p.stride = gbm_bo_get_stride_for_plane(bo, i);
            d.planes.push_back(std::move(p));
        }
        // The descriptors are ours now; the bo itself is not needed. Keeping
        // it would mean the caller's buffer had two owners with different
        // lifetimes, which is the shape of every double-free here.
        gbm_bo_destroy(bo);

        if (const BadBuffer bad = validate(d); bad != BadBuffer::ok)
            return fail(std::errc::io_error, describe(bad));
        return d;
    }

    bool make_current() {
        return eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_) == EGL_TRUE;
    }

    static Result<std::unique_ptr<Device>> open_node(const std::string& node);

private:
    friend class GlImage;

    jaal::platform::owned_handle fd_{};
    std::uint64_t                device_ = 0;
    std::string                  node_;
    gbm_device*                  gbm_ = nullptr;
    EGLDisplay                   display_ = EGL_NO_DISPLAY;
    EGLContext                   context_ = EGL_NO_CONTEXT;
    std::vector<Layout>          formats_;

    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture_ = nullptr;
};

GlImage::~GlImage() {
    // Make the context current before deleting GL objects, or they are
    // deleted in whatever context happens to be bound — which on a
    // compositor's teardown path is often none, and the objects leak.
    if (device_.make_current()) {
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        const GLuint tex = texture_.raw();
        if (tex) glDeleteTextures(1, &tex);
    }
    if (image_ != EGL_NO_IMAGE) eglDestroyImage(display_, image_);
}

Status GlImage::read(std::uint32_t* dst, std::int32_t dst_stride_px) {
    if (!dst) return fail(std::errc::invalid_argument, "read() needs somewhere to write");
    if (size_.empty()) return fail(std::errc::invalid_argument, "the image has no pixels");
    if (dst_stride_px < size_.width)
        return fail(std::errc::invalid_argument, "the destination stride is too small");
    if (!device_.make_current())
        return fail(std::errc::io_error, "eglMakeCurrent() failed");

    const auto w = static_cast<std::size_t>(size_.width);
    const auto h = static_cast<std::size_t>(size_.height);

    if (!fbo_) glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           texture_.raw(), 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return fail(std::errc::io_error, "the imported image is not readable as a framebuffer");
    }

    scratch_.resize(w * h * 4);
    // GL_RGBA/GL_UNSIGNED_BYTE is the one combination every GLES2
    // implementation must support. Asking for BGRA works on most drivers and
    // fails on some, silently returning nothing.
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_RGBA,
                 GL_UNSIGNED_BYTE, scratch_.data());
    const GLenum err = glGetError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (err != GL_NO_ERROR) return fail(std::errc::io_error, "glReadPixels() failed");

    // RGBA bytes to premultiplied ARGB32, flipping if the client drew
    // bottom-up. GL's origin is the bottom-left, so a read-back is upside
    // down unless the image says otherwise.
    const bool alpha = format_.has_alpha();
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t src_row = y_invert_ ? y : h - 1 - y;
        const std::uint8_t* src = scratch_.data() + src_row * w * 4;
        std::uint32_t* out = dst + y * static_cast<std::size_t>(dst_stride_px);
        for (std::size_t x = 0; x < w; ++x, src += 4) {
            const std::uint32_t a = alpha ? src[3] : 0xffu;
            out[x] = a << 24 | static_cast<std::uint32_t>(src[0]) << 16 |
                     static_cast<std::uint32_t>(src[1]) << 8 | src[2];
        }
    }
    return Status{};
}

Result<std::unique_ptr<Device>> GlDevice::open_node(const std::string& node) {
    auto get_display = proc<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");
    auto query_formats = proc<PFNEGLQUERYDMABUFFORMATSEXTPROC>("eglQueryDmaBufFormatsEXT");
    auto query_modifiers = proc<PFNEGLQUERYDMABUFMODIFIERSEXTPROC>("eglQueryDmaBufModifiersEXT");
    if (!get_display || !query_formats || !query_modifiers)
        return fail(std::errc::not_supported, "EGL lacks the dmabuf extensions");

    auto g = std::unique_ptr<GlDevice>(new GlDevice);

    const int raw = ::open(node.c_str(), O_RDWR | O_CLOEXEC);
    if (raw < 0) return fail_errno("opening the render node failed");
    g->fd_ = jaal::platform::owned_handle{raw};

    struct stat st {};
    if (::fstat(g->fd_.get(), &st) != 0) return fail_errno("fstat() on the render node failed");
    g->device_ = st.st_rdev;
    g->node_ = node;

    g->gbm_ = gbm_create_device(g->fd_.get());
    if (!g->gbm_) return fail(std::errc::io_error, "gbm_create_device() failed");

    g->display_ = get_display(EGL_PLATFORM_GBM_KHR, g->gbm_, nullptr);
    if (g->display_ == EGL_NO_DISPLAY)
        return fail(std::errc::io_error, "eglGetPlatformDisplay() failed");

    EGLint major = 0, minor = 0;
    if (!eglInitialize(g->display_, &major, &minor))
        return fail(std::errc::io_error, "eglInitialize() failed");

    const char* exts = eglQueryString(g->display_, EGL_EXTENSIONS);
    if (!has_extension(exts, "EGL_EXT_image_dma_buf_import_modifiers"))
        return fail(std::errc::not_supported, "the driver cannot import dmabufs with modifiers");

    if (!eglBindAPI(EGL_OPENGL_ES_API))
        return fail(std::errc::not_supported, "EGL has no GLES support");

    // Surfaceless: a compositor draws into buffers, never into a window, and
    // asking for a window surface here is how this fails on a headless
    // machine that is otherwise perfectly able to render.
    const EGLint ctx_attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    g->context_ = eglCreateContext(g->display_, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
    if (g->context_ == EGL_NO_CONTEXT)
        return fail(std::errc::io_error, "eglCreateContext() failed");
    if (!g->make_current()) return fail(std::errc::io_error, "eglMakeCurrent() failed");

    g->image_target_texture_ =
        proc<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>("glEGLImageTargetTexture2DOES");
    if (!g->image_target_texture_)
        return fail(std::errc::not_supported, "GL cannot bind an EGLImage to a texture");

    // What this GPU can import. Only the formats dye knows how to composite,
    // and only the layouts the driver reports as non-external: an external
    // one needs a different sampler in the shader, and offering it means
    // clients allocate buffers that then render as garbage.
    EGLint format_count = 0;
    query_formats(g->display_, 0, nullptr, &format_count);
    std::vector<EGLint> fourccs(static_cast<std::size_t>(format_count));
    if (format_count > 0)
        query_formats(g->display_, format_count, fourccs.data(), &format_count);

    for (EGLint raw_fourcc : fourccs) {
        const auto format = Format::from_fourcc(static_cast<std::uint32_t>(raw_fourcc));
        if (!format) continue;

        EGLint mod_count = 0;
        query_modifiers(g->display_, raw_fourcc, 0, nullptr, nullptr, &mod_count);
        if (mod_count <= 0) {
            // No modifier list means linear only, which every driver can do.
            g->formats_.push_back(Layout{*format, Modifier::linear()});
            continue;
        }
        std::vector<EGLuint64KHR> mods(static_cast<std::size_t>(mod_count));
        std::vector<EGLBoolean> external(static_cast<std::size_t>(mod_count));
        query_modifiers(g->display_, raw_fourcc, mod_count, mods.data(), external.data(),
                        &mod_count);
        for (EGLint i = 0; i < mod_count; ++i) {
            if (external[static_cast<std::size_t>(i)]) continue;
            g->formats_.push_back(
                Layout{*format, Modifier{static_cast<std::uint64_t>(mods[static_cast<std::size_t>(i)])}});
        }
    }

    if (g->formats_.empty())
        return fail(std::errc::not_supported, "this GPU imports none of the formats dye handles");

    return std::unique_ptr<Device>(std::move(g));
}

}  // namespace

Result<std::unique_ptr<Device>> Device::open(const char* node) {
    return GlDevice::open_node(node);
}

Result<std::unique_ptr<Device>> Device::open() {
    const auto nodes = render_nodes();
    if (nodes.empty())
        return fail(std::errc::no_such_device, "no render node in /dev/dri");

    // The first that works. A machine can have a node it cannot open (a
    // permission problem, a device the driver has not claimed), and giving
    // up on the first is how a two-GPU laptop fails to start.
    Error last = Error::make(std::errc::no_such_device, "no usable render node");
    for (const auto& node : nodes) {
        auto d = GlDevice::open_node(node);
        if (d) return d;
        last = d.error();
    }
    return std::unexpected(last);
}

bool gpu_available() noexcept { return true; }

}  // namespace dye
