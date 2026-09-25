// device.cpp — Vulkan on a DRM render node.
//
// The only file in dye that includes a Vulkan header. Everything above works
// in terms of Device and Image, which is what lets the rest of the library
// compile and be tested on a machine with no Vulkan at all.
//
// No window, no swapchain, no surface. A compositor draws into buffers it
// then hands to KMS; asking Vulkan for a surface here is how this fails on a
// headless machine that is otherwise perfectly able to render.
#include "dye/device.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>

namespace dye {

namespace {

// ---------------------------------------------------------------------------
// Errors.
//
// Every Vulkan call returns a VkResult, which is most of why this API is
// easier to be correct in than GL's global error flag. Mapping them to
// std::errc keeps dye's errors in one vocabulary (jaal's) while preserving
// the specific code in `native` for anyone who needs it.
// ---------------------------------------------------------------------------

std::errc errc_of(VkResult r) noexcept {
    switch (r) {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:      return std::errc::not_enough_memory;
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
        case VK_ERROR_FEATURE_NOT_PRESENT:
        case VK_ERROR_EXTENSION_NOT_PRESENT:     return std::errc::not_supported;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:   return std::errc::bad_file_descriptor;
        case VK_ERROR_DEVICE_LOST:               return std::errc::io_error;
        default:                                 return std::errc::io_error;
    }
}

auto fail(std::errc code, std::string_view what) {
    return std::unexpected(Error::make(code, what));
}

auto fail_vk(VkResult r, std::string_view what) {
    Error e = Error::make(errc_of(r), what);
    e.native = static_cast<std::int32_t>(r);
    return std::unexpected(e);
}

// ---------------------------------------------------------------------------
// Format mapping.
//
// DRM fourccs and Vulkan formats describe the same bytes in opposite word
// order, which is exactly the kind of thing that is wrong in one place and
// then wrong everywhere. One table, used in both directions.
//
// DRM names channels in little-endian memory order read as a 32-bit word;
// Vulkan names them in memory order. So DRM's ARGB8888 — B,G,R,A in memory —
// is Vulkan's B8G8R8A8.
// ---------------------------------------------------------------------------

struct FormatPair {
    Format   drm;
    VkFormat vk;
};

const FormatPair kFormats[] = {
    {formats::argb8888, VK_FORMAT_B8G8R8A8_UNORM},
    {formats::xrgb8888, VK_FORMAT_B8G8R8A8_UNORM},   // the X byte is ignored on sample
    {formats::abgr8888, VK_FORMAT_R8G8B8A8_UNORM},
    {formats::xbgr8888, VK_FORMAT_R8G8B8A8_UNORM},
};

VkFormat vk_format_of(Format f) noexcept {
    for (const FormatPair& p : kFormats)
        if (p.drm == f) return p.vk;
    return VK_FORMAT_UNDEFINED;
}

// ---------------------------------------------------------------------------
// The instance.
//
// One per process, created once. Vulkan allows several but there is no
// reason for a compositor to have more, and creating one per Device would
// make enumerate_gpus() and open() disagree about what exists.
// ---------------------------------------------------------------------------

struct Instance {
    VkInstance handle = VK_NULL_HANDLE;
    Error      failure = Error::make(std::errc::not_supported, "Vulkan was never initialised");
    bool       ok = false;

    // The function pointers for extensions that are not in core. Loaded once
    // here rather than at every call site, which is where a forgotten null
    // check turns into a crash.
    PFN_vkGetMemoryFdKHR                       get_memory_fd = nullptr;
    PFN_vkGetImageDrmFormatModifierPropertiesEXT get_image_modifier = nullptr;
};

Instance& instance() {
    static Instance inst = [] {
        Instance i;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "dye";
        // 1.2 for timeline semaphores in core; 1.3 is not required, so a
        // machine on an older loader still works.
        app.apiVersion = VK_API_VERSION_1_2;

        // VK_KHR_get_physical_device_properties2 is core in 1.1, so nothing
        // instance-level is needed beyond what the loader gives us. Asking
        // for extensions that are already core is how initialisation fails
        // on a strict loader.
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;

        const VkResult r = vkCreateInstance(&ci, nullptr, &i.handle);
        if (r != VK_SUCCESS) {
            i.failure = Error::make(errc_of(r), "vkCreateInstance() failed");
            i.failure.native = static_cast<std::int32_t>(r);
            return i;
        }
        i.ok = true;
        return i;
    }();
    return inst;
}

// ---------------------------------------------------------------------------
// Enumeration.
// ---------------------------------------------------------------------------

/// The device extensions a compositor cannot work without.
///
/// Checked before the device is created, so "this GPU cannot do what we
/// need" is a clear message rather than a failure three calls later.
constexpr std::array<const char*, 5> kRequiredDeviceExtensions{
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,        // import a dmabuf
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,   // ...as a dmabuf specifically
    VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME, // with a known tiling
    VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,         // required by the above
    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,      // a buffer another device wrote
};

/// Extensions that are wanted but not fatal. Explicit sync lives here: a GPU
/// without it still composites, it just cannot do the good thing.
constexpr std::array<const char*, 2> kOptionalDeviceExtensions{
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME,
};

bool has_extension(const std::vector<VkExtensionProperties>& have, const char* want) {
    return std::any_of(have.begin(), have.end(), [&](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, want) == 0;
    });
}

std::vector<VkExtensionProperties> extensions_of(VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> out(n);
    if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, out.data());
    return out;
}

DeviceKind kind_of(VkPhysicalDeviceType t) noexcept {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return DeviceKind::discrete;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return DeviceKind::integrated;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return DeviceKind::software;
        default:                                     return DeviceKind::other;
    }
}

/// Everything about one physical device, before deciding to open it.
struct Candidate {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    DeviceInfo       info{};
    std::vector<const char*> extensions{};   // required + whatever optional it has
    bool             has_external_semaphore_fd = false;
};

std::vector<Candidate> candidates() {
    std::vector<Candidate> out;
    Instance& inst = instance();
    if (!inst.ok) return out;

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst.handle, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) vkEnumeratePhysicalDevices(inst.handle, &n, devs.data());

    for (VkPhysicalDevice pd : devs) {
        Candidate c;
        c.handle = pd;

        const auto exts = extensions_of(pd);

        // The DRM nodes, if the driver reports them. This is the link to
        // /dev/dri that makes multi-GPU correct; a driver without the
        // extension leaves both invalid, and the device can then only be
        // chosen as "the best one" rather than by node.
        VkPhysicalDeviceDrmPropertiesEXT drm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        if (has_extension(exts, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) props.pNext = &drm;
        vkGetPhysicalDeviceProperties2(pd, &props);

        c.info.name = props.properties.deviceName;
        c.info.kind = kind_of(props.properties.deviceType);
        if (props.pNext && drm.hasRender)
            c.info.render_node = {static_cast<std::uint32_t>(drm.renderMajor),
                                  static_cast<std::uint32_t>(drm.renderMinor)};
        if (props.pNext && drm.hasPrimary)
            c.info.primary_node = {static_cast<std::uint32_t>(drm.primaryMajor),
                                   static_cast<std::uint32_t>(drm.primaryMinor)};

        VkPhysicalDeviceDriverProperties driver{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           &driver};
        vkGetPhysicalDeviceProperties2(pd, &props2);
        c.info.driver = driver.driverName;

        // Missing required extensions is the common reason a real GPU is
        // unusable, and the reason has to name WHICH one — "no GPU" on a
        // machine with a working card is the least helpful message there is.
        std::string missing;
        for (const char* want : kRequiredDeviceExtensions) {
            if (has_extension(exts, want)) {
                c.extensions.push_back(want);
            } else {
                if (!missing.empty()) missing += ", ";
                missing += want;
            }
        }
        for (const char* want : kOptionalDeviceExtensions)
            if (has_extension(exts, want)) {
                c.extensions.push_back(want);
                if (std::strcmp(want, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0)
                    c.has_external_semaphore_fd = true;
            }

        if (!missing.empty()) {
            c.info.usable = false;
            c.info.unusable_because = "missing " + missing;
        } else {
            c.info.usable = true;
        }
        out.push_back(std::move(c));
    }
    return out;
}

/// Which queue family can do graphics? Also reports a transfer-only one, so
/// uploads can run without stalling the renderer.
struct Queues {
    std::uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
};

Queues queues_of(VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    if (n) vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());

    Queues q;
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto flags = fams[i].queueFlags;
        if (q.graphics == VK_QUEUE_FAMILY_IGNORED && (flags & VK_QUEUE_GRAPHICS_BIT))
            q.graphics = i;
        // Transfer-ONLY: a family that also does graphics is the one we are
        // trying to avoid loading. This machine has such a family (2 queues),
        // and using it is what keeps an upload from stalling a frame.
        if (q.transfer == VK_QUEUE_FAMILY_IGNORED && (flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & VK_QUEUE_GRAPHICS_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT))
            q.transfer = i;
    }
    // Every graphics queue can transfer, so falling back to it is correct,
    // just less parallel.
    if (q.transfer == VK_QUEUE_FAMILY_IGNORED) q.transfer = q.graphics;
    return q;
}

// ---------------------------------------------------------------------------
// VkDevice
// ---------------------------------------------------------------------------

class VulkanDevice;

class VulkanImage final : public Image {
public:
    VulkanImage(VulkanDevice& device, VkImage image, VkDeviceMemory memory, VkImageView view)
        : device_(device), image_(image), memory_(memory), view_(view) {}

    ~VulkanImage() override;

    Status read(std::uint32_t* dst, std::int32_t dst_stride_px) override;

    [[nodiscard]] VkImage vk_image() const noexcept { return image_; }
    [[nodiscard]] VkImageView vk_view() const noexcept { return view_; }

private:
    friend class VulkanDevice;

    VulkanDevice&  device_;
    VkImage        image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView    view_ = VK_NULL_HANDLE;
    /// Has anything transitioned this image out of UNDEFINED yet? An image
    /// read before it is ever drawn into must still be transitioned, and
    /// doing it twice is a validation error.
    bool           laid_out_ = false;
};

class VulkanDevice final : public Device {
public:
    ~VulkanDevice() override {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);
        if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }

    const DeviceInfo& info() const noexcept override { return info_; }
    std::span<const Layout> formats() const noexcept override { return formats_; }

    Result<std::unique_ptr<Image>> import(BufferDescription d) override;
    Result<BufferDescription> allocate(std::int32_t width, std::int32_t height,
                                       Format format) override;

    static Result<std::unique_ptr<Device>> create(Candidate c);

    // -- used by VulkanImage ------------------------------------------------
    VkDevice vk() const noexcept { return device_; }
    VkPhysicalDevice physical() const noexcept { return physical_; }

    /// Run one command buffer and wait for it.
    ///
    /// Only for read-back and layout transitions, both of which are already
    /// off the hot path. The renderer submits its own work without waiting.
    Status run_now(const std::function<void(VkCommandBuffer)>& record);

    /// A memory type satisfying `bits` with these properties, or nothing.
    std::optional<std::uint32_t> memory_type(std::uint32_t bits,
                                             VkMemoryPropertyFlags want) const;

private:
    friend class VulkanImage;

    /// Everything this GPU can import, asked per format and per modifier.
    ///
    /// Vulkan answers this properly: for each modifier it reports the plane
    /// count AND the feature flags, so "can I sample this layout" is settled
    /// before a buffer exists. EGL's equivalent query reports modifiers it
    /// then refuses to import — which is the measurement that chose this API.
    void query_formats();

    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice         device_ = VK_NULL_HANDLE;
    VkQueue          graphics_queue_ = VK_NULL_HANDLE;
    VkQueue          transfer_queue_ = VK_NULL_HANDLE;
    Queues           families_{};
    VkCommandPool    command_pool_ = VK_NULL_HANDLE;
    DeviceInfo       info_{};
    std::vector<Layout> formats_;
    VkPhysicalDeviceMemoryProperties memory_{};

    PFN_vkGetMemoryFdKHR get_memory_fd_ = nullptr;
};

// ---------------------------------------------------------------------------
// VulkanDevice: the implementation.
// ---------------------------------------------------------------------------

void VulkanDevice::query_formats() {
    for (const Format& f : formats::all) {
        const VkFormat vk = vk_format_of(f);
        if (vk == VK_FORMAT_UNDEFINED) continue;

        VkDrmFormatModifierPropertiesListEXT list{
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
        VkFormatProperties2 props{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &list};
        vkGetPhysicalDeviceFormatProperties2(physical_, vk, &props);
        if (list.drmFormatModifierCount == 0) continue;

        std::vector<VkDrmFormatModifierPropertiesEXT> mods(list.drmFormatModifierCount);
        list.pDrmFormatModifierProperties = mods.data();
        vkGetPhysicalDeviceFormatProperties2(physical_, vk, &props);

        for (const auto& m : mods) {
            // Single-plane only: every format dye composites is one plane,
            // and importing a multi-plane buffer without the sampling code
            // for it would accept buffers that then render as garbage.
            if (m.drmFormatModifierPlaneCount != 1) continue;

            // It must be SAMPLABLE. This is the check EGL's query cannot
            // express, and the reason it advertises layouts it then refuses:
            // a modifier the driver knows about is not necessarily one it
            // can read in a shader.
            if (!(m.drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
                continue;

            formats_.push_back(Layout{f, Modifier{m.drmFormatModifier}});
        }
    }
}

std::optional<std::uint32_t> VulkanDevice::memory_type(std::uint32_t bits,
                                                       VkMemoryPropertyFlags want) const {
    for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        if ((memory_.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return std::nullopt;
}

Status VulkanDevice::run_now(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = command_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult r = vkAllocateCommandBuffers(device_, &ai, &cmd);
    if (r != VK_SUCCESS) return fail_vk(r, "vkAllocateCommandBuffers() failed");

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(cmd, &bi);
    if (r != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        return fail_vk(r, "vkBeginCommandBuffer() failed");
    }

    record(cmd);

    r = vkEndCommandBuffer(cmd);
    if (r != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        return fail_vk(r, "vkEndCommandBuffer() failed");
    }

    // A fence rather than vkQueueWaitIdle: waiting on the whole queue stalls
    // work that has nothing to do with this, which on a compositor is the
    // frame currently being drawn.
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    r = vkCreateFence(device_, &fi, nullptr, &fence);
    if (r != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
        return fail_vk(r, "vkCreateFence() failed");
    }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    r = vkQueueSubmit(graphics_queue_, 1, &si, fence);
    if (r == VK_SUCCESS) {
        // Two seconds. Not forever: a wedged GPU must not hang a compositor
        // with no way out, and a read-back that takes two seconds has
        // already failed at its job.
        r = vkWaitForFences(device_, 1, &fence, VK_TRUE, 2'000'000'000ULL);
    }

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);

    if (r == VK_TIMEOUT) return fail(std::errc::timed_out, "the GPU did not finish in time");
    if (r != VK_SUCCESS) return fail_vk(r, "submitting to the GPU failed");
    return Status{};
}

Result<std::unique_ptr<Image>> VulkanDevice::import(BufferDescription d) {
    // Arithmetic first, driver second. A malformed buffer must produce the
    // same error on every vendor, and only this order gives that.
    if (const BadBuffer bad = validate(d); bad != BadBuffer::ok)
        return fail(std::errc::invalid_argument, describe(bad));

    if (!supports(d.layout))
        return fail(std::errc::not_supported,
                    "this GPU cannot import that format and layout");

    const VkFormat vk = vk_format_of(d.layout.format);

    // The plane layouts, exactly as the client described them. Vulkan takes
    // these explicitly rather than inferring, which is what lets a buffer
    // allocated by another process be imported at all.
    std::array<VkSubresourceLayout, kMaxPlanes> planes{};
    for (std::size_t i = 0; i < d.planes.size(); ++i) {
        planes[i].offset = d.planes[i].offset;
        planes[i].rowPitch = d.planes[i].stride;
    }

    VkImageDrmFormatModifierExplicitCreateInfoEXT drm{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
    drm.drmFormatModifier = d.layout.modifier.raw();
    drm.drmFormatModifierPlaneCount = static_cast<std::uint32_t>(d.planes.size());
    drm.pPlaneLayouts = planes.data();

    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                        &drm};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = vk;
    ici.extent = {static_cast<std::uint32_t>(d.width), static_cast<std::uint32_t>(d.height), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // UNDEFINED, always: the contents are whatever the client wrote, and
    // claiming any other layout tells the driver it may discard them.
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VkResult r = vkCreateImage(device_, &ici, nullptr, &image);
    if (r != VK_SUCCESS) return fail_vk(r, "the driver refused this dmabuf's layout");

    // From here every failure must destroy what came before it. A compositor
    // imports thousands of buffers; one leaked VkImage per malformed one is
    // a slow death rather than a crash.
    struct Cleanup {
        VkDevice dev;
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        bool armed = true;
        ~Cleanup() {
            if (!armed) return;
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (img) vkDestroyImage(dev, img, nullptr);
            if (mem) vkFreeMemory(dev, mem, nullptr);
        }
    } cleanup{device_, image};

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, image, &req);

    // The descriptor is DUPLICATED, not moved: Vulkan takes ownership of the
    // fd it is given, and the BufferDescription still owns the original.
    // Handing over the same number would mean two owners closing it, and the
    // second close lands on whatever the process opened next.
    auto dup = d.planes.front().fd.duplicate();
    if (!dup) return std::unexpected(dup.error());

    VkImportMemoryFdInfoKHR import_fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import_fd.fd = dup->release();   // Vulkan owns it now, even on failure

    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                            &import_fd};
    dedicated.image = image;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedicated};
    mai.allocationSize = req.size;
    const auto type = memory_type(req.memoryTypeBits, 0);
    if (!type) return fail(std::errc::not_supported, "no memory type accepts this buffer");
    mai.memoryTypeIndex = *type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    r = vkAllocateMemory(device_, &mai, nullptr, &memory);
    if (r != VK_SUCCESS) return fail_vk(r, "importing the dmabuf's memory failed");
    cleanup.mem = memory;

    r = vkBindImageMemory(device_, image, memory, 0);
    if (r != VK_SUCCESS) return fail_vk(r, "binding the imported memory failed");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = vk;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageView view = VK_NULL_HANDLE;
    r = vkCreateImageView(device_, &vci, nullptr, &view);
    if (r != VK_SUCCESS) return fail_vk(r, "creating a view of the imported image failed");
    cleanup.view = view;

    auto out = std::make_unique<VulkanImage>(*this, image, memory, view);
    out->size_ = {d.width, d.height};
    out->format_ = d.layout.format;
    out->y_invert_ = d.y_invert;
    out->texture_ = TextureId{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(view))};

    cleanup.armed = false;
    // `d` dies here, and with it the client's descriptors. The driver has
    // its own duplicate; keeping the client's would pin its memory for as
    // long as this image lived, and a hundred windows meant a hundred
    // descriptors and eventually EMFILE.
    return out;
}

Result<BufferDescription> VulkanDevice::allocate(std::int32_t width, std::int32_t height,
                                                 Format format) {
    if (width <= 0 || height <= 0 || width > kMaxDimension || height > kMaxDimension)
        return fail(std::errc::invalid_argument, "that is not a buffer size");

    const VkFormat vk = vk_format_of(format);
    if (vk == VK_FORMAT_UNDEFINED)
        return fail(std::errc::invalid_argument, "unsupported pixel format");

    // Every layout this GPU takes for this format, offered to the driver so
    // it picks its own best. Naming one specifically is how this fails on a
    // GPU whose preference differs.
    std::vector<std::uint64_t> mods;
    for (const Layout& l : formats_)
        if (l.format == format) mods.push_back(l.modifier.raw());
    if (mods.empty())
        return fail(std::errc::not_supported, "this GPU does not handle that format");

    VkImageDrmFormatModifierListCreateInfoEXT mod_list{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
    mod_list.drmFormatModifierCount = static_cast<std::uint32_t>(mods.size());
    mod_list.pDrmFormatModifiers = mods.data();

    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                        &mod_list};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = vk;
    ici.extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VkResult r = vkCreateImage(device_, &ici, nullptr, &image);
    if (r != VK_SUCCESS) return fail_vk(r, "the driver could not allocate that image");

    struct Cleanup {
        VkDevice dev;
        VkImage img;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        ~Cleanup() {
            if (img) vkDestroyImage(dev, img, nullptr);
            if (mem) vkFreeMemory(dev, mem, nullptr);
        }
    } cleanup{device_, image};

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, image, &req);

    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                            &export_info};
    dedicated.image = image;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedicated};
    mai.allocationSize = req.size;
    const auto type = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!type) return fail(std::errc::not_enough_memory, "no device-local memory for this image");
    mai.memoryTypeIndex = *type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    r = vkAllocateMemory(device_, &mai, nullptr, &memory);
    if (r != VK_SUCCESS) return fail_vk(r, "allocating image memory failed");
    cleanup.mem = memory;

    r = vkBindImageMemory(device_, image, memory, 0);
    if (r != VK_SUCCESS) return fail_vk(r, "binding image memory failed");

    // Which layout did the driver actually choose? Asking rather than
    // assuming: a description that claims the wrong modifier is a buffer
    // nobody else can import.
    VkImageDrmFormatModifierPropertiesEXT chosen{
        VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
    auto get_modifier = reinterpret_cast<PFN_vkGetImageDrmFormatModifierPropertiesEXT>(
        vkGetDeviceProcAddr(device_, "vkGetImageDrmFormatModifierPropertiesEXT"));
    if (!get_modifier)
        return fail(std::errc::not_supported, "the driver cannot report an image's modifier");
    r = get_modifier(device_, image, &chosen);
    if (r != VK_SUCCESS) return fail_vk(r, "asking the image's modifier failed");

    // And where the planes landed.
    VkImageSubresource sub{};
    sub.aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;
    VkSubresourceLayout layout{};
    vkGetImageSubresourceLayout(device_, image, &sub, &layout);

    // Export the memory as a dmabuf the rest of the world can use.
    VkMemoryGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gfi.memory = memory;
    gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    int fd = -1;
    r = get_memory_fd_(device_, &gfi, &fd);
    if (r != VK_SUCCESS) return fail_vk(r, "exporting the buffer as a dmabuf failed");

    BufferDescription out;
    out.width = width;
    out.height = height;
    out.layout = Layout{format, Modifier{chosen.drmFormatModifier}};
    Plane p;
    p.fd = jaal::platform::owned_handle{fd};
    p.offset = static_cast<std::uint32_t>(layout.offset);
    p.stride = static_cast<std::uint32_t>(layout.rowPitch);
    out.planes.push_back(std::move(p));

    // The image and memory go; the dmabuf keeps the allocation alive. Vulkan
    // objects with a different lifetime from the buffer they describe are
    // the shape of every double-free here.
    // (cleanup runs on scope exit and destroys both)

    if (const BadBuffer bad = validate(out); bad != BadBuffer::ok)
        return fail(std::errc::io_error, describe(bad));
    return out;
}

// ---------------------------------------------------------------------------
// VulkanImage
// ---------------------------------------------------------------------------

VulkanImage::~VulkanImage() {
    VkDevice dev = device_.vk();
    if (dev == VK_NULL_HANDLE) return;
    if (view_) vkDestroyImageView(dev, view_, nullptr);
    if (image_) vkDestroyImage(dev, image_, nullptr);
    if (memory_) vkFreeMemory(dev, memory_, nullptr);
}

Status VulkanImage::read(std::uint32_t* dst, std::int32_t dst_stride_px) {
    if (!dst) return fail(std::errc::invalid_argument, "read() needs somewhere to write");
    if (size_.empty()) return fail(std::errc::invalid_argument, "the image has no pixels");
    if (dst_stride_px < size_.width)
        return fail(std::errc::invalid_argument, "the destination stride is too small");

    VkDevice dev = device_.vk();
    const auto w = static_cast<std::uint32_t>(size_.width);
    const auto h = static_cast<std::uint32_t>(size_.height);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;

    // A host-visible staging buffer. The image itself is very likely tiled
    // and device-local, so it cannot be mapped — copying through a buffer is
    // not a detour, it is the only way.
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(dev, &bci, nullptr, &staging);
    if (r != VK_SUCCESS) return fail_vk(r, "creating a staging buffer failed");

    struct Cleanup {
        VkDevice dev;
        VkBuffer buf;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        ~Cleanup() {
            if (buf) vkDestroyBuffer(dev, buf, nullptr);
            if (mem) vkFreeMemory(dev, mem, nullptr);
        }
    } cleanup{dev, staging};

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, staging, &req);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    const auto type = device_.memory_type(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!type) return fail(std::errc::not_supported, "no host-visible memory for read-back");
    mai.memoryTypeIndex = *type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    r = vkAllocateMemory(dev, &mai, nullptr, &memory);
    if (r != VK_SUCCESS) return fail_vk(r, "allocating staging memory failed");
    cleanup.mem = memory;

    r = vkBindBufferMemory(dev, staging, memory, 0);
    if (r != VK_SUCCESS) return fail_vk(r, "binding staging memory failed");

    // The transition is from UNDEFINED the first time (the client's contents
    // are preserved by TRANSFER_SRC, which is what UNDEFINED -> TRANSFER_SRC
    // guarantees) and from TRANSFER_SRC afterwards.
    const VkImageLayout old_layout =
        laid_out_ ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;

    const auto status = device_.run_now([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_src.oldLayout = old_layout;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        // FOREIGN: the buffer was last touched by another device or process
        // (the client that drew it). Saying so is what makes the contents
        // visible to us rather than undefined.
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = image_;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_src.srcAccessMask = 0;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &to_src);

        VkBufferImageCopy copy{};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;    // tightly packed
        copy.bufferImageHeight = 0;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                               &copy);
    });
    if (!status) return status;
    laid_out_ = true;

    void* mapped = nullptr;
    r = vkMapMemory(dev, memory, 0, bytes, 0, &mapped);
    if (r != VK_SUCCESS) return fail_vk(r, "mapping the staging buffer failed");

    // Vulkan gave us the bytes in the image's own order; the renderer wants
    // premultiplied ARGB32. B8G8R8A8 is already that order on a
    // little-endian host, R8G8B8A8 needs red and blue swapped.
    const auto* src = static_cast<const std::uint8_t*>(mapped);
    const bool swap_rb = !is_rgb(format_.order());
    const bool opaque = !format_.has_alpha();

    for (std::uint32_t y = 0; y < h; ++y) {
        // Unlike GL, Vulkan's origin is the TOP left, so a read-back is the
        // right way up unless the client said otherwise.
        const std::uint32_t src_row = y_invert_ ? h - 1 - y : y;
        const std::uint8_t* in = src + static_cast<std::size_t>(src_row) * w * 4;
        std::uint32_t* out = dst + static_cast<std::size_t>(y) * dst_stride_px;
        for (std::uint32_t x = 0; x < w; ++x, in += 4) {
            const std::uint32_t b = swap_rb ? in[2] : in[0];
            const std::uint32_t g = in[1];
            const std::uint32_t rr = swap_rb ? in[0] : in[2];
            const std::uint32_t a = opaque ? 0xffu : in[3];
            out[x] = a << 24 | rr << 16 | g << 8 | b;
        }
    }

    vkUnmapMemory(dev, memory);
    return Status{};
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

Result<std::unique_ptr<Device>> VulkanDevice::create(Candidate c) {
    if (!c.info.usable)
        return fail(std::errc::not_supported, "this GPU cannot import dmabufs");

    auto d = std::unique_ptr<VulkanDevice>(new VulkanDevice);
    d->physical_ = c.handle;
    d->info_ = c.info;
    d->families_ = queues_of(c.handle);

    if (d->families_.graphics == VK_QUEUE_FAMILY_IGNORED)
        return fail(std::errc::not_supported, "this GPU has no graphics queue");

    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = d->families_.graphics;
        q.queueCount = 1;
        q.pQueuePriorities = &priority;
        queue_infos.push_back(q);
        if (d->families_.transfer != d->families_.graphics) {
            q.queueFamilyIndex = d->families_.transfer;
            queue_infos.push_back(q);
        }
    }

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
    dci.pQueueCreateInfos = queue_infos.data();
    dci.enabledExtensionCount = static_cast<std::uint32_t>(c.extensions.size());
    dci.ppEnabledExtensionNames = c.extensions.data();

    VkResult r = vkCreateDevice(c.handle, &dci, nullptr, &d->device_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateDevice() failed");

    vkGetDeviceQueue(d->device_, d->families_.graphics, 0, &d->graphics_queue_);
    vkGetDeviceQueue(d->device_, d->families_.transfer, 0, &d->transfer_queue_);
    vkGetPhysicalDeviceMemoryProperties(c.handle, &d->memory_);

    d->get_memory_fd_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(d->device_, "vkGetMemoryFdKHR"));
    if (!d->get_memory_fd_)
        return fail(std::errc::not_supported, "the driver cannot export memory as an fd");

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = d->families_.graphics;
    r = vkCreateCommandPool(d->device_, &pci, nullptr, &d->command_pool_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateCommandPool() failed");

    d->query_formats();
    if (d->formats_.empty())
        return fail(std::errc::not_supported,
                    "this GPU imports none of the formats dye handles");

    return std::unique_ptr<Device>(std::move(d));
}

}  // namespace

// ---------------------------------------------------------------------------
// The public entry points.
// ---------------------------------------------------------------------------

Result<std::vector<DeviceInfo>> enumerate_gpus() {
    Instance& inst = instance();
    if (!inst.ok) return std::unexpected(inst.failure);

    std::vector<DeviceInfo> out;
    for (const Candidate& c : candidates()) out.push_back(c.info);
    if (out.empty()) return fail(std::errc::no_such_device, "Vulkan found no GPU");
    return out;
}

Result<std::unique_ptr<Device>> Device::open(DeviceNumber drm_node) {
    if (!drm_node.valid())
        return fail(std::errc::invalid_argument, "that is not a device number");

    Instance& inst = instance();
    if (!inst.ok) return std::unexpected(inst.failure);

    for (Candidate& c : candidates()) {
        // Either node identifies the GPU: a caller holding a primary node
        // (from heddle, for scanout) must find the same device as one
        // holding the render node.
        if (c.info.render_node == drm_node || c.info.primary_node == drm_node)
            return VulkanDevice::create(std::move(c));
    }
    return fail(std::errc::no_such_device, "no Vulkan device owns that DRM node");
}

namespace {

/// Pick between several GPUs. Discrete, then integrated, then anything, and
/// software only if allowed.
Result<std::unique_ptr<Device>> open_best(bool allow_software) {
    Instance& inst = instance();
    if (!inst.ok) return std::unexpected(inst.failure);

    auto all = candidates();
    if (all.empty()) return fail(std::errc::no_such_device, "Vulkan found no GPU");

    const auto rank = [](const DeviceInfo& i) {
        switch (i.kind) {
            case DeviceKind::discrete:   return 0;
            case DeviceKind::integrated: return 1;
            case DeviceKind::other:      return 2;
            case DeviceKind::software:   return 3;
        }
        return 4;
    };

    std::stable_sort(all.begin(), all.end(), [&](const Candidate& a, const Candidate& b) {
        return rank(a.info) < rank(b.info);
    });

    // The first error is the one reported: it is about the best device, and
    // so the most likely to be what the user cares about.
    std::optional<Error> first_error;
    for (Candidate& c : all) {
        if (c.info.is_software() && !allow_software) continue;
        if (!c.info.usable) {
            if (!first_error)
                first_error = Error::make(std::errc::not_supported,
                                          "the GPU is missing extensions dye needs");
            continue;
        }
        auto d = VulkanDevice::create(std::move(c));
        if (d) return d;
        if (!first_error) first_error = d.error();
    }

    if (first_error) return std::unexpected(*first_error);
    return fail(std::errc::no_such_device,
                allow_software ? "no usable Vulkan device"
                               : "no usable GPU (only a software renderer was found)");
}

}  // namespace

// A compositor must never silently fall back to llvmpipe: a desktop at 4 fps
// with no explanation is worse than one that says it found no GPU.
Result<std::unique_ptr<Device>> Device::open() { return open_best(false); }

// A test, meanwhile, wants llvmpipe — it runs everywhere and is a real
// Vulkan implementation.
Result<std::unique_ptr<Device>> Device::open_any() { return open_best(true); }

std::vector<Layout> Device::shared_layouts(const Device& other, Format format) const {
    std::vector<Layout> out;
    for (const Layout& mine : formats()) {
        if (mine.format != format) continue;
        if (other.supports(mine)) out.push_back(mine);
    }
    return out;
}

bool gpu_available() noexcept { return true; }

}  // namespace dye
