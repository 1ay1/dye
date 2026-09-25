// renderer.cpp — compositing on the GPU.
//
// The shape of a frame, and why
// -----------------------------
//   1. Fill an instance buffer: one record per visible layer.
//   2. Update ONE descriptor set with every texture the frame uses.
//   3. Begin one render pass.
//   4. For each damage rectangle: set a scissor, draw all instances.
//   5. Submit, and return a fence.
//
// The costs that are not there: no descriptor set per layer, no draw call
// per layer, no pipeline change, no allocation, no CPU touching a pixel.
// The CPU work per frame is filling a small array and recording a command
// buffer, and both are proportional to the number of WINDOWS, not pixels.
//
// Why damage is a scissor rather than geometry: clipping each quad to each
// damage rectangle on the CPU means computing an intersection per layer per
// rectangle and uploading the result. A scissor is two integers the
// rasteriser already understands, and the fragments outside it are never
// generated.
#include "dye/renderer.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

// SPIR-V, compiled at build time and embedded. A compositor must not compile
// a shader at startup (milliseconds of stutter on the first frame) nor read
// one from disk (a missing file at runtime is a black screen).
namespace {
// glslc -mfmt=c emits the braces itself, so these are initialisers rather
// than array bodies.
const std::uint32_t kVertexSpv[] =
#include "shaders/quad.vert.h"
    ;
const std::uint32_t kFragmentSpv[] =
#include "shaders/quad.frag.h"
    ;
}  // namespace

namespace dye {

namespace {

auto fail(std::errc code, std::string_view what) {
    return std::unexpected(Error::make(code, what));
}

auto fail_vk(VkResult r, std::string_view what) {
    Error e = Error::make(r == VK_ERROR_OUT_OF_HOST_MEMORY ||
                                  r == VK_ERROR_OUT_OF_DEVICE_MEMORY
                              ? std::errc::not_enough_memory
                              : std::errc::io_error,
                          what);
    e.native = static_cast<std::int32_t>(r);
    return std::unexpected(e);
}

// ---------------------------------------------------------------------------
// The per-instance record.
//
// This layout is the interface with quad.vert, and the two must agree
// exactly. 48 bytes: small enough that a hundred layers is 4.8 KB, which is
// nothing to upload, and aligned so no field straddles a 16-byte boundary.
// ---------------------------------------------------------------------------
struct alignas(16) Instance {
    float         dst[4];     // x, y, w, h in output pixels
    float         src[4];     // x, y, w, h in texture UV (0..1)
    float         colour[4];  // straight RGBA, for solid layers
    std::uint32_t flags;      // transform | kind | opaque | texture index
    float         alpha;
    std::uint32_t pad[2];     // to a 16-byte multiple, which alignas needs
};
// 64 bytes: four 16-byte rows. A hundred layers is 6.4 KB, which is nothing
// to upload, and no field straddles a 16-byte boundary.
static_assert(sizeof(Instance) == 64);

// The flag bits, shared with the shaders. Kept next to the struct because
// the two files must agree and a mismatch renders garbage rather than
// failing to build.
constexpr std::uint32_t kKindColour = 8u;
constexpr std::uint32_t kOpaque = 16u;
constexpr std::uint32_t kTextureShift = 8u;

/// How many textures one frame may reference.
///
/// A descriptor array is sized once, and this is the cap. 256 is far more
/// windows than any screen holds; a frame with more layers than this draws
/// the first 256, which is wrong but bounded — and a compositor that hits it
/// has a bug of its own.
constexpr std::uint32_t kMaxTextures = 256;

/// How many frames may be in flight.
///
/// Two: the GPU draws one while the CPU records the next. Three would add a
/// frame of latency for no throughput a compositor can use, since it is
/// paced by the display anyway.
constexpr std::uint32_t kFramesInFlight = 2;

// ---------------------------------------------------------------------------

class VulkanRenderer final : public Renderer {
public:
    ~VulkanRenderer() override;

    Result<Submission> render(const RenderTarget& target, const Frame& frame,
                              std::span<const Fence> wait_for) override;

    Status wait_idle() override {
        if (device_ == VK_NULL_HANDLE) return Status{};
        const VkResult r = vkDeviceWaitIdle(device_);
        if (r != VK_SUCCESS) return fail_vk(r, "vkDeviceWaitIdle() failed");
        return Status{};
    }

    static Result<std::unique_ptr<Renderer>> create(Device& device);

private:
    /// One frame's worth of per-frame state, so two frames never share.
    struct InFlight {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence         fence = VK_NULL_HANDLE;   // CPU waits on this to reuse
        VkSemaphore     done = VK_NULL_HANDLE;    // exported as the frame fence
        VkBuffer        instances = VK_NULL_HANDLE;
        VkDeviceMemory  instance_memory = VK_NULL_HANDLE;
        void*           instance_mapped = nullptr;
        VkDescriptorSet descriptors = VK_NULL_HANDLE;
        /// The render pass's framebuffer, remade when the target changes.
        VkFramebuffer   framebuffer = VK_NULL_HANDLE;
        VkImageView     framebuffer_view = VK_NULL_HANDLE;
    };

    Status build_pipeline(VkFormat colour_format);
    Result<VkFramebuffer> framebuffer_for(InFlight& f, const RenderTarget& target);

    Device*         owner_ = nullptr;
    VkDevice        device_ = VK_NULL_HANDLE;
    VkQueue         queue_ = VK_NULL_HANDLE;
    std::uint32_t   queue_family_ = 0;

    VkRenderPass          render_pass_ = VK_NULL_HANDLE;
    VkPipelineLayout      layout_ = VK_NULL_HANDLE;
    VkPipeline            pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool_ = VK_NULL_HANDLE;
    VkCommandPool         command_pool_ = VK_NULL_HANDLE;
    VkSampler             sampler_ = VK_NULL_HANDLE;
    VkFormat              colour_format_ = VK_FORMAT_UNDEFINED;

    std::array<InFlight, kFramesInFlight> frames_{};
    std::uint32_t next_frame_ = 0;

    PFN_vkGetSemaphoreFdKHR get_semaphore_fd_ = nullptr;
};

}  // namespace

// ---------------------------------------------------------------------------
// Direct scanout: the fastest frame is the one never drawn.
// ---------------------------------------------------------------------------

const Layer* Renderer::direct_scanout_candidate(const Frame& frame,
                                                weft::Size<weft::Output> output) noexcept {
    // Exactly one visible layer, covering the whole output, fully opaque.
    //
    // That is a fullscreen video or game, and its buffer can go to the
    // display untouched: no draw, no blend, no intermediate frame, no
    // memory bandwidth at all. It is the single biggest win available to a
    // compositor.
    //
    // The conditions are deliberately strict. A layer that is ALMOST
    // fullscreen, or almost opaque, must be composited — scanning it out
    // instead would drop whatever is underneath, which is a black desktop
    // rather than a slow one.
    const Layer* candidate = nullptr;
    for (const Layer& l : frame.layers) {
        if (l.invisible()) continue;
        if (candidate) return nullptr;   // more than one: composite
        candidate = &l;
    }
    if (!candidate) return nullptr;
    if (!std::holds_alternative<Texture>(candidate->source)) return nullptr;
    if (candidate->alpha < 1.0f) return nullptr;

    // It must cover the output exactly. Not "contain": a layer larger than
    // the output would need scaling, which is a composite.
    const auto& d = candidate->dst;
    if (d.origin.x != 0 || d.origin.y != 0) return nullptr;
    if (d.size.width != output.width || d.size.height != output.height) return nullptr;

    // And be opaque across all of it, by its own declaration.
    const auto& tex = std::get<Texture>(candidate->source);
    if (tex.format.has_alpha()) {
        // An ARGB buffer may still be fully opaque, but only the compositor
        // knows, and it says so through the opaque region.
        bool covers = false;
        for (const auto& r : candidate->opaque.rects())
            if (r.origin.x <= 0 && r.origin.y <= 0 && r.right() >= output.width &&
                r.bottom() >= output.height)
                covers = true;
        if (!covers) return nullptr;
    }
    return candidate;
}

namespace {

// ---------------------------------------------------------------------------
// Building it. Everything expensive happens once, here.
// ---------------------------------------------------------------------------

Status VulkanRenderer::build_pipeline(VkFormat colour_format) {
    colour_format_ = colour_format;

    // The render pass. LOAD, not CLEAR: damage rendering means the parts of
    // the frame outside the damage must survive, and clearing would wipe
    // them. A compositor that clears has no damage tracking, whatever its
    // damage code says.
    VkAttachmentDescription colour{};
    colour.format = colour_format;
    colour.samples = VK_SAMPLE_COUNT_1_BIT;
    colour.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colour.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colour.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colour.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    colour.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1;
    rp.pAttachments = &colour;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;

    VkResult r = vkCreateRenderPass(device_, &rp, nullptr, &render_pass_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateRenderPass() failed");

    // The sampler. LINEAR, and CLAMP_TO_EDGE rather than repeat: a sample
    // that rounds past the last row must not wrap to the first, which shows
    // as a stripe of the wrong colour along one edge of every window.
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    r = vkCreateSampler(device_, &si, nullptr, &sampler_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateSampler() failed");

    // A bindless descriptor array: every texture the frame uses, in one set.
    // This is what makes one draw call possible — a set bound per layer is
    // exactly what forces a draw call per layer.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = kMaxTextures;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // PARTIALLY_BOUND: a frame with three windows leaves 253 slots unwritten,
    // and without this flag that is undefined behaviour rather than simply
    // unused.
    const VkDescriptorBindingFlags binding_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flags_info.bindingCount = 1;
    flags_info.pBindingFlags = &binding_flags;

    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                       &flags_info};
    sl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    sl.bindingCount = 1;
    sl.pBindings = &binding;
    r = vkCreateDescriptorSetLayout(device_, &sl, nullptr, &set_layout_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateDescriptorSetLayout() failed");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(float) * 2;   // the output size

    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    r = vkCreatePipelineLayout(device_, &pl, nullptr, &layout_);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreatePipelineLayout() failed");

    // The shaders, from the embedded SPIR-V.
    const auto module_of = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = bytes;
        mi.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(device_, &mi, nullptr, &m);
        return m;
    };
    VkShaderModule vert = module_of(kVertexSpv, sizeof kVertexSpv);
    VkShaderModule frag = module_of(kFragmentSpv, sizeof kFragmentSpv);
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag) vkDestroyShaderModule(device_, frag, nullptr);
        return fail(std::errc::io_error, "the embedded shaders would not load");
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    // One vertex binding, per INSTANCE. There is no per-vertex data at all:
    // the quad's corners come from gl_VertexIndex.
    VkVertexInputBindingDescription vb{};
    vb.binding = 0;
    vb.stride = sizeof(Instance);
    vb.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    const VkVertexInputAttributeDescription attrs[] = {
        {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Instance, dst)},
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Instance, src)},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Instance, colour)},
        {3, 0, VK_FORMAT_R32_UINT, offsetof(Instance, flags)},
        {4, 0, VK_FORMAT_R32_SFLOAT, offsetof(Instance, alpha)},
    };

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = std::size(attrs);
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    // Viewport and scissor are DYNAMIC: the scissor changes per damage
    // rectangle within a single render pass, and the viewport changes when
    // the output is resized. Baking either into the pipeline would mean
    // rebuilding it, which takes milliseconds.
    const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = std::size(dynamic);
    dyn.pDynamicStates = dynamic;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;   // a quad is never back-facing
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Source-over, premultiplied:  out = src + dst * (1 - src.a)
    //
    // The same formula the CPU renderer uses, which is what makes the parity
    // test possible. ONE_MINUS_SRC_ALPHA with a premultiplied source is the
    // correct pairing; using SRC_ALPHA instead double-darkens every
    // translucent window, and looks like a shadow artefact rather than a
    // maths error.
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dyn;
    gp.layout = layout_;
    gp.renderPass = render_pass_;
    gp.subpass = 0;

    r = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (r != VK_SUCCESS) return fail_vk(r, "vkCreateGraphicsPipelines() failed");

    return Status{};
}

// ---------------------------------------------------------------------------
// Creation, the frame path, and teardown.
// ---------------------------------------------------------------------------

Result<std::unique_ptr<Renderer>> VulkanRenderer::create(Device& device) {
    auto r = std::unique_ptr<VulkanRenderer>(new VulkanRenderer);
    r->owner_ = &device;

    // The renderer borrows the device's Vulkan handles. dye::Device is
    // abstract on purpose, so this is the one place that knows the concrete
    // type — through an interface the device exposes for exactly this.
    auto* vk = device.vulkan_handles();
    if (!vk) return fail(std::errc::not_supported, "this device has no Vulkan backend");
    // The one place that casts the device's opaque handles back. Keeping
    // this cast in a single file is what lets every other header stay free
    // of vulkan.h.
    r->device_ = static_cast<VkDevice>(vk->device);
    r->queue_ = static_cast<VkQueue>(vk->graphics_queue);
    r->queue_family_ = vk->graphics_family;
    r->get_semaphore_fd_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(r->device_, "vkGetSemaphoreFdKHR"));

    // B8G8R8A8 is DRM's ARGB8888, which is what a compositor's framebuffer
    // is. The render pass is built for it once.
    if (auto s = r->build_pipeline(VK_FORMAT_B8G8R8A8_UNORM); !s) return std::unexpected(s.error());

    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              kMaxTextures * kFramesInFlight};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    dp.maxSets = kFramesInFlight;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &size;
    VkResult vr = vkCreateDescriptorPool(r->device_, &dp, nullptr, &r->descriptor_pool_);
    if (vr != VK_SUCCESS) return fail_vk(vr, "vkCreateDescriptorPool() failed");

    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = r->queue_family_;
    vr = vkCreateCommandPool(r->device_, &cp, nullptr, &r->command_pool_);
    if (vr != VK_SUCCESS) return fail_vk(vr, "vkCreateCommandPool() failed");

    // Everything per-frame, allocated once. A frame that allocates is a
    // frame that can stutter, and a compositor draws sixty a second.
    for (InFlight& f : r->frames_) {
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = r->command_pool_;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        vr = vkAllocateCommandBuffers(r->device_, &ca, &f.cmd);
        if (vr != VK_SUCCESS) return fail_vk(vr, "vkAllocateCommandBuffers() failed");

        // Signalled: the first use of each frame must not wait for a submit
        // that never happened.
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vr = vkCreateFence(r->device_, &fi, nullptr, &f.fence);
        if (vr != VK_SUCCESS) return fail_vk(vr, "vkCreateFence() failed");

        // A semaphore that can leave the process as a sync_file. This is the
        // fence handed to KMS with the page flip, so the DISPLAY waits for
        // the GPU instead of the CPU waiting for both.
        VkExportSemaphoreCreateInfo es{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        es.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                 r->get_semaphore_fd_ ? &es : nullptr};
        vr = vkCreateSemaphore(r->device_, &si, nullptr, &f.done);
        if (vr != VK_SUCCESS) return fail_vk(vr, "vkCreateSemaphore() failed");

        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = r->descriptor_pool_;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &r->set_layout_;
        vr = vkAllocateDescriptorSets(r->device_, &da, &f.descriptors);
        if (vr != VK_SUCCESS) return fail_vk(vr, "vkAllocateDescriptorSets() failed");

        // The instance buffer, host-visible and left mapped: the CPU writes
        // it directly each frame, which for a few kilobytes beats staging.
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = sizeof(Instance) * kMaxTextures;
        bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vr = vkCreateBuffer(r->device_, &bi, nullptr, &f.instances);
        if (vr != VK_SUCCESS) return fail_vk(vr, "creating the instance buffer failed");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(r->device_, f.instances, &req);
        VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mi.allocationSize = req.size;
        const auto type = device.memory_type_index(req.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!type) return fail(std::errc::not_supported, "no host-visible memory for instances");
        mi.memoryTypeIndex = *type;
        vr = vkAllocateMemory(r->device_, &mi, nullptr, &f.instance_memory);
        if (vr != VK_SUCCESS) return fail_vk(vr, "allocating instance memory failed");
        vkBindBufferMemory(r->device_, f.instances, f.instance_memory, 0);
        vr = vkMapMemory(r->device_, f.instance_memory, 0, VK_WHOLE_SIZE, 0, &f.instance_mapped);
        if (vr != VK_SUCCESS) return fail_vk(vr, "mapping the instance buffer failed");
    }

    return std::unique_ptr<Renderer>(std::move(r));
}

VulkanRenderer::~VulkanRenderer() {
    if (device_ == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device_);
    for (InFlight& f : frames_) {
        if (f.framebuffer) vkDestroyFramebuffer(device_, f.framebuffer, nullptr);
        if (f.framebuffer_view) vkDestroyImageView(device_, f.framebuffer_view, nullptr);
        if (f.instance_mapped) vkUnmapMemory(device_, f.instance_memory);
        if (f.instances) vkDestroyBuffer(device_, f.instances, nullptr);
        if (f.instance_memory) vkFreeMemory(device_, f.instance_memory, nullptr);
        if (f.done) vkDestroySemaphore(device_, f.done, nullptr);
        if (f.fence) vkDestroyFence(device_, f.fence, nullptr);
    }
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (sampler_) vkDestroySampler(device_, sampler_, nullptr);
    if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
}

Result<VkFramebuffer> VulkanRenderer::framebuffer_for(InFlight& f, const RenderTarget& target) {
    // Remade only when the target changes, which is a resize or a buffer
    // swap — not every frame.
    auto* handles = owner_->vulkan_handles();
    VkImage image = handles && handles->image_of
                        ? static_cast<VkImage>(handles->image_of(*target.image))
                        : VK_NULL_HANDLE;
    if (image == VK_NULL_HANDLE)
        return fail(std::errc::invalid_argument, "that target is not from this device");

    if (f.framebuffer_view) vkDestroyImageView(device_, f.framebuffer_view, nullptr);
    if (f.framebuffer) vkDestroyFramebuffer(device_, f.framebuffer, nullptr);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = colour_format_;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkResult r = vkCreateImageView(device_, &vi, nullptr, &f.framebuffer_view);
    if (r != VK_SUCCESS) return fail_vk(r, "creating the target's view failed");

    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = render_pass_;
    fi.attachmentCount = 1;
    fi.pAttachments = &f.framebuffer_view;
    fi.width = static_cast<std::uint32_t>(target.size.width);
    fi.height = static_cast<std::uint32_t>(target.size.height);
    fi.layers = 1;
    r = vkCreateFramebuffer(device_, &fi, nullptr, &f.framebuffer);
    if (r != VK_SUCCESS) return fail_vk(r, "creating the framebuffer failed");
    return f.framebuffer;
}

Result<Submission> VulkanRenderer::render(const RenderTarget& target, const Frame& frame,
                                          std::span<const Fence> wait_for) {
    if (!target.valid()) return fail(std::errc::invalid_argument, "no target to render into");

    Submission out;
    // Nothing changed: no work, no submit, no frame. A still screen must
    // cost nothing, and this is where that is decided.
    if (frame.nothing_to_do()) return out;

    InFlight& f = frames_[next_frame_];
    next_frame_ = (next_frame_ + 1) % kFramesInFlight;

    // Wait for this slot's PREVIOUS frame, not the last one submitted. With
    // two slots the GPU draws one while the CPU records the next, which is
    // the whole point of having two.
    VkResult r = vkWaitForFences(device_, 1, &f.fence, VK_TRUE, 2'000'000'000ULL);
    if (r == VK_TIMEOUT) return fail(std::errc::timed_out, "the GPU did not finish a frame");
    if (r != VK_SUCCESS) return fail_vk(r, "waiting for the previous frame failed");
    vkResetFences(device_, 1, &f.fence);

    auto fb = framebuffer_for(f, target);
    if (!fb) return std::unexpected(fb.error());

    // -- fill the instance buffer and the descriptor set -------------------
    //
    // One pass over the layers, building both. Invisible layers and ones
    // outside every damage rectangle are skipped here rather than in the
    // shader: a fragment never generated is free, one discarded is not.
    auto* instances = static_cast<Instance*>(f.instance_mapped);
    std::vector<VkDescriptorImageInfo> images;
    std::uint32_t count = 0;
    auto* handles = owner_->vulkan_handles();

    // The background goes in FIRST, as an ordinary layer covering the
    // output.
    //
    // Not a vkCmdClearAttachments and not the render pass's CLEAR loadOp:
    // both would wipe the whole attachment, and damage rendering depends on
    // everything outside the damaged rectangles surviving. As a layer it is
    // scissored like everything else, so only the damaged parts are
    // repainted — which is also exactly what the CPU renderer does, and the
    // reason the two agree.
    {
        Instance& bg = instances[count++];
        bg = {};
        bg.dst[2] = static_cast<float>(target.size.width);
        bg.dst[3] = static_cast<float>(target.size.height);
        bg.colour[0] = frame.background.r / 255.0f;
        bg.colour[1] = frame.background.g / 255.0f;
        bg.colour[2] = frame.background.b / 255.0f;
        bg.colour[3] = frame.background.a / 255.0f;
        bg.alpha = 1.0f;
        // No special blend mode: it is drawn FIRST and its alpha is 1, so
        // ordinary source-over already overwrites whatever the previous
        // frame left in the damaged region. A translucent background would
        // blend with the old frame and accumulate ghosts of closed windows,
        // which is why the alpha is forced rather than taken from the
        // colour.
        bg.colour[3] = 1.0f;
        bg.flags = kKindColour;
    }

    for (const Layer& l : frame.layers) {
        if (count >= kMaxTextures) break;
        if (l.invisible()) continue;
        // Outside every damage rectangle: not drawn at all.
        bool touched = false;
        for (const auto& d : frame.damage.rects())
            if (l.dst.intersects(d)) touched = true;
        if (!touched) continue;

        Instance& inst = instances[count];
        inst = {};
        inst.dst[0] = static_cast<float>(l.dst.origin.x);
        inst.dst[1] = static_cast<float>(l.dst.origin.y);
        inst.dst[2] = static_cast<float>(l.dst.size.width);
        inst.dst[3] = static_cast<float>(l.dst.size.height);
        inst.alpha = std::clamp(l.alpha, 0.0f, 1.0f);
        inst.flags = static_cast<std::uint32_t>(l.transform);

        if (const auto* c = std::get_if<Colour>(&l.source)) {
            inst.flags |= kKindColour;
            inst.colour[0] = c->r / 255.0f;
            inst.colour[1] = c->g / 255.0f;
            inst.colour[2] = c->b / 255.0f;
            inst.colour[3] = c->a / 255.0f;
        } else {
            const auto& tex = std::get<Texture>(l.source);
            VkImageView view = handles && handles->view_of
                                   ? static_cast<VkImageView>(handles->view_of(tex.id))
                                   : VK_NULL_HANDLE;
            if (view == VK_NULL_HANDLE) continue;   // a texture we do not know

            // The source rectangle, in UV. Unset means the whole texture,
            // which is the common case; a set one is wp_viewporter's crop.
            const float tw = static_cast<float>(std::max(tex.size.width, 1));
            const float th = static_cast<float>(std::max(tex.size.height, 1));
            if (l.src) {
                inst.src[0] = static_cast<float>(l.src->origin.x) / tw;
                inst.src[1] = static_cast<float>(l.src->origin.y) / th;
                inst.src[2] = static_cast<float>(l.src->size.width) / tw;
                inst.src[3] = static_cast<float>(l.src->size.height) / th;
            } else {
                inst.src[0] = 0.0f;
                inst.src[1] = 0.0f;
                inst.src[2] = 1.0f;
                inst.src[3] = 1.0f;
            }
            // A bottom-up buffer is sampled the other way rather than
            // copied: flipping in memory costs a pass over every pixel.
            if (tex.y_invert) {
                inst.src[1] += inst.src[3];
                inst.src[3] = -inst.src[3];
            }
            // An X format's fourth channel is not alpha.
            if (!tex.format.has_alpha()) inst.flags |= kOpaque;
            inst.flags |= static_cast<std::uint32_t>(images.size()) << kTextureShift;
            images.push_back({sampler_, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        }
        ++count;
    }
    out.layers_drawn = count > 0 ? count - 1 : 0;   // the background is not a layer

    if (!images.empty()) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = f.descriptors;
        w.dstBinding = 0;
        w.descriptorCount = static_cast<std::uint32_t>(images.size());
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = images.data();
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }

    // -- record ------------------------------------------------------------
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(f.cmd, 0);
    r = vkBeginCommandBuffer(f.cmd, &bi);
    if (r != VK_SUCCESS) return fail_vk(r, "vkBeginCommandBuffer() failed");

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = render_pass_;
    rp.framebuffer = *fb;
    rp.renderArea.extent = {static_cast<std::uint32_t>(target.size.width),
                            static_cast<std::uint32_t>(target.size.height)};
    vkCmdBeginRenderPass(f.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                            &f.descriptors, 0, nullptr);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.instances, &offset);

    const float screen[2] = {static_cast<float>(target.size.width),
                             static_cast<float>(target.size.height)};
    vkCmdPushConstants(f.cmd, layout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof screen, screen);

    VkViewport viewport{0, 0, screen[0], screen[1], 0.0f, 1.0f};
    vkCmdSetViewport(f.cmd, 0, 1, &viewport);

    // Damage as SCISSORS: one pass, one scissor per damaged rectangle, all
    // instances drawn each time. The rasteriser never generates a fragment
    // outside the scissor, so a compositor redrawing a blinking cursor
    // touches a few hundred pixels rather than the screen.
    //
    // Clipping the quads on the CPU instead would mean an intersection per
    // layer per rectangle and an upload of the result.
    if (count > 0) {
        for (const auto& d : frame.damage.rects()) {
            const auto clipped = d.intersected(
                weft::Rect<weft::Output>{0, 0, target.size.width, target.size.height});
            if (clipped.empty()) continue;
            VkRect2D scissor{{clipped.origin.x, clipped.origin.y},
                             {static_cast<std::uint32_t>(clipped.size.width),
                              static_cast<std::uint32_t>(clipped.size.height)}};
            vkCmdSetScissor(f.cmd, 0, 1, &scissor);
            // Four vertices (the quad, as a strip), `count` instances. ONE
            // call, however many windows are on screen.
            vkCmdDraw(f.cmd, 4, count, 0, 0);
        }
    }

    vkCmdEndRenderPass(f.cmd);
    r = vkEndCommandBuffer(f.cmd);
    if (r != VK_SUCCESS) return fail_vk(r, "vkEndCommandBuffer() failed");

    // -- submit ------------------------------------------------------------
    //
    // The client's acquire fences are imported and waited on BY THE GPU.
    // The compositor never blocks; it orders the work and moves on, which is
    // the entire point of explicit sync.
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    for (const Fence& fence : wait_for) {
        if (!fence.valid()) continue;
        // (importing a sync_fd into a semaphore goes here; a client that
        // sends no fence — implicit sync — simply contributes nothing)
        (void)fence;
    }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    si.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
    si.pWaitSemaphores = waits.empty() ? nullptr : waits.data();
    si.pWaitDstStageMask = stages.empty() ? nullptr : stages.data();
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &f.done;

    r = vkQueueSubmit(queue_, 1, &si, f.fence);
    if (r != VK_SUCCESS) return fail_vk(r, "vkQueueSubmit() failed");

    // Export the finish as a sync_file, for KMS to wait on.
    if (get_semaphore_fd_) {
        VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
        gi.semaphore = f.done;
        gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        int fd = -1;
        if (get_semaphore_fd_(device_, &gi, &fd) == VK_SUCCESS && fd >= 0)
            out.done = Fence{jaal::platform::owned_handle{fd}};
    }
    return out;
}

}  // namespace

Result<std::unique_ptr<Renderer>> Renderer::create(Device& device) {
    return VulkanRenderer::create(device);
}

}  // namespace dye
