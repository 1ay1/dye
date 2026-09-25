#pragma once
// dye/renderer.hpp — compositing on the GPU.
//
// What this replaces
// ------------------
// The path before this was: import a client's buffer, copy it back into
// main memory, and blend it there with the CPU. Measured on a 4060, one
// 1080p window on a 3440x1440 screen:
//
//     read-back      5.6 ms   (after the HOST_CACHED fix; it was 637 ms)
//     CPU composite 22.5 ms
//     ------------------------
//     total         28.1 ms   of a 41.7 ms budget at 23.976 fps
//
// That is a video that plays, drops frames and desynchronises. With this,
// the imported buffer IS the texture: nothing is copied, nothing is blended
// on the CPU, and the per-frame cost is a command buffer and a submit.
//
// How it is fast
// --------------
// Five decisions, each measurable:
//
//   * ONE draw call per frame, not per window. Every layer is an instance;
//     fifty windows is one call with fifty instances.
//   * Bindless textures. A descriptor set bound per layer is what forces a
//     draw call per layer; here the texture index is an instance attribute.
//   * Damage is a SCISSOR, not geometry. One render pass, one scissor per
//     damaged rectangle, and the GPU rasterises only what changed.
//   * Nothing is allocated per frame. Command buffers, descriptor sets and
//     the instance buffer are pooled at startup. A frame that allocates is
//     a frame that can stutter.
//   * The pipeline is built once. No shader compilation, ever, after
//     startup.
//
// What it does NOT do
// -------------------
// It does not present. Putting a finished frame on a screen is KMS's job
// (heddle) or the parent compositor's (the nested backend); this renders
// into an image and says when it is done. Keeping those separate is what
// lets the same renderer serve both backends.

#include <cstdint>
#include <memory>
#include <span>

#include <jaal/core/error.hpp>

#include "device.hpp"
#include "draw.hpp"
#include "fence.hpp"

namespace dye {

/// Somewhere to render into.
///
/// A dye::Image the compositor allocated (Device::allocate) and will then
/// hand to KMS or to a parent compositor. The renderer does not own it: a
/// framebuffer outlives any one frame, and double buffering means there are
/// at least two.
struct RenderTarget {
    Image*                   image = nullptr;
    weft::Size<weft::Output> size{};

    [[nodiscard]] bool valid() const noexcept { return image != nullptr && !size.empty(); }
};

/// What came of submitting a frame.
struct Submission {
    /// Signalled when the GPU has finished drawing.
    ///
    /// Handed to KMS with the page flip so the display waits on the GPU
    /// rather than the CPU waiting on both. Not waiting on this before
    /// scanning out is how a compositor shows half-drawn frames under load.
    Fence done{};

    /// How many layers were actually drawn, after invisible ones and ones
    /// outside the damage were skipped. For logs and for the benchmark; a
    /// number that does not fall when a window is occluded means the
    /// occlusion logic is not working.
    std::uint32_t layers_drawn = 0;
};

/// The GPU compositor.
///
/// One per device, made once, used for every frame. Making one per frame
/// would rebuild the pipeline, which takes milliseconds.
class Renderer {
public:
    virtual ~Renderer() = default;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /// Build a renderer on this device.
    ///
    /// Compiles the pipeline and allocates the pools. Everything expensive
    /// happens here, so that nothing expensive happens per frame.
    [[nodiscard]] static Result<std::unique_ptr<Renderer>> create(Device& device);

    /// Draw one frame.
    ///
    /// `wait_for` are fences the GPU must wait on before reading — a
    /// client's acquire fence, saying "my drawing is finished". Waiting on
    /// the GPU rather than the CPU is the whole point of explicit sync: the
    /// compositor never blocks, it just orders the work.
    ///
    /// Returns as soon as the work is submitted. The frame is NOT finished
    /// when this returns; that is what the returned fence is for.
    [[nodiscard]] virtual Result<Submission> render(const RenderTarget& target,
                                                    const Frame& frame,
                                                    std::span<const Fence> wait_for = {}) = 0;

    /// Can a layer go straight to the display, with no compositing at all?
    ///
    /// True when one opaque layer covers the whole output. A fullscreen
    /// video or game then costs nothing: its buffer is scanned out directly,
    /// with no draw, no blend and no intermediate frame. This is the single
    /// biggest win available to a compositor, and it falls out of the
    /// modifier negotiation rather than needing anything new.
    ///
    /// The caller still has to ASK the display whether it can scan that
    /// layout out (heddle); this only answers the compositing question.
    [[nodiscard]] static const Layer* direct_scanout_candidate(
        const Frame& frame, weft::Size<weft::Output> output) noexcept;

    /// Wait for everything submitted so far. For teardown and for tests.
    [[nodiscard]] virtual Status wait_idle() = 0;

protected:
    Renderer() = default;
};

}  // namespace dye
