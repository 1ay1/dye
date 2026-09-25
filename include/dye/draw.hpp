#pragma once
// dye/draw.hpp — what to draw, as data.
//
// A renderer's input is usually a pile of loose numbers: a source rectangle,
// a destination rectangle, a transform, an alpha, a clip. They are all ints
// and floats, they arrive in an order chosen by whoever wrote the function,
// and two of them (source and destination) have the same type and opposite
// meanings. Swapping those is the bug that renders a window scaled to a few
// pixels in the corner, and it compiles.
//
// So the draw list is data, in weft's typed spaces:
//
//   * where it goes is a Rect<Output>       -- the screen
//   * where it comes from is a Rect<Buffer> -- the client's pixels
//   * what changed is a Region<Output>      -- the damage
//
// Those three do not convert to each other, so the swap does not compile.
// They are also exactly the types weft's scene graph already produces, so a
// compositor is not translating between two geometries.
//
// Nothing here knows about GL, EGL or the CPU. This is the description both
// backends consume, which is what lets one be checked against the other
// pixel for pixel (tests/test_parity.cpp).

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <weft/region.hpp>
#include <weft/space.hpp>

#include "buffer.hpp"
#include "format.hpp"

namespace dye {

// ---------------------------------------------------------------------------
// Transform — how a client's buffer is oriented relative to the screen.
//
// wl_output.transform's eight cases. A rotation and a flip, not a free 2x2
// matrix: the protocol only ever asks for these, and a matrix would let a
// renderer accept something it cannot implement.
// ---------------------------------------------------------------------------
enum class Transform : std::uint8_t {
    normal = 0,
    rotate_90 = 1,
    rotate_180 = 2,
    rotate_270 = 3,
    flipped = 4,
    flipped_90 = 5,
    flipped_180 = 6,
    flipped_270 = 7,
};

/// Does this transform exchange width and height?
///
/// The check every size calculation needs and half of them forget: a 90
/// degree rotation means a 1920x1080 buffer covers a 1080x1920 area. Getting
/// it wrong gives a window with its edges cut off, on rotated monitors only.
[[nodiscard]] constexpr bool swaps_axes(Transform t) noexcept {
    switch (t) {
        case Transform::rotate_90:
        case Transform::rotate_270:
        case Transform::flipped_90:
        case Transform::flipped_270:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] constexpr bool is_flipped(Transform t) noexcept {
    return static_cast<std::uint8_t>(t) >= 4;
}

// ---------------------------------------------------------------------------
// Source — where a layer's pixels come from.
//
// Either a texture the GPU holds (an imported client buffer) or a flat
// colour. A colour is not a degenerate 1x1 texture: single-pixel buffers and
// solid backgrounds are common enough that making them a case saves an
// allocation, an import and a sample per frame.
// ---------------------------------------------------------------------------

/// An opaque handle to something the device holds. Zero is nothing.
///
/// A distinct type rather than a GLuint so a texture name cannot be passed
/// where a framebuffer name goes, which is the same class of bug as heddle's
/// CRTC-vs-framebuffer swap and has the same fix.
class TextureId {
public:
    constexpr TextureId() = default;
    constexpr explicit TextureId(std::uint32_t raw) noexcept : raw_(raw) {}

    [[nodiscard]] constexpr std::uint32_t raw() const noexcept { return raw_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return raw_ != 0; }
    constexpr explicit operator bool() const noexcept { return valid(); }

    friend constexpr bool operator==(TextureId, TextureId) = default;

private:
    std::uint32_t raw_ = 0;
};

/// Straight (NOT premultiplied) 8-bit colour.
///
/// Premultiplication is the other silent renderer bug: blending
/// premultiplied colour with the straight formula double-darkens the edges
/// of every translucent window, which reads as a shadow artefact rather than
/// as a maths error. Saying which one this is, in the name, is the fix.
struct Colour {
    std::uint8_t r = 0, g = 0, b = 0, a = 255;

    friend constexpr bool operator==(Colour, Colour) = default;
};

/// Pixels the device holds, plus how to read them.
struct Texture {
    TextureId id{};
    /// The texture's own size, so a source rectangle can be checked against
    /// it rather than trusted.
    weft::Size<weft::Buffer> size{};
    /// What the bytes mean. Carried because an X format must be composited
    /// as opaque however its fourth byte happens to be filled.
    Format format{};
    /// The client drew bottom-up (GL's origin). Sampled differently rather
    /// than copied straight.
    bool y_invert = false;
};

using Source = std::variant<Texture, Colour>;

// ---------------------------------------------------------------------------
// Layer — one thing to draw.
// ---------------------------------------------------------------------------

/// One quad: where it comes from, where it goes, and how it is combined.
struct Layer {
    Source source{};

    /// The part of the source to read, in the BUFFER's pixels. Unset means
    /// all of it. Set by wp_viewporter's crop.
    std::optional<weft::Rect<weft::Buffer>> src{};

    /// Where it lands, in the OUTPUT's pixels. A different type from `src`,
    /// so the two cannot be swapped.
    weft::Rect<weft::Output> dst{};

    /// The buffer's orientation relative to the output.
    Transform transform = Transform::normal;

    /// Whole-surface opacity, 0..1 (wp_alpha_modifier). Multiplied with the
    /// source's own alpha.
    float alpha = 1.0f;

    /// The part of `dst` that is fully opaque, in output pixels. Not a
    /// rendering instruction but an optimisation fact: a renderer may skip
    /// everything underneath it. Wrong values here cause windows to
    /// disappear behind each other, so it is the compositor's job to be
    /// honest and the renderer's to treat it as a hint it may ignore.
    weft::Region<weft::Output> opaque{};

    /// Is this layer completely see-through, and so worth skipping?
    [[nodiscard]] bool invisible() const noexcept {
        if (alpha <= 0.0f || dst.empty()) return true;
        if (const auto* c = std::get_if<Colour>(&source)) return c->a == 0;
        return false;
    }
};

// ---------------------------------------------------------------------------
// Frame — everything to draw, once.
// ---------------------------------------------------------------------------

/// A complete description of one frame.
///
/// Back to front: layers[0] is furthest away. That is the order weft's scene
/// walk produces and the order a painter's-algorithm renderer needs, and
/// writing it down here stops each backend deciding for itself.
struct Frame {
    std::vector<Layer> layers{};

    /// What changed since the last frame, in output pixels. A renderer only
    /// has to produce correct pixels INSIDE this region.
    ///
    /// Empty means nothing changed and the frame can be skipped entirely.
    /// That is a real state, not a mistake, so it is represented rather than
    /// guarded against: a compositor with a still screen should do no work.
    weft::Region<weft::Output> damage{};

    /// What to put where nothing is drawn. A compositor without a wallpaper
    /// still has to write something, or the previous frame shows through.
    Colour background{0, 0, 0, 255};

    [[nodiscard]] bool nothing_to_do() const noexcept { return damage.empty(); }
};

/// The surface being drawn into.
struct Target {
    weft::Size<weft::Output> size{};
    /// The format of the destination. A renderer needs it to know whether
    /// to write alpha at all.
    Format format{};
};

}  // namespace dye
