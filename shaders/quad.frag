#version 450
#extension GL_EXT_nonuniform_qualifier : require
// quad.frag — sample one layer's texture, or fill a solid colour.
//
// The texture comes from a bindless array indexed per instance. The
// alternative — one descriptor set per layer, bound between draws — is what
// forces a compositor into one draw call per window; with this, fifty
// windows stay one call.
//
// `nonuniformEXT` is required and not decoration: the index varies per
// instance within a single draw, which is exactly the case the extension
// exists for. Without it the result is undefined, and in practice wrong on
// some drivers and right on others, which is the worst kind of bug.

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) in vec2  uv;
layout(location = 1) in vec4  colour;
layout(location = 2) in float alpha;
layout(location = 3) flat in uint flags;

layout(location = 0) out vec4 out_colour;

const uint kKindColour  = 8u;     // a solid colour, not a texture
const uint kOpaque      = 16u;    // an X format: ignore the fourth channel
const uint kTextureShift = 8u;    // the texture index lives above the flags

// The texture index is packed into the top bits of the same integer, so one
// attribute carries both. An extra attribute per instance would be 4 bytes
// per layer per frame for something that fits in the spare bits of one we
// already send.
layout(push_constant) uniform Screen {
    vec2 size;
} screen;

void main() {
    if ((flags & kKindColour) != 0u) {
        // A solid colour. Premultiplied here so the blend below is the same
        // arithmetic for both paths — a renderer with two different blend
        // formulae has two places to get premultiplication wrong.
        out_colour = vec4(colour.rgb * colour.a, colour.a) * alpha;
        return;
    }

    uint index = flags >> kTextureShift;
    vec4 texel = texture(textures[nonuniformEXT(index)], uv);

    // An X format's fourth channel is NOT alpha: a client may leave anything
    // there. Compositing it as alpha renders the window full of holes, and
    // it usually looks fine because most allocators leave 0xff — which is
    // what makes this worth a flag rather than a guess.
    if ((flags & kOpaque) != 0u) texel.a = 1.0;

    // The client's buffer is already premultiplied (the protocol says so for
    // argb8888), so only the whole-surface opacity is applied here.
    out_colour = texel * alpha;
}
