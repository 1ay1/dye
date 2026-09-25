#version 450
// quad.vert — one instanced quad per layer.
//
// There is no vertex buffer. The four corners of a unit quad come from
// gl_VertexIndex, and everything that differs between layers arrives as an
// instance attribute. A frame with fifty windows is ONE draw call with fifty
// instances, not fifty draw calls — which is the difference between a
// compositor that scales to a busy desktop and one that does not.
//
// Why no vertex buffer: the geometry is always a rectangle. Uploading four
// corners per layer per frame would be a buffer write, a barrier and a bind,
// all to communicate something the shader can compute from an integer.

// Per-instance: where this layer goes and where it reads from.
layout(location = 0) in vec4  in_dst;        // x, y, w, h in output pixels
layout(location = 1) in vec4  in_src;        // x, y, w, h in texture UV (0..1)
layout(location = 2) in vec4  in_colour;     // straight RGBA, for solid layers
layout(location = 3) in uint  in_flags;      // transform | kind
layout(location = 4) in float in_alpha;      // whole-surface opacity

layout(push_constant) uniform Screen {
    vec2 size;        // the output, in pixels
} screen;

layout(location = 0) out vec2  uv;
layout(location = 1) out vec4  colour;
layout(location = 2) out float alpha;
layout(location = 3) flat out uint flags;

const uint kTransformMask = 7u;

void main() {
    // The unit quad from the vertex index: 0,1,2,3 -> (0,0) (1,0) (0,1) (1,1).
    // Drawn as a triangle strip, so four vertices and no index buffer.
    vec2 corner = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));

    // Where it lands, in pixels, then into clip space.
    //
    // Vulkan's framebuffer origin is the TOP left and its clip space has Y
    // pointing down, so the two agree and no flip is needed. The flip that
    // catches everyone is GL's bottom-left origin, and this is not GL.
    vec2 pixel = in_dst.xy + corner * in_dst.zw;
    gl_Position = vec4(pixel / screen.size * 2.0 - 1.0, 0.0, 1.0);

    // Undo the client's buffer transform.
    //
    // A client draws rotated so a rotated display can scan its buffer out
    // directly; sampling has to undo that. Done here rather than per
    // fragment, it costs four vertices instead of two million pixels.
    uint t = in_flags & kTransformMask;
    vec2 c = corner;

    // 90 and 270 exchange the axes. A size calculation that forgets this is
    // the "rotated window has its edges cut off" bug.
    if (t == 1u || t == 3u || t == 5u || t == 7u) c = c.yx;

    if (t == 1u || t == 5u) c.y = 1.0 - c.y;   // 90
    if (t == 2u || t == 6u) c = 1.0 - c;       // 180
    if (t == 3u || t == 7u) c.x = 1.0 - c.x;   // 270
    if (t >= 4u) c.x = 1.0 - c.x;              // the flipped variants

    // Into the source rectangle (wp_viewporter's crop), in UV.
    uv = in_src.xy + c * in_src.zw;

    colour = in_colour;
    alpha = in_alpha;
    flags = in_flags;
}
