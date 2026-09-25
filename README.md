# dye

**A GPU library for compositors.** Vulkan, typed, multi-GPU from the first
line. A dye is what gives a weave its colour, which is what this does for a
[tapestry](https://github.com/1ay1/tapestry).

Sixth in the set: [loom](https://github.com/1ay1/loom) speaks the Wayland
protocol, [weft](https://github.com/1ay1/weft) decides what is on screen,
[heddle](https://github.com/1ay1/heddle) talks to the display hardware, dye
puts the pixels there.

## Why Vulkan and not EGL

Not a preference. A measurement.

A compositor's central job is: take a client's `dmabuf`, turn it into
something samplable, draw it. Client and compositor have to agree on a
**modifier** — the driver's tiling layout — and the one layout every client
can produce without knowing anything about the GPU is `LINEAR`.

On an NVIDIA RTX 4060 with the proprietary driver, asked for ARGB8888:

| | EGL / GLES | Vulkan |
|---|---|---|
| layouts advertised | 48 | 7 |
| of which LINEAR | **0** | **1** |
| importing a linear dmabuf | **refused** (`EGL_BAD_PARAMETER`) | **works** |

Same buffer, same process, same driver. EGL refuses it with an explicit
`MOD_LINEAR` attribute *and* with no modifier attribute at all. Vulkan
creates the image, imports the fd and binds the memory.

A compositor on EGL + NVIDIA cannot accept the one buffer layout a
software-rendering client can produce. That decided the API; the rest —
native fd fences, per-modifier feature flags, errors as return values — is
corroborating. `DESIGN.md` has the detail and the probes.

## The bug it exists to prevent

A DRM fourcc is a `uint32_t`. So is a modifier's low half, a stride, an
offset, a width, a plane count and a GL enum. Every API in this space takes
a row of them:

```c
eglCreateImage(dpy, ctx, target, buf, attrs);   // fourcc, stride, offset,
                                                // modifier, width, height...
```

Swap two and it compiles. It then produces a black window, or a torn one, on
a machine you do not own.

```cpp
void import(Format fmt, Modifier mod);
import(mod, fmt);              // COMPILE ERROR
Modifier m = some_uint64;      // COMPILE ERROR: say Modifier{x}
Format f = fourcc_from_client;  // COMPILE ERROR: parse it, it may be junk
```

A `Format` knows its own alpha and byte order. A `Modifier` is separate from
it, because the same pixels in a different memory layout are not the same
buffer. The trap: **modifier 0 is real** (`LINEAR`), so a forgotten integer
claims tiled memory is CPU-readable, and that is how NVIDIA output gets
garbled.

## What it does

```cpp
auto gpu = dye::Device::open(drm_node);        // heddle found the node
auto image = gpu->import(client_buffer);       // a dmabuf, now a texture
auto renderer = dye::Renderer::create(*gpu);

dye::Frame frame;
frame.layers.push_back({image->as_texture(), dst_rect});
frame.damage.add(changed);

auto sub = renderer->render(target, frame);    // one draw call
kms.flip(target, std::move(sub->done));        // the display waits, not you
```

Five things, and only five: which layouts the GPU can import, importing a
client's buffer, allocating one it can definitely import, reading pixels back
(for capture and tests), and knowing when the GPU has finished.

It does **not** find devices — that is heddle's job — and it does not
present. Putting a frame on a screen is KMS or a parent compositor; dye
renders and says when it is done.

## Fast on purpose

Compositing a 1080p window onto a 3440x1440 screen:

| | per frame |
|---|---|
| CPU renderer | 22.52 ms |
| **GPU renderer** | **0.15 ms** |

150x, and the decisions behind it are each measurable:

- **One draw call per frame, not per window.** Every layer is an instance and
  the quad's corners come from `gl_VertexIndex`, so there is no vertex
  buffer. Fifty windows is one call with fifty instances.
- **Bindless textures.** The texture index is an instance attribute. A
  descriptor set bound per layer is exactly what forces a draw call per
  layer.
- **Damage is a scissor, not geometry.** One render pass, one scissor per
  damaged rectangle; the rasteriser never generates a fragment outside it.
  `LOAD` rather than `CLEAR`, because damage rendering needs everything
  outside the damage to survive.
- **Nothing allocated per frame.** Command buffers, descriptor sets and the
  instance buffer are pooled at startup.
- **The pipeline is built once.** No shader compilation after the first
  frame, ever — the SPIR-V is compiled at build time and embedded.
- **Direct scanout.** `direct_scanout_candidate()` spots the fullscreen
  opaque case, where the right amount of compositing is none.

### The expensive mistake

Reading a 1080p frame back took **637 ms**, and the pixels were perfectly
correct.

Host-visible memory on a discrete GPU is write-combined by default: fine to
write, catastrophic to *read*, because every access crosses PCIe uncached.
Asking for `HOST_CACHED` as well took the same operation to **5.6 ms**.

Both versions produce identical output, so no correctness test could ever
have found it. `tests/bench_readback.cpp` exists for that reason and prints a
frame budget.

## Tested against an oracle

A GPU renderer is otherwise tested by looking at it, which finds the obvious
bugs and none of the ones that matter: a half-pixel sampling offset, an edge
row blended twice, alpha applied in the wrong space.

So `tests/test_parity.cpp` runs the same `dye::Frame` through the GPU
renderer and through the CPU one, and compares the output pixel for pixel.
The CPU renderer is the oracle: simple enough to read, with its arithmetic
checked against hand-computed values. Three of five cases match exactly; the
other two differ only by linear filtering.

Writing that test found two real bugs: the GPU renderer never drew the
background, and texture ids were view pointers truncated to 32 bits — so two
images could collide and the renderer would draw the wrong one.

The format, buffer and CPU-renderer layers are pure: they compile and are
fully tested on a machine with no GPU, no Vulkan and no display, which is
what CI is.

```
cmake -S . -B build && cmake --build build -j12
ctest --test-dir build -j12     # tests + 6 compile-fail cases
./build/dye_bench               # what a frame costs
```

## Requires

- C++23, Vulkan 1.2, `glslc`
- [jaal](https://github.com/1ay1/jaal) — handles and errors
- [weft](https://github.com/1ay1/weft) — typed coordinate spaces

Without Vulkan it still builds: `Device::open()` returns an ordinary error
and the CPU renderer does the work.
