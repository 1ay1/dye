# dye — design

A GPU library for compositors. Vulkan, typed, multi-GPU from the first line.

This document is the plan and the reasoning. Everything factual in it was
measured on the hardware named at the bottom, not taken from documentation —
where the two disagreed, the measurement won, and that disagreement is why
this library exists in the shape it does.

---

## 1. Why Vulkan, and not EGL/GLES

Not a preference. A measurement.

The compositor stack's central operation is: take a client's `dmabuf`, turn
it into something samplable, draw it. The client and the compositor must
agree on a **modifier** — the driver's tiling/compression layout — and the
only layout every client can produce without knowing anything about the GPU
is `DRM_FORMAT_MOD_LINEAR`.

On this machine (NVIDIA RTX 4060, proprietary driver), asked for ARGB8888:

| | EGL / GLES | Vulkan |
|---|---|---|
| layouts advertised | 48 | 7 |
| of which LINEAR | **0** | **1** |
| importing a linear dmabuf | **refused** (`EGL_BAD_PARAMETER`) | **works** |

The same buffer, allocated by GBM, from the same process. EGL refuses it
both with an explicit `MOD_LINEAR` attribute and with no modifier attribute
at all. Vulkan creates the image, imports the fd, and binds the memory.

That single result decides the API. A compositor on EGL/NVIDIA cannot accept
the one buffer layout a software-rendering client can produce.

Everything else is corroborating:

- **Explicit sync is native.** `VK_KHR_external_semaphore_fd` exports and
  imports both `SYNC_FD` and `OPAQUE_FD`, for binary *and* timeline
  semaphores — measured, all six combinations. That is exactly what
  `linux-drm-syncobj-v1` needs. On EGL it is an extension pile that NVIDIA
  has historically handled badly, and implicit sync is the cause of most
  compositor stutter.
- **Modifiers are first-class.** One query returns per-modifier plane counts
  *and* feature flags, so "can I sample this layout" is answered before a
  buffer exists. EGL's two `eglQueryDmaBuf*` calls, as above, are not
  merely coarser — they are wrong.
- **Errors are values.** Every call returns a `VkResult`. GL sets a global
  flag you must remember to poll; the EGL backend this replaces called
  `glGetError()` five times and each one was a thing that could be forgotten.
- **Explicit everything** suits a typed library. GL's global bind-state
  fights the design; Vulkan's explicit handles and lifetimes are the design.

The cost is real and worth naming: more code for the first triangle, shaders
compiled at build time with `glslc`, and llvmpipe as the software fallback
instead of Mesa's GL. All three are one-time costs paid in this library so
that no compositor above it pays them.

---

## 2. Device discovery lives in heddle, not here

Compositors break on multi-GPU machines, and the reasons are specific and
avoidable. But fixing them is **heddle's** job, not dye's: heddle is the DRM
library, it already owns device nodes, and rendering does not need to know
what a CRTC is.

This is settled and shipped (heddle `fa384a1`). `heddle/discover.hpp` gives:

```cpp
struct DrmDevice {
    std::optional<RenderNode>  render;    // /dev/dri/renderD*  draw
    std::optional<PrimaryNode> primary;   // /dev/dri/card*     display
    std::string pci_address, driver;
};
Result<std::vector<DrmDevice>> enumerate_devices();
Result<DeviceNumber> device_number_of(borrowed_handle fd);
```

with the two node kinds as **different types**, so opening the privileged
node when you only needed to render does not compile.

The bug that forced it: `heddle::Device::open()` defaulted to
`/dev/dri/card0`. This machine has `card1` and no `card0`, so heddle's own
hardware tests found nothing, skipped every check, and still printed "all
passed". Card numbering was never stable. The default is gone, and the
pairing of a GPU's two nodes now goes through sysfs rather than arithmetic
on the numbers.

### What dye needs from it

Only this: a **device number**, `major:minor`. Two integers.

```cpp
Result<std::unique_ptr<Device>> Device::open(DeviceNumber drm_node);
```

dye matches that against every Vulkan physical device's
`VK_EXT_physical_device_drm`, which reports exactly the same pair. Measured
here:

```
[0] NVIDIA GeForce RTX 4060 (discrete)
    drm: primary=1 226:1   render=1 226:128
[1] llvmpipe (CPU)
    drm: primary=0 0:0     render=0 0:0
```

`226:128` is `/dev/dri/renderD128`; `226:1` is `/dev/dri/card1`. That match is
the authoritative link between "this Vulkan device" and "this kernel device",
and it is inherently Vulkan's business, so it is the one piece of device
handling that belongs in dye. Note also that llvmpipe reports no DRM nodes at
all — which is how a software device is identified, without matching on its
name.

So: **heddle finds the device, dye matches it to a GPU.** Neither depends on
the other; `major:minor` is the whole shared vocabulary.

---

## 3. Multi-GPU, designed in

Two devices, as a first-class concept rather than a later patch:

```cpp
struct Setup {
    Device& render;    // where client buffers are imported and composited
    Device& scanout;   // where the finished frame is displayed
};                     // usually, but not always, the same device
```

The negotiation, in one place:

1. Ask both devices for their modifiers for a format.
2. Intersect. Advertise the intersection to clients, so a client's buffer is
   importable by **both** by construction.
3. If the intersection is empty, copy — and say so, once, in a log line. A
   compositor that silently copies every frame is slow for reasons nobody can
   find; one that silently does not renders black.

This is ~100 lines when it is in the design, and a rewrite when it is not.

---

## 4. Explicit sync

Measured available here, every combination:

```
binary   + SYNC_FD      export=yes  import=yes
binary   + OPAQUE_FD    export=yes  import=yes
timeline + OPAQUE_FD    export=yes  import=yes
dmabuf memory                       import=yes
```

So dye's fences are real, and they are file descriptors:

```cpp
class Fence {                       // a point in time on the GPU
    owned_handle fd;                // exportable, importable, waitable
};
```

A client hands over an acquire fence; the compositor waits on the GPU, not
the CPU. The compositor hands back a release fence. This is what
`linux-drm-syncobj-v1` is, and it is the difference between a compositor that
stutters under load and one that does not.

Implicit sync — where the driver guesses from buffer access — is the default
on EGL and is where most compositor jank comes from.

---

## 5. Speed

"Fast" is a set of specific decisions, not an adjective.

**One draw call per frame, not per window.** Every layer's quad goes into one
instanced draw: position, source rect, alpha and transform as per-instance
attributes. 50 windows is one call.

**Damage is scissors, not geometry.** `vkCmdSetScissor` per damage rectangle,
one render pass. The GPU rasterises only what changed; a still screen costs
one empty submit.

**No read-back in the hot path.** The EGL backend imported and then read
pixels into memory for a CPU blit — correct, and the reason tapestry drops
frames. Here the imported image *is* the texture.

**Direct scanout when a layer covers the output.** A fullscreen client's
buffer goes to the display untouched: zero compositing, zero copies. This is
the single biggest win for games and video, and it falls out of having the
modifier negotiation above.

**Three queues, used properly.** This GPU offers a graphics family (16), a
transfer-only family (2) and a compute family (8). Uploads go on the transfer
queue so they do not stall the renderer.

**Nothing allocated per frame.** Command buffers, descriptor sets and staging
memory are pooled at startup. A frame that allocates is a frame that can
stutter.

**The pipeline is built once.** No shader compilation, no pipeline creation
after the first frame, ever.

---

## 6. What stays exactly as it is

The **CPU renderer** (`dye/cpu.hpp`) does not change, and becomes more
valuable. It is:

- the fallback with no GPU — a VM, CI, a driver that refused to load;
- the **oracle**. The same `Frame` goes through Vulkan and through the CPU
  renderer, and the outputs are compared pixel for pixel. Without that, a GPU
  renderer is tested by looking at it, which finds the obvious bugs and none
  of the ones that matter: a half-pixel sampling offset, an edge row blended
  twice, alpha applied in the wrong space.

`Format`, `Layout`, `BufferDescription` and `validate()` do not change
either. Buffer validation stays pure and stays **before** any driver call,
because a driver's response to a malformed buffer ranges from a clean error
to a GPU hang, and no compositor can ship "it depends on the vendor" as its
error handling.

---

## 7. Layering

```
  tapestry        the compositor: policy, protocol, windows
  ────────────────────────────────────────────────────────
  dye             Vulkan: import, composite, fences         ← this library
  heddle          KMS: connectors, modes, planes, atomic
  shed            logind/seatd: opening devices
  loom            the Wayland protocol
  weft            geometry: typed coordinate spaces
  jaal            the runtime: loop, handles, errors
```

dye depends on jaal (handles, errors) and weft (geometry) and nothing else.
It does **not** depend on heddle: rendering does not need to know what a
CRTC is, and keeping it that way is what lets the nested backend exist. The
compositor above passes heddle's `DeviceNumber` to dye's `Device::open`;
two integers, and neither library includes the other's headers.

---

## 8. Order of work

1. **`dye/device.hpp` on Vulkan** — instance, physical-device selection by
   DRM device number, queues, dmabuf import and allocation with modifiers.
   Replaces the EGL backend; the interface above it does not move.
   (Discovery is done: heddle `fa384a1`.)
2. **`dye/fence.hpp`** — semaphores as fds, both directions.
3. **`dye/vulkan_renderer`** — one pass, instanced quads, damage as scissors.
4. **`tests/test_parity.cpp`** — the same `Frame` through both renderers,
   compared pixel for pixel. The test that makes the rest trustworthy.
5. **Direct scanout and multi-GPU copy** — once 1–4 are green.

---

## Measurements

Everything factual above was measured on:

```
NVIDIA GeForce RTX 4060 (AD107), proprietary driver
/dev/dri/card1 + /dev/dri/renderD128   (note: no card0)
Vulkan 1.4.357 instance, 1.4.351 device
Mesa llvmpipe 22.1.8 as the second physical device
```

The probes live in `tests/`, so these numbers can be re-taken on other
hardware rather than trusted.
