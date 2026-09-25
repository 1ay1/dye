// MUST NOT COMPILE: the swap this library exists to prevent.
//
// Every dmabuf API takes the format and the modifier adjacent:
//
//   eglCreateImage(..., EGL_LINUX_DRM_FOURCC_EXT, fourcc,
//                       EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, mod, ...)
//
// In C both are integers and swapping them compiles, then produces a black
// window on one vendor's driver and a torn one on another's.
#include "common.hpp"
void import(Format fmt, Modifier mod);
void f(Format fmt, Modifier mod) {
    import(mod, fmt);   // swapped
}
