// MUST NOT COMPILE: nor does a modifier decay back into one.
//
// eglCreateImage takes the modifier as two separate 32-bit halves, next to
// a fourcc, a stride and an offset. A modifier that converts to an integer
// can be passed to any of those slots.
#include "common.hpp"
void f(Modifier m) {
    std::uint64_t raw = m;   // say m.raw()
    (void)raw;
}
