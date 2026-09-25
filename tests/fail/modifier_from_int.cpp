// MUST NOT COMPILE: a modifier is not a number you happen to have.
//
// The value that makes this dangerous is 0 — a real, meaningful modifier
// (linear). An uninitialised or forgotten integer therefore reads as "the
// CPU can map this", which is how tiled memory gets memcpy'd.
#include "common.hpp"
void f(std::uint64_t raw) {
    Modifier m = raw;   // no implicit conversion: say Modifier{raw}
    (void)m;
}
