// MUST NOT COMPILE: same reason, one level up. A compositor passes these
// around constantly (protocol handler to renderer to scanout), and a copy
// anywhere on that path is a double close.
#include "common.hpp"
void f(const BufferDescription& d) {
    BufferDescription copy = d;
    (void)copy;
}
