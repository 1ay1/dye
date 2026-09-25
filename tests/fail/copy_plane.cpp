// MUST NOT COMPILE: a plane owns a descriptor, so copying one would close
// the same descriptor twice — and the second close lands on whatever the
// process opened in between.
#include "common.hpp"
void f(const Plane& p) {
    Plane copy = p;
    (void)copy;
}
