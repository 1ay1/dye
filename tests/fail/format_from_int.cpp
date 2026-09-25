// MUST NOT COMPILE: a fourcc from a client is not yet a Format.
//
// It has to be validated (Format::from_fourcc returns an optional) so the
// unsupported case becomes a protocol error instead of a format whose
// bytes_per_pixel is 0 and whose rendering is silently wrong.
#include "common.hpp"
void f(std::uint32_t fourcc) {
    Format fmt = fourcc;
    (void)fmt;
}
