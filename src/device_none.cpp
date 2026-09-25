// device_none.cpp — the build with no EGL.
//
// dye's format, buffer and CPU-renderer layers are pure: they compile and
// are fully tested on a machine with no GPU, no EGL and no display, which is
// what CI is. This file is what keeps that true — Device::open() links and
// returns an ordinary error rather than the library failing to build.
//
// The alternative, #ifdef'ing every call site in the compositor, spreads
// "might there be a GPU" through code that has nothing to do with it.
#include "dye/device.hpp"

namespace dye {

namespace {
auto no_gpu() {
    return std::unexpected(
        Error::make(std::errc::not_supported, "this build of dye has no GPU backend"));
}
}  // namespace

Result<std::unique_ptr<Device>> Device::open() { return no_gpu(); }
Result<std::unique_ptr<Device>> Device::open(const char*) { return no_gpu(); }

bool gpu_available() noexcept { return false; }

}  // namespace dye
