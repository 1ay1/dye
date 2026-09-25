// Shared by every compile-fail case. This header itself must ALWAYS compile:
// each case has to fail for its own semantic reason, not because the setup
// is broken.
#pragma once
#include <dye/buffer.hpp>
#include <dye/format.hpp>
using namespace dye;
