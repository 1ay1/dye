#pragma once
// The one assertion macro, so every test file reports the same way.
#include <cstdio>

extern int dye_test_failures;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++dye_test_failures;                                          \
        }                                                                 \
    } while (0)
