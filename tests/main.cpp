// tests/main.cpp — the shared runner.
//
// Each test file contributes a `run_*` function; this calls them and reports
// once. Splitting them keeps a file about one subject, which is what makes a
// failure readable.
#include "check.hpp"

#include <cstdio>

int dye_test_failures = 0;

void run_format_tests();
void run_buffer_tests();

int main() {
    std::printf("dye\n\n");
    run_format_tests();
    run_buffer_tests();
    std::printf("\n%s\n", dye_test_failures ? "FAILED" : "all passed");
    return dye_test_failures ? 1 : 0;
}
