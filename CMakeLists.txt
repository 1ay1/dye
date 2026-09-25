cmake_minimum_required(VERSION 3.20)
project(dye VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)   # jaal's error/handle types need std::expected
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(DYE_BUILD_TESTS "Build the test suite" ON)
option(DYE_BUILD_EXAMPLES "Build the examples" ON)
option(DYE_SANITIZE "Enable sanitizers (asan|off)" "off")

# ---------------------------------------------------------------------------
# jaal — the runtime this stack sits on, and where the vocabulary types come
# from: jaal::error, jaal::result, owned_handle. Taken from the sibling
# checkout when there is one, so the stack builds from a bare clone.
# ---------------------------------------------------------------------------
if(NOT TARGET jaal::jaal)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../jaal/CMakeLists.txt")
    set(JAAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(JAAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(JAAL_BUILD_BENCH OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../jaal" "${CMAKE_BINARY_DIR}/jaal"
                     EXCLUDE_FROM_ALL)
  else()
    find_package(jaal REQUIRED)
  endif()
endif()

# weft — the geometry. dye's renderer speaks Rect<Output> and Region<Output>
# rather than loose ints, which is what keeps a damage rectangle in the wrong
# space from compiling.
if(NOT TARGET weft::weft)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../weft/CMakeLists.txt")
    set(WEFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WEFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../weft" "${CMAKE_BINARY_DIR}/weft"
                     EXCLUDE_FROM_ALL)
  else()
    find_package(weft REQUIRED)
  endif()
endif()

# ---------------------------------------------------------------------------
# The GPU backend is OPTIONAL.
#
# dye's format, buffer and CPU-renderer layers are pure: they compile and are
# fully tested on a machine with no GPU, no EGL and no display, which is what
# CI is. EGL/GL/GBM buy the GPU backend on top of that, and nothing below
# depends on them.
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(DYE_GPU IMPORTED_TARGET egl gbm glesv2)

# The device layer is compiled: one source with the EGL headers, or a stub
# that reports "no GPU backend" as an ordinary error. Which one is a build
# decision, so no call site needs an #ifdef.
if(DYE_GPU_FOUND)
  set(DYE_DEVICE_SOURCE src/device.cpp)
  message(STATUS "dye: GPU backend enabled (EGL + GBM + GLES2)")
else()
  set(DYE_DEVICE_SOURCE src/device_none.cpp)
  message(STATUS "dye: no EGL/GBM/GLES2 — building without a GPU backend")
endif()

add_library(dye STATIC ${DYE_DEVICE_SOURCE})
add_library(dye::dye ALIAS dye)
target_include_directories(dye PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(dye PUBLIC cxx_std_23)
target_link_libraries(dye PUBLIC jaal::jaal weft::weft)
if(DYE_GPU_FOUND)
  target_link_libraries(dye PRIVATE PkgConfig::DYE_GPU)
endif()
target_compile_options(dye PRIVATE -Wall -Wextra)
set_target_properties(dye PROPERTIES EXPORT_NAME dye)

if(DYE_SANITIZE STREQUAL "asan")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
  message(STATUS "dye: AddressSanitizer + UBSan enabled")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(DIRECTORY include/dye DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS dye EXPORT dyeTargets)
install(EXPORT dyeTargets NAMESPACE dye:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dye)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dyeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dyeConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dye)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/dyeConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dyeConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/dyeConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dye)

# ---------------------------------------------------------------------------
if(NOT DYE_BUILD_TESTS)
  return()
endif()

enable_testing()

add_executable(dye_tests tests/test_format.cpp tests/test_buffer.cpp
                         tests/test_cpu.cpp tests/test_device.cpp tests/main.cpp)
target_link_libraries(dye_tests PRIVATE dye)
target_compile_options(dye_tests PRIVATE -Wall -Wextra)
add_test(NAME format COMMAND dye_tests)

# The confusions that must not compile. A library selling type safety has to
# prove the errors actually happen.
add_test(NAME compile_fail
         COMMAND ${CMAKE_COMMAND}
                 -DCXX=${CMAKE_CXX_COMPILER}
                 -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                 -DJAAL_INCLUDE=$<TARGET_PROPERTY:jaal::jaal,INTERFACE_INCLUDE_DIRECTORIES>
                 -DWEFT_INCLUDE=$<TARGET_PROPERTY:weft::weft,INTERFACE_INCLUDE_DIRECTORIES>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compile_fail.cmake)
