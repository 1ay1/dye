# compile_fail.cmake — every case below MUST fail to compile.
#
# dye's claim is that a fourcc, a modifier, a stride and an offset stop being
# interchangeable integers. That claim is worth nothing unless the confusions
# actually fail, so each case is a real one and the suite fails if any of
# them compiles.
#
# It also fails on the WRONG failure: a typo or a missing header would make
# any file "fail" and turn this into a test that proves nothing.

set(CASES
  modifier_from_int
  modifier_as_int
  format_from_int
  swap_format_and_modifier
  copy_plane
  copy_description)

set(failed 0)
set(passed "")

foreach(case ${CASES})
  set(src "${SRC_DIR}/tests/fail/${case}.cpp")
  if(NOT EXISTS "${src}")
    message(SEND_ERROR "missing case file: ${src}")
    set(failed 1)
    continue()
  endif()

  execute_process(
    COMMAND ${CXX} -std=c++23 -fsyntax-only
            -I${SRC_DIR}/include -I${JAAL_INCLUDE} -I${WEFT_INCLUDE} "${src}"
    RESULT_VARIABLE rc OUTPUT_QUIET ERROR_VARIABLE errout)

  string(REGEX MATCH "stray|No such file|expected [^ ]+ before" bogus "${errout}")
  if(bogus)
    message(SEND_ERROR "  ${case}: rejected for the WRONG reason (${bogus})")
    set(failed 1)
  elseif(rc EQUAL 0)
    message(SEND_ERROR "  ${case}: COMPILED but should not have")
    set(failed 1)
  else()
    message(STATUS "  ${case}: rejected")
    list(APPEND passed ${case})
  endif()
endforeach()

if(failed)
  message(FATAL_ERROR "compile-fail suite: a confusion was accepted")
endif()
list(LENGTH passed n)
message(STATUS "compile-fail: ${n} confusions rejected")
