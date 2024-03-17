################################################################################
# thirdparty
#
# Bundled third-party and external dependencies.
################################################################################

# RE-flex
#
# We have a perfectly good FindREflex CMake module that we want to use. To make
# it work with the bundled dependency, we set its result variables, which will
# causes later invocations of it to return immediately.

add_subdirectory(RE-flex EXCLUDE_FROM_ALL)

set(REflex_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/bin/reflex CACHE STRING "" FORCE)
set(REflex_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/lib/libreflex_static_lib.a CACHE STRING "" FORCE)
set(REflex_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RE-flex/include CACHE STRING "" FORCE)

add_executable(REflex::reflex ALIAS Reflex)
add_library(REflex::libreflex ALIAS ReflexLibStatic)
