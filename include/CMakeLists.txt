# Configure the project version file.
configure_file(
    ${CMAKE_SOURCE_DIR}/include/cmake.inc.in
    ${CMAKE_BINARY_DIR}/include/cmake.inc
    @ONLY
)

# Configure the Git revision include file shared by all subprojects.
include(GitRevision)
configure_file(
    ${CMAKE_SOURCE_DIR}/include/git.inc.in
    ${CMAKE_BINARY_DIR}/include/git.inc
    @ONLY
)
