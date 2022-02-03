/* --- PROJECT VERSION IMPLEMENTATION --------------------------------------- */

#include "cmake.inc"
#include "git.inc"

const std::string_view  APIVersion::string      = CMAKE_PROJECT_VERSION;
const unsigned          APIVersion::major       = CMAKE_PROJECT_VERSION_MAJOR;
const unsigned          APIVersion::minor       = CMAKE_PROJECT_VERSION_MINOR;
const unsigned          APIVersion::patch       = CMAKE_PROJECT_VERSION_PATCH;

const std::string_view  GitRevision::string     = GIT_DESCRIBE;
const std::string_view  GitRevision::head       = GIT_HEAD;
const bool              GitRevision::is_dirty   = GIT_DIRTY;

/* -------------------------------------------------------------------------- */
