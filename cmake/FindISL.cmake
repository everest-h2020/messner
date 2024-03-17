#[========================================================================[.rst:
FindISL
-------

Finds and provides the integer set library.

Supports the following signature::

    find_package(ISL
        [version] [EXACT]       # Minimum or EXACT version e.g. 0.23
        [REQUIRED]              # Fail with error if ISL is not found
    )

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``ISL::ISL``
    The ISL library.

Result Variables
^^^^^^^^^^^^^^^^

``ISL_FOUND``
    True if the system has the ISL library.
``ISL_VERSION``
    The version of the ISL library which was found.
``ISL_INCLUDE_DIRS``
    Include directories needed to use ISL.
``ISL_LINK_LIBRARIES``
    Transitive dependencies of ISL.
``ISL_LIBRARIES``
    Libraries needed to link to ISL.

Cache variables
^^^^^^^^^^^^^^^

``ISL_LIBRARY``
    Path to the ISL library.
``ISL_INCLUDE_DIR``
    Path to the ISL include directory.

Hints
^^^^^

``ISL_ROOT``
    Path to an ISL installation or build.
``ISL_USE_STATIC_LIBS``
    If set to ``ON``, only static library files will be accepted, otherwise
    shared libraries are preferred. (Defaults to ``OFF``.)

#]========================================================================]

### Step 0: Detect a system-managed installation. ###

find_package(PkgConfig REQUIRED)
pkg_search_module(PC_ISL QUIET isl)

### Step 1: Find the library and include path. ###

# Allows the user to specify a custom search path.
set(ISL_ROOT "" CACHE PATH "Path to an ISL installation or build.")

if(ISL_USE_STATIC_LIBS)
    # Static-only find hack.
    set(_isl_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

find_library(ISL_LIBRARY
    NAMES
        isl
    HINTS
        # In case of builddir:
        "${ISL_ROOT}/.libs/"
        # In case of install dir:
        "${ISL_ROOT}/lib/"
        "${PC_ISL_LIBDIR}"
    DOC "Path to the isl library."
)
find_path(ISL_INCLUDE_DIR
    NAMES
        isl/union_map.h
    HINTS
        "${ISL_ROOT}/include/"
        "${PC_ISL_INCLUDEDIR}"
    DOC "Path to the isl include directory."
)

if(ISL_USE_STATIC_LIBS)
    # Undo our static-only find hack.
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_isl_CMAKE_FIND_LIBRARY_SUFFIXES}")
endif()

mark_as_advanced(
    ISL_ROOT
    ISL_LIBRARY
    ISL_INCLUDE_DIR
)

### Step 2: Examine what we found. ###

include(LibUtils)

if(EXISTS "${ISL_LIBRARY}")
    # Assume no transitive dependencies.
    set(_isl_DEPENDENCY_INCLUDE_DIRS "")
    set(_isl_DEPENDENCY_LIBRARIES "")

    # Detect known transitive dependencies by undefined symbols.
    evp_get_library_undefined_symbols("${ISL_LIBRARY}" _isl_UNDEFINED_SYMBOLS)
    if(_isl_UNDEFINED_SYMBOLS MATCHES "__gmp")
        # ISL was compiled against GMP, which must be linked (but not included).
        message(VERBOSE "Detected GMP dependency for ISL.")
        find_package(GMP REQUIRED)
        list(APPEND _isl_DEPENDENCY_LIBRARIES ${GMP_LIBRARIES})
    endif()
    # Are there any more?

    if("${ISL_LIBRARY}" MATCHES "^${PC_ISL_LIBDIR}")
        # The found library is the system-managed one.
        set(ISL_VERSION "${PC_ISL_VERSION}")
    else()
        # A local install or build was found, and must be examined for the version.
        # ISL has a bonkers library versioning scheme that cannot be trusted.
        # The only sure-fire solution is to compile and run a snippet.
        file(WRITE
            ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/islver.c
            "#include <assert.h>
             #include <isl/version.h>
             int main() {
                printf(\"%s\", isl_version());
                return 0;
             }"
        )
        try_run(
            _islver_RETURNED
            _islver_COMPILED
            ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/islver.c
            COMPILE_DEFINITIONS     "-I${ISL_INCLUDE_DIR}"
            LINK_LIBRARIES          "${ISL_LIBRARY};${_isl_DEPENDENCY_LIBRARIES}"
            RUN_OUTPUT_VARIABLE     _islver_OUTPUT
        )
        set(ISL_VERSION "NOTFOUND")
        if(_islver_COMPILED)
            string(REGEX MATCH "([0-9\\.]+)" ISL_VERSION "${_islver_OUTPUT}")
        endif()
    endif()
endif()

# Run the standard handler to process all variables.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ISL
    REQUIRED_VARS
        ISL_INCLUDE_DIR
        ISL_LIBRARY
    VERSION_VAR
        ISL_VERSION
)
if(NOT ISL_FOUND)
    # Optional dependency not fulfilled.
    return()
endif()

### Step 3: Declare targets and macros. ###

# Set legacy result variables.
set(ISL_INCLUDE_DIRS "${ISL_INCLUDE_DIR};${_isl_DEPENDENCY_INCLUDE_DIRS}")
set(ISL_LINK_LIBRARIES "${_isl_DEPENDENCY_LIBRARIES}")
set(ISL_LIBRARIES "${ISL_LIBRARY};${ISL_LINK_LIBRARIES}")

# Create imported target if it does not exist.
if(NOT TARGET ISL::ISL)
    add_library(ISL::ISL UNKNOWN IMPORTED)

    set_target_properties(ISL::ISL PROPERTIES
        VERSION                         "${ISL_VERSION}"
        IMPORTED_LOCATION               "${ISL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES   "${ISL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES        "${ISL_LINK_LIBRARIES}"
    )
endif()
