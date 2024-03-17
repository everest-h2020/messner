#[========================================================================[.rst:
FindMPFR
-------

Finds and provides the GNU multiprecision floating-point library.

Supports the following signature::

    find_package(MPFR
        [version] [EXACT]       # Minimum or EXACT version e.g. 6.0.2
        [REQUIRED]              # Fail with error if MPFR is not found
    )

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``MPFR::MPFR``
    The MPFR library.

Result Variables
^^^^^^^^^^^^^^^^

``MPFR_FOUND``
    True if the system has the MPFR library.
``MPFR_VERSION``
    The version of the MPFR library which was found.
``MPFR_INCLUDE_DIRS``
    Include directories needed to use MPFR.
``MPFR_LINK_LIBRARIES``
    Transitive dependencies of MPFR.
``MPFR_LIBRARIES``
    Libraries needed to link to MPFR.

Cache variables
^^^^^^^^^^^^^^^

``MPFR_LIBRARY``
    Path to the MPFR library.
``MPFR_INCLUDE_DIR``
    Path to the MPFR include directory.

Hints
^^^^^

``MPFR_ROOT``
    Path to a MPFR installation or build.
``MPFR_USE_STATIC_LIBS``
    If set to ``ON``, only static library files will be accepted, otherwise
    shared libraries are preferred. (Defaults to ``OFF``.)

#]========================================================================]

### Step 0: GMP is always required. ###

find_package(GMP REQUIRED)

### Step 1: Find the library and include path. ###

# Allows the user to specify a custom search path.
set(MPFR_ROOT "" CACHE PATH "Path to a MPFR installation or build.")

if(MPFR_USE_STATIC_LIBS)
    # Static-only find hack.
    set(_mpfr_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

find_library(MPFR_LIBRARY
    NAMES
        mpfr
    HINTS
        "${GMP_ROOT}/lib/"
    DOC "Path to the MPFR library."
)
find_path(MPFR_INCLUDE_DIR
    NAMES
        mpfr.h
    HINTS
        "${GMP_ROOT}/include/"
    DOC "Path to the MPFR include directory."
)

if(MPFR_USE_STATIC_LIBS)
    # Undo our static-only find hack.
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_mpfr_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

mark_as_advanced(
    MPFR_ROOT
    MPFR_LIBRARY
    MPFR_INCLUDE_DIR
)

### Step 2: Examine what we found. ###

include(LibUtils)

if(EXISTS "${MPFR_LIBRARY}")
    # A local install or build was found, and must be examined for the version.
    evp_detect_library_version("${MPFR_LIBRARY}" MPFR_VERSION)
endif()

# Run the standard handler to process all variables.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR
    REQUIRED_VARS
        MPFR_INCLUDE_DIR
        MPFR_LIBRARY
    VERSION_VAR
        MPFR_VERSION
)
if(NOT MPFR_FOUND)
    # Optional dependency not fulfilled.
    return()
endif()

### Step 3: Declare targets and macros. ###

# Set legacy result variables.
set(MPFR_INCLUDE_DIRS "${MPFR_INCLUDE_DIR};${GMP_INCLUDE_DIR}")
set(MPFR_LINK_LIBRARIES "${GMP_LIBRARY}")
set(MPFR_LIBRARIES "${MPFR_LIBRARY};${MPFR_LINK_LIBRARIES}")

# Create imported target if it does not exist.
if(NOT TARGET MPFR::MPFR)
    add_library(MPFR::MPFR UNKNOWN IMPORTED)

    set_target_properties(MPFR::MPFR PROPERTIES
        VERSION                         "${MPFR_VERSION}"
        IMPORTED_LOCATION               "${MPFR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES   "${MPFR_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES        "${MPFR_LINK_LIBRARIES}"
    )
endif()
