#[========================================================================[.rst:
FindREflex
----------

Finds and provides the RE/flex lexer generator.

Supports the following signature::

    find_package(REflex
        [version] [EXACT]       # Minimum or EXACT version e.g. 3.0.1
        [REQUIRED]              # Fail with error if RE/flex is not found
    )

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``REflex::reflex``
    The RE/flex executable.
``REflex::libreflex``
    The RE/flex library.

Result Variables
^^^^^^^^^^^^^^^^

``REflex_FOUND``
    True if the system has the RE/flex library and executable.
``REflex_VERSION``
    The version of the RE/flex executable which was found.
``REflex_INCLUDE_DIRS``
    Include directories needed to use RE/flex runtime library.
``REflex_LIBRARIES``
    Libraries needed to link to the RE/flex runtime.

Cache variables
^^^^^^^^^^^^^^^

``REflex_EXECUTABLE``
    Path to the RE/flex executable.
``REflex_LIBRARY``
    Path to the RE/flex library.
``REflex_INCLUDE_DIR``
    Path to the RE/flex include directory.

Hints
^^^^^

``REflex_ROOT``
    Path to a REflex installation or build.
``REflex_USE_STATIC_LIBS``
    If set to ``ON``, only static library files will be accepted, otherwise
    shared libraries are preferred. (Defaults to ``OFF``.)

#]========================================================================]

### Step 1: Find the executable, library, and include path. ###

# Allows the user to specify a custom search path.
set(REflex_ROOT "" CACHE PATH "Path to a RE/flex installation or build.")

if(REflex_USE_STATIC_LIBS)
    # Static-only find hack.
    set(_reflex_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

find_program(REflex_EXECUTABLE
    NAMES
        reflex
    HINTS
        "${REflex_ROOT}/bin/"
    PATHS
        "/usr/local/bin/"
    DOC "Path to the reflex executable."
)
find_path(REflex_INCLUDE_DIR
    NAMES
        reflex/flexlexer.h
    HINTS
        "${REflex_ROOT}/include/"
    PATHS
        "/usr/local/include/"
    PATH_SUFFIXES
        reflex
    DOC "Path to the reflex include directory."
)
find_library(REflex_LIBRARY
    NAMES
        reflex
    HINTS
        "${REflex_ROOT}/lib/"
    PATHS
        "/usr/local/lib/"
    DOC "Path to the reflex library."
)

mark_as_advanced(
    REflex_ROOT
    REflex_EXECUTABLE
    REflex_INCLUDE_DIR
    REflex_LIBRARY
)

if(REflex_USE_STATIC_LIBS)
    # Undo our static-only find hack.
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_reflex_CMAKE_FIND_LIBRARY_SUFFIXES}")
endif()

if (EXISTS "${REflex_EXECUTABLE}")
    # Try to fetch the version from the found executable.
    execute_process(
        COMMAND
            "${REflex_EXECUTABLE}" --version
        RESULT_VARIABLE _reflex_VERSION_RESULT
        OUTPUT_VARIABLE _reflex_VERSION
        ERROR_QUIET
    )

    string(REGEX MATCH "^reflex ([0-9\.]+)" _MATCH "${_reflex_VERSION}")
    if ((NOT _MATCH) OR _reflex_VERSION_RESULT)
        message(SEND_ERROR "The detected reflex executable is invalid!")
    else()
        set(REflex_VERSION ${CMAKE_MATCH_1})
    endif()
endif()

### Step 2: Examine what we found. ###

# Run the standard handler to process all variables.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(REflex
    REQUIRED_VARS
        REflex_EXECUTABLE
        REflex_INCLUDE_DIR
        REflex_LIBRARY
    VERSION_VAR
        REflex_VERSION
)
if(NOT REflex_FOUND)
    # Optional dependency not fulfilled.
    return()
endif()

### Step 3: Declare targets and macros. ###

# Set legacy result variables.
set(REflex_INCLUDE_DIRS "${REflex_INCLUDE_DIR}")
set(REflex_LIBRARIES "${REflex_LIBRARY}")

if(NOT TARGET REflex::reflex)
    add_executable(REflex::reflex IMPORTED)

    set_target_properties(REflex::reflex PROPERTIES
        VERSION                         "${REflex_VERSION}"
        IMPORTED_LOCATION               "${REflex_EXECUTABLE}"
    )
endif()

if(NOT TARGET REflex::libreflex)
    add_library(REflex::libreflex UNKNOWN IMPORTED)

    set_target_properties(REflex::libreflex PROPERTIES
        VERSION                         "${REflex_VERSION}"
        IMPORTED_LOCATION               "${REflex_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES   "${REflex_INCLUDE_DIR}"
    )
endif()

#[========================================================================[.rst:
.. command:: reflex_target

Implements a ``flex_target`` using RE/flex, copying the signature of that
command exactly. All variables ``FLEX_`` are named ``REflex``, however.

#]========================================================================]
macro(reflex_target _name _input _output)
    cmake_parse_arguments(REflex_TARGET_ARG
        ""
        "COMPILE_FLAGS;DEFINES_FILE"
        ""
        ${ARGN}
    )

    set(REflex_TARGET_outputs "${_output}")
    set(REflex_EXECUTABLE_opts "")

    if(NOT "${REflex_TARGET_ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(SEND_ERROR
            "REflex_TARGET(<Name> <Input> <Output> [COMPILE_FLAGS <string>] [DEFINES_FILE <string>])"
        )
    else()
        if(NOT "${REflex_TARGET_ARG_COMPILE_FLAGS}" STREQUAL "")
            set(REflex_EXECUTABLE_opts "${REflex_TARGET_ARG_COMPILE_FLAGS}")
            separate_arguments(REflex_EXECUTABLE_opts)
        endif()
        if(NOT "${REflex_TARGET_ARG_DEFINES_FILE}" STREQUAL "")
            list(APPEND REflex_TARGET_outputs "${REflex_TARGET_ARG_DEFINES_FILE}")
            list(APPEND REflex_EXECUTABLE_opts --header-file=${REflex_TARGET_ARG_DEFINES_FILE})
        endif()

        add_custom_command(
            OUTPUT              ${REflex_TARGET_outputs}
            COMMAND             REflex::reflex ${REflex_EXECUTABLE_opts} -o${_output} ${_input}
            VERBATIM
            DEPENDS             ${_input}
            COMMENT             "[REFLEX][${_name}] Building scanner with RE/flex ${REflex_VERSION}"
            WORKING_DIRECTORY   ${CMAKE_CURRENT_SOURCE_DIR}
        )

        set(REflex_${_name}_DEFINED TRUE)
        set(REflex_${_name}_OUTPUTS ${_output})
        set(REflex_${_name}_INPUT ${_input})
        set(REflex_${_name}_COMPILE_FLAGS ${REflex_EXECUTABLE_opts})
        if("${REflex_TARGET_ARG_DEFINES_FILE}" STREQUAL "")
            set(REflex_${_name}_OUTPUT_HEADER "")
        else()
            set(REflex_${_name}_OUTPUT_HEADER ${REflex_TARGET_ARG_DEFINES_FILE})
        endif()
    endif()
endmacro()

#[========================================================================[.rst:
.. command:: reflex_target

Implements a ``add_flex_bison_dependency`` for a RE/flex target, copying the
signature of that command exactly.

#]========================================================================]
macro(add_reflex_bison_dependency _reflex _bison)
    if(NOT REflex_${_reflex}_OUTPUTS)
        message(SEND_ERROR "REflex target `${_reflex}' does not exist.")
    endif()
    if(NOT BISON_${_bison}_OUTPUT_HEADER)
        message(SEND_ERROR "Bison target `${_bison}' does not exist.")
    endif()

    set_source_files_properties(${REflex_${_reflex}_OUTPUTS} PROPERTIES
        OBJECT_DEPENDS ${BISON_${_bison}_OUTPUT_HEADER}
    )
endmacro()
