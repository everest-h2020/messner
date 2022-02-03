# TODO: Add license.

################################################################################
# evp-tools
#
# The evp-tools project.
################################################################################

# NOTE: Subproject structure considered harmful.
#
# Technically, we want every self-contained target to live in its own CMake
# project (${PROJECT_NAME} idiom). However, this works very badly with the
# legacy style that LLVM/MLIR are using CMake.
#
# As a result, we are using the big monoproject, global target properties
# approach.

cmake_minimum_required(VERSION 3.13)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(evp-tools
    VERSION     0.1.1
    LANGUAGES   C CXX
)

# Set global language standard to C++17.
set(CMAKE_CXX_STANDARD 17)

# Set shared output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

# Fix some policy problems with LLVM/MLIR.
cmake_policy(SET CMP0116 OLD)

# Find MLIR.
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Load LLVM and MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Apply target properties for LLVM/MLIR globally.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Apply target properties for this project globally.
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_BINARY_DIR}/include/)

# Generated includes.
add_subdirectory(include)
# Libraries.
add_subdirectory(lib)
# Executables.
add_subdirectory(tools)
# Regression tests.
add_subdirectory(test)
