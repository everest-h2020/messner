################################################################################
# messner-lsp-server
#
# The messner MLIR language server.
################################################################################

project(messner-lsp-server)

add_executable(${PROJECT_NAME}
    messner-lsp-server.cpp)

# Link all standard MLIR dialect and extension libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRLspServerLib
        ${dialect_libs})
