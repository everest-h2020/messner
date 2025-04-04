################################################################################
# messner-opt
#
# The messner optimizer driver.
################################################################################

project(messner-opt)

add_executable(${PROJECT_NAME}
    messner-opt.cpp)

# Link all standard MLIR dialect, extension and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIROptLib
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs})
