################################################################################
# messner
#
# The messner CLI facade.
################################################################################

project(messner)

add_llvm_executable(${PROJECT_NAME}
    messner.cpp
)

# Link all standard MLIR dialect and translation libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRPluginsLib
        MLIRSupport
        MLIROptLib
        ${dialect_libs}
        ${translation_libs}
)
