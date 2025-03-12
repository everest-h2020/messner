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
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRPluginsLib
        MLIRSupport
        MLIROptLib
        LLVMOrcJIT
        LLVMX86Info
        LLVMX86Desc
        LLVMX86CodeGen
        LLVMX86TargetMCA
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        ${translation_libs}
)
