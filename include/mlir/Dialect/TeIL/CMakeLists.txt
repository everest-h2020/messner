# Enums.td #####################################################################

set(LLVM_TARGET_DEFINITIONS "Enums.td")

mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(TeILEnumsIncGen)

################################################################################

add_subdirectory(IR)
