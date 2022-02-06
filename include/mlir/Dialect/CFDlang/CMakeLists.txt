add_subdirectory(IR)

# Passes.td ####################################################################

set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name CFDlang)

add_public_tablegen_target(CFDlangPassesIncGen)

################################################################################

# Documentation.td #############################################################

add_mlir_doc(Passes CFDlangPasses ./ -gen-pass-doc)

################################################################################
