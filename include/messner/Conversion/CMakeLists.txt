set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name EVPConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix EVPConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix EVPConversion)

add_public_tablegen_target(EVPConversionPassesIncGen)
