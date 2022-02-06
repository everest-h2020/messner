#ifndef CFDLANG_PASSES
#define CFDLANG_PASSES

include "mlir/Pass/PassBase.td"

def ContractionFactorization
        : Pass<"cfdlang-factorize-contractions", "::mlir::ModuleOp"> {
    let summary = "Simplifies contractions by factorizing their operands";
    let constructor = "mlir::cfdlang::createContractionFactorizationPass()";
}

#endif
