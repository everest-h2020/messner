#ifndef CFDLANG_PASSES
#define CFDLANG_PASSES

include "mlir/Pass/PassBase.td"

def ContractionFactorization
        : Pass<"cfdlang-factorize-contractions", "::mlir::ModuleOp"> {
    let summary = "Simplifies contractions by factorizing their operands";
    let constructor = "mlir::cfdlang::createContractionFactorizationPass()";
}

def Kernelize
        : Pass<"cfdlang-kernelize", "::mlir::ModuleOp"> {
    let summary = "Turns cfdlang programs into kernel functions";
    let constructor = "mlir::cfdlang::createKernelizePass()";
}

def Codegen
        : Pass<"cfdlang-codegen", "::mlir::ModuleOp"> {
    let summary = "Turns cfdlang kernels into linalg implementations";
    let constructor = "mlir::cfdlang::createCodegenPass()";
}

#endif
