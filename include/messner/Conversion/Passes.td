//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EVP conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef MESSNER_CONVERSION_PASSES
#define MESSNER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// EKLToArith
//===----------------------------------------------------------------------===//

def ConvertEKLToArith : Pass<"ekl-to-arith"> {
    let summary = "Converts EKL arithmetic, logic and relations to arith ops";
    let description = [{
        TODO.
    }];

    let constructor = "messner::createConvertEKLToArithPass()";
}

//===----------------------------------------------------------------------===//
// EKLToLinalg
//===----------------------------------------------------------------------===//

def ConvertEKLToLinalg : Pass<"ekl-to-linalg"> {
    let summary = "Converts compound EKL expressions to linalg ops";
    let description = [{
        TODO.
    }];

    let constructor = "messner::createConvertEKLToLinalgPass()";
}

#endif
