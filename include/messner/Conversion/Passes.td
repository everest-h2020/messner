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
    let dependentDialects = ["ekl::EKLDialect", "arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// EKLToFunc
//===----------------------------------------------------------------------===//

def ConvertEKLToFunc : Pass<"ekl-to-func"> {
    let summary = "Converts EKL programs to functions";
    let description = [{
        TODO.
    }];

    let constructor = "messner::createConvertEKLToFuncPass()";
    let dependentDialects = [
        "ekl::EKLDialect",
        "func::FuncDialect",
        "bufferization::BufferizationDialect"];
}


//===----------------------------------------------------------------------===//
// EKLToIndex
//===----------------------------------------------------------------------===//

def ConvertEKLToIndex : Pass<"ekl-to-index"> {
    let summary = "Converts compound EKL expressions to index ops";
    let description = [{
        TODO.
    }];

    let constructor = "messner::createConvertEKLToIndexPass()";
    let dependentDialects = ["ekl::EKLDialect", "index::IndexDialect"];
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
    let dependentDialects = [
        "ekl::EKLDialect",
        "tensor::TensorDialect",
        "linalg::LinalgDialect",
        "bufferization::BufferizationDialect"];
}

#endif
