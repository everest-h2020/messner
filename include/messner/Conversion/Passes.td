//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EVP conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef MESSNER_CONVERSION_PASSES
#define MESSNER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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
        "arith::ArithDialect",
        "bufferization::BufferizationDialect",
        "linalg::LinalgDialect",
        "tensor::TensorDialect"];
}

//===----------------------------------------------------------------------===//
// EKLToStandard
//===----------------------------------------------------------------------===//

def ConvertEKLToStandard : Pass<"ekl-to-std"> {
    let summary = "Converts simple EKL expressions to standard ops";
    let description = [{
        TODO.
    }];

    let constructor = "messner::createConvertEKLToStandardPass()";
    let dependentDialects = [
        "ekl::EKLDialect",
        "arith::ArithDialect",
        "index::IndexDialect",
        "math::MathDialect"];
}

#endif
