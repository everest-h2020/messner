//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL passes.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_PASSES
#define EKL_PASSES

include "mlir/Pass/PassBase.td"

def TypeCheck : Pass<"ekl-type-check", "::mlir::ekl::KernelOp"> {
    let summary = "Type checks an EKL kernel";
    let description = [{
        The `ekl-type-check` pass visits `ekl.kernel` operations and attempts
        to restrict the type bounds on all expressions as far as possible.

        If the input program is ill-formed, this pass will fail, indicating
        where the type checking process got stuck and why.
    }];

    let constructor = "::mlir::ekl::createTypeCheckPass()";
}

def Lower : Pass<"ekl-lower", "::mlir::ekl::KernelOp"> {
    let summary = "Lowers EKL operations to simpler ones";
    let description = [{
        The `ekl-lower` pass visits `ekl.program` operations and rewrites as
        many complex EKL operations to simpler ones as possible. In essence,
        this is a canonicalization pass that is destructive w.r.t. the EKL
        program AST.
    }];

    let constructor = "::mlir::ekl::createLowerPass()";
}

#endif
