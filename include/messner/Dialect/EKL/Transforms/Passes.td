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

#endif
