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

def DecayNumber : Pass<"ekl-decay-number", "::mlir::ekl::KernelOp"> {
    let summary = "Decays arithmetic operations to the highest ABI precision";
    let description = [{
        The `ekl-decay-number` pass visits `ekl.kernel` operations and finds
        arithmetic operations that combine `!ekl.number`-typed operands with
        fixed-precision operands. It then decays the `!ekl.number` operands to
        the most precise of the other types.
    }];

    let constructor = "::mlir::ekl::createDecayNumberPass()";
}

def Homogenize : Pass<"ekl-homogenize", "::mlir::ekl::KernelOp"> {
    let summary = "Rewrites EKL expression to remove implicit behavior";
    let description = [{
        The `ekl-homogenize` pass visits `ekl.kernel` operations and eliminates
        all the implicit casts.
    }];

    let constructor = "::mlir::ekl::createHomogenizePass()";
}

def Lower : Pass<"ekl-lower", "::mlir::ekl::KernelOp"> {
    let summary = "TODO";
    let description = [{
        TODO
    }];

    let constructor = "::mlir::ekl::createLowerPass()";
}

#endif
