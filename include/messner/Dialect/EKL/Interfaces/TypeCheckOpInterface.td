//===- TypeCheckOpInterface.td -----------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL TypeCheckOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_TYPECHECKOPINTERFACE
#define EKL_TYPECHECKOPINTERFACE

include "mlir/IR/OpBase.td"

def EKL_TypeCheckOpInterface : OpInterface<"TypeCheckOpInterface"> {
    let cppNamespace = "::mlir::ekl";

    let description = [{
        Interface for an operation that participates in late type checking.

        Since EKL programs are subject to late type checking, the type (bounds)
        of expressions may change in the IR. This happens during an inference
        pass that updates the IR.

        Operations that implement this interface opt-in to participating in this
        type inference process. Their operand types are allowed to change, at
        which point they are asked to refine their result type bounds. Through
        this mechanism, they are also required to validate their invariants on
        their operand types. Operations that don't implement this interface
        prevent their uses from having their types changed.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Performs type checking for this operation.

                The caller ensures that the operation is in an ODS-verified
                state. The callee must be able to verify the type bounds on its
                operands from this partially verified state, as part of this
                interface's trait verifier.
            }],
            /*retTy=*/"LogicalResult",
            /*methodName=*/"typeCheck",
            /*args=*/(ins "::mlir::ekl::AbstractTypeChecker &":$typeChecker),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">];

    let verify = [{ return impl::verifyTypeCheckOpInterface($_op); }];
}

#endif
