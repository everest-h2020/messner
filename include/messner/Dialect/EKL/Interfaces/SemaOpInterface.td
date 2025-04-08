//===- SemaOpInterface.td ----------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL SemaOpInterface interface.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_SEMAOPINTERFACE
#define EKL_SEMAOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Typing/TypeCheckOpInterface.td"

def EKL_SemaOpInterface : OpInterface<"SemaOpInterface", [TypeCheckOpInterface]> {
    let cppNamespace = "::mlir::ekl";

    let description = [{
        Interface for an EKL AST node-like operation that participates in
        semantic analysis.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Check the semantics of this operation and collect all
                applicable diagnostics.

                If failure is returned, semantic analysis of the program must be
                terminated. Otherwise, the behavior is determined by the caller.
            }],
            /*retTy=*/"::mlir::LogicalResult",
            /*methodName=*/"checkSemantics",
            /*args=*/(ins "::llvm::SmallVectorImpl<::mlir::Diagnostic> &":$diagnostics),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return ::mlir::success(); }]>];

    // NOTE: Verify the op and its descendants first, then verify this interface.
    let verifyWithRegions = 1;
    let verify = [{ return ::mlir::ekl::verifySemaOpInterface($_op); }];
}

#endif // EKL_SEMAOPINTERFACE
