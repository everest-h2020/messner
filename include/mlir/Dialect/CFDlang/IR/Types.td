//===- Types.td - CFDlang type declarations ----------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the CFDlang dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CFDLANG_TYPES
#define CFDLANG_TYPES

include "mlir/Dialect/CFDlang/IR/Base.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Scalar type
//===----------------------------------------------------------------------===//

// Unspecified scalar type token.
def CFDlang_ScalarType : CFDlang_Type<"Scalar", [MemRefElementTypeInterface]> {
    let mnemonic = "scalar";
    let summary = "Generic unspecified scalar type";

    let description = [{
        The CFDlang language has no concept of number representations, and
        therefore uses this opaque stand-in type.

        Lowering implementations must select a concrete representation and
        replace the placeholder operations performed on values of this type
        with appropriate implementations.
    }];
}

#endif
