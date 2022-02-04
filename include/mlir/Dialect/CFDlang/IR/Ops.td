//===- Ops.td - CFDlang op declarations --------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the CFDlang dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CFDLANG_OPS
#define CFDLANG_OPS

include "mlir/Dialect/CFDlang/IR/Interfaces.td"
include "mlir/Dialect/CFDlang/IR/Types.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

def CFDlang_ProgramOp
        : CFDlang_Op<
            "program",
            [
                NoSideEffect,
                /*AffineScope,*/
                IsolatedFromAbove,
                NoRegionArguments,
                DeclareOpInterfaceMethods<
                    Symbol,
                    [/*
                        "isOptionalSymbol",
                        "canDiscardOnUseEmpty",
                        "getVisibility",
                        "setVisibility"
                    */]
                >,
                SymbolTable
            ] # GraphRegionNoTerminator.traits
        > {
    let summary = "Top level container for a CFDlang program";
    let description = [{
        The `cfdlang.program` operation declares a new CFDlang program in this
        translation unit, optionally associating it with a symbol name.

        Example:

        ```mlir
        cfdlang.prorgam @my_prog {
            ...
        }
        ```
    }];

    let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";

    code extraClassDeclaration = [{
    Optional<StringRef> getName() { return sym_name(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface
    //===------------------------------------------------------------------===//

    bool isOptionalSymbol()
    {
        // Also valid without a symbol name.
        return true;
    }
    bool canDiscardOnUseEmpty()
    {
        // Never discard unless pruned.
        return false;
    }
    mlir::SymbolTable::Visibility getVisibility()
    {
        // Always has nested visibility.
        return mlir::SymbolTable::Visibility::Nested;
    }
    void setVisibility(mlir::SymbolTable::Visibility) {}
    }];
}

#endif
