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

include "mlir/Dialect/CFDlang/Concepts/Atom.td"
include "mlir/Dialect/CFDlang/IR/Interfaces.td"
include "mlir/Dialect/CFDlang/IR/Types.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

//===----------------------------------------------------------------------===//
// DeclarationOp
//===----------------------------------------------------------------------===//

// Template for declarations.
class CFDlang_DeclarationBase<
    string mnemonic,
    list<OpTrait> traits = [],
    list<string> overrides = []
>
        : CFDlang_Op<
            mnemonic,
            traits # [
                NoSideEffect,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<Symbol, overrides>,
                DeclareOpInterfaceMethods<CFDlang_DeclarationOp, overrides>,
                HasParent<"cfdlang::ProgramOp">
            ]
        > {

    let arguments = (ins SymbolNameAttr:$sym_name, CFDlang_AtomType:$type);

    let assemblyFormat = "$sym_name `:` custom<AtomTypeAttr>($type) attr-dict";
}

//===----------------------------------------------------------------------===//
// InputOp
//===----------------------------------------------------------------------===//

def CFDlang_InputOp
        : CFDlang_DeclarationBase<
            "input",
            [],
            [/*
                "isDeclaration",
                "canDiscardOnUseEmpty",
                "getVisibility",
                "setVisibility"
            */]
        > {
    let summary = "Input atom declaration";
    let description = [{
        The `cfdlang.input` operation declares a new atom symbol in the current
        program that will be bound on invocation.

        Example:

        ```mlir
        cfdlang.input @a : [2 2]
        ```
    }];

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface
    //===------------------------------------------------------------------===//

    bool isDeclaration()
    {
        // Always just a declaration.
        return true;
    }
    bool canDiscardOnUseEmpty()
    {
        // Even though a declaration, may never be discarded.
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

//===----------------------------------------------------------------------===//
// DefinitionOp
//===----------------------------------------------------------------------===//

// Template for definitions.
class CFDlang_DefinitionBase<
    string mnemonic,
    list<OpTrait> traits = [],
    list<string> overrides = []
>
        : CFDlang_DeclarationBase<
            mnemonic,
            traits # [
                /*AffineScope,*/
                // TODO: Defines with arguments?
                NoRegionArguments,
                DeclareOpInterfaceMethods<CFDlang_DefinitionOp, overrides>,
                SingleBlockImplicitTerminator<"cfdlang::YieldOp">
            ],
            overrides
        > {

    let arguments = (ins SymbolNameAttr:$sym_name, CFDlang_AtomType:$type);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $sym_name `:` custom<AtomTypeAttr>($type) attr-dict $body
    }];
}

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

def CFDlang_DefineOp
        : CFDlang_DefinitionBase<
            "define",
            [],
            [/*
                "getVisibility",
                "setVisibility"
            */]
        > {
    let summary = "Internal atom definition";
    let description = [{
        The `cfdlang.define` operation declares and defines a new atom that is
        internal to the current program.

        Example:

        ```mlir
        cfdlang.define @my_var : [3 3] {
            ...
            cfdlang.yield %0 : [3 3]
        }
        ```
    }];

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface
    //===------------------------------------------------------------------===//

    mlir::SymbolTable::Visibility getVisibility()
    {
        // Always has nested visibility.
        return mlir::SymbolTable::Visibility::Private;
    }
    void setVisibility(mlir::SymbolTable::Visibility) {}
    }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def CFDlang_OutputOp
        : CFDlang_DefinitionBase<
            "output",
            [],
            [/*
                "getVisibility",
                "setVisibility"
            */]
        > {
    let summary = "Output atom definition";
    let description = [{
        The `cfdlang.output` operation is analogous to the `cfdlang.define`
        operation, except that the declared atom is visible to an invoker.

        Example:

        ```mlir
        cfdlang.output @v : [11 11 11] {
            ...
            cfdlang.yield %0 : [11 11 11]
        }
        ```
    }];

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface
    //===------------------------------------------------------------------===//

    mlir::SymbolTable::Visibility getVisibility()
    {
        return mlir::SymbolTable::Visibility::Nested;
    }
    void setVisibility(mlir::SymbolTable::Visibility) {}
    }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def CFDlang_YieldOp
        : CFDlang_Op<
            "yield",
            [NoSideEffect, ReturnLike, Terminator]
        > {
    let summary = "Block terminator";
    let description = [{
        The `cfdlang.yield` operation terminates a block in CFDlang, such as
        the definition blocks.

        Example:

        ```mlir
        cfdlang.yield %0 : [3 3]
        ```
    }];

    let arguments = (ins CFDlang_Atom:$atom);

    let builders = [OpBuilder<(ins), "">];

    let assemblyFormat = "$atom `:` custom<AtomType>(type($atom)) attr-dict";
}

#endif
