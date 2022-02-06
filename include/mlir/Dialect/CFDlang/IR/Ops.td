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
include "mlir/Dialect/TeIL/Concepts/Natural.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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
                /*AffineScope,*/
                IsolatedFromAbove,
                NoRegionArguments,
                Symbol,
                // DeclareOpInterfaceMethods<
                //     Symbol,
                //     [/*
                //         "isOptionalSymbol",
                //         "canDiscardOnUseEmpty",
                //         "getVisibility",
                //         "setVisibility"
                //     */]
                // >,
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

    let builders = [
        OpBuilder<(ins CArg<"StringAttr", "nullptr">:$name), [{
            $_state.addRegion()->emplaceBlock();
            if (name) {
                $_state.addAttribute(
                    mlir::SymbolTable::getSymbolAttrName(),
                    name
                );
            }
        }]>,
        OpBuilder<(ins CArg<"StringRef", "">:$name), [{
            build(
                $_builder,
                $_state,
                name.empty()
                    ? nullptr
                    : $_builder.getStringAttr(name)
            );
        }]>
    ];
    let skipDefaultBuilders = 1;

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
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<Symbol, overrides>,
                DeclareOpInterfaceMethods<CFDlang_DeclarationOp, overrides>,
                HasParent<"cfdlang::ProgramOp">
            ]
        > {

    let arguments = (ins SymbolNameAttr:$sym_name, CFDlang_AtomType:$type);

    let builders = [
        OpBuilder<(ins "Twine":$name, "AtomType":$atomType), [{
            build(
                $_builder,
                $_state,
                $_builder.getStringAttr(name),
                AtomTypeAttr::get(atomType)
            );
        }]>,
        OpBuilder<(ins "Twine":$name, CArg<"shape_t", "teil::scalar_shape">:$shape), [{
            build(
                $_builder,
                $_state,
                name,
                AtomType::get($_builder.getContext(), shape)
            );
        }]>
    ];

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
        // BUG: The verifier seems to ignore getVisibility() and thus we need
        //      to return false here or the thing fails.
        return false;
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

    let builders = [
        OpBuilder<(ins "StringAttr":$name, "AtomType":$atomType), [{
            $_state.addRegion()->emplaceBlock();
            $_state.addAttribute(
                mlir::SymbolTable::getSymbolAttrName(),
                name
            );
            $_state.addAttribute("type", TypeAttr::get(atomType));
        }]>,
        OpBuilder<(ins "const Twine&":$name, "AtomType":$atomType), [{
            build($_builder, $_state, $_builder.getStringAttr(name), atomType);
        }]>
    ];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $sym_name `:` custom<AtomTypeAttr>($type) attr-dict-with-keyword $body
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

//===----------------------------------------------------------------------===//
// EvalOp
//===----------------------------------------------------------------------===//

def CFDlang_EvalOp
        : CFDlang_Op<
            "eval",
            [
                NoSideEffect,
                ConstantLike,
                DeclareOpInterfaceMethods<
                    SymbolUserOpInterface,
                    ["verifySymbolUses"]
                >,
                DeclareOpInterfaceMethods<
                    CFDlang_AtomOp,
                    [/*"inferAtomType", "reifyAtomSize"*/]
                >
            ]
        > {
    let summary = "Atom evaluation";
    let description = [{
        The `cfdlang.eval` operation requests the evaluation of an atom symbol
        with the specified name. It materializes an SSA value for a symbol.

        Example:
        ```mlir
        %S = cfdlang.eval @S : [11 11]
        ```
    }];

    // TODO: Operands for defines with arguments?
    let arguments = (ins SymbolRefAttr:$name);
    let results = (outs CFDlang_Atom:$result);

    let hasFolder = 1;

    let assemblyFormat = [{
        $name `:` custom<AtomType>(type($result)) attr-dict
    }];

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // AtomOp
    //===------------------------------------------------------------------===//

    static LogicalResult inferAtomShape(
        MLIRContext*,
        Optional<Location>,
        ValueRange operands,
        DictionaryAttr,
        RegionRange,
        teil::ShapeBuilder &atomShape
    )
    {
        // TODO: Implement.
        return failure();
    }

    FailureOr<teil::AtomSize> reifyAtomSize(OpBuilder &builder)
    {
        // TODO: Implement.
        return failure();
    }
    }];
}

//===----------------------------------------------------------------------===//
// AtomOp
//===----------------------------------------------------------------------===//

// Template for atom ops.
class CFDlang_AtomOpBase<
    string mnemonic,
    list<OpTrait> traits = [],
    list<string> overrides = []
>
        : CFDlang_Op<
            mnemonic,
            traits # [
                NoSideEffect,
                DeclareOpInterfaceMethods<CFDlang_AtomOp, overrides>,
                // The following native trait implements all the interfaces below.
                NativeOpTrait<"CFDlang_AtomInference">,
                InferTypeOpInterface,
                // NOTE: See Interfaces.h for more information on why we exclude
                //       this interface.
                //InferShapedTypeOpInterface,
                ReifyRankedShapedTypeOpInterface
            ]
        > {

    let results = (outs CFDlang_Atom:$result);

    /*let assemblyFormat = [{
        operands : custom<AtomType>(type(operands)) attr-dict
    }];*/
    let printer = [{ ::printAtomOp(p, *this); }];
    let parser = [{ return ::parseAtomOp<$cppClass>(parser, result); }];
}

//===----------------------------------------------------------------------===//
// ArithmeticOp
//===----------------------------------------------------------------------===//

// Template for arithmetic ops.
class CFDlang_ArithmeticBase<
    string mnemonic,
    list<OpTrait> traits = [],
    list<string> overrides = []
>
        : CFDlang_AtomOpBase<
            mnemonic,
            traits,
            overrides # [/*
                "inferAtomShape",
                "reifyAtomSize"
            */]
        > {

    let arguments = (ins CFDlang_Atom:$lhs, CFDlang_Atom:$rhs);
    let results = (outs CFDlang_Atom:$result);

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // AtomOp
    //===------------------------------------------------------------------===//

    static LogicalResult inferAtomShape(
        MLIRContext*,
        Optional<Location>,
        ValueRange operands,
        DictionaryAttr,
        RegionRange,
        teil::ShapeBuilder &atomShape
    )
    {
        return interface_defaults::inferAtomShape(operands, atomShape);
    }

    FailureOr<teil::AtomSize> reifyAtomSize(OpBuilder &builder)
    {
        return interface_defaults::reifyAtomSize(builder, *this);
    }
    }];
}

def CFDlang_AddOp : CFDlang_ArithmeticBase<"add", [Commutative]>;
def CFDlang_SubOp : CFDlang_ArithmeticBase<"sub">;
def CFDlang_MulOp : CFDlang_ArithmeticBase<"mul", [Commutative]>;
def CFDlang_DivOp : CFDlang_ArithmeticBase<"div">;

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

def CFDlang_ProductOp
        : CFDlang_AtomOpBase<
            "prod",
            [],
            ["inferAtomShape", "reifyAtomSize"]
        > {
    let arguments = (ins CFDlang_Atom:$lhs, CFDlang_Atom:$rhs);
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

def CFDlang_ContractOp
        : CFDlang_AtomOpBase<
            "cont",
            [],
            ["inferAtomShape", "reifyAtomSize"]
        > {
    let arguments = (ins CFDlang_Atom:$operand, TeIL_NatArrayAttr:$indices);

    let builders = [
        OpBuilder<(ins "Atom":$operand, "nat_indices_t":$indices), [{
            build(
                $_builder,
                $_state,
                operand,
                teil::NatArrayAttr::get($_builder.getContext(), indices)
            );
        }]>
    ];

    /*let assemblyFormat = [{
        $operand : custom<AtomType>(type($operand)) indices custom<NatPairs>($indices) attr-dict
    }];*/
    let printer = [{ ::print(p, *this); }];
    let parser = [{ return ::parseContractOp(parser, result); }];
}

#endif
