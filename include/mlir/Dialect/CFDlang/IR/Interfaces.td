//===- Interfaces.td - CFDlang interface declarations ------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the CFDlang interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CFDLANG_INTERFACES
#define CFDLANG_INTERFACES

include "mlir/Concepts/Inference.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DeclarationOp
//===----------------------------------------------------------------------===//

def CFDlang_DeclarationOp : OpInterface<"DeclarationOp"> {
    let cppNamespace = "::mlir::cfdlang";

    let description = [{
        Interface for a CFDlang operation that declares an atom.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Returns the symbol name of this declaration.
            }],
            /*retTy=*/"StringRef",
            /*methodName=*/"getName",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<SymbolOpInterface>($_op.getOperation()).getName();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom type of this declaration.
            }],
            /*retTy=*/"AtomType",
            /*methodName=*/"getAtomType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return $_op->template getAttrOfType<TypeAttr>("type")
                    .getValue()
                    .template cast<AtomType>();
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom shape of this declaration.
            }],
            /*retTy=*/"shape_t",
            /*methodName=*/"getAtomShape",
            /*args=*/(ins),
            /*methodBody=*/[{
                return $_op.getAtomType().getShape();
            }],
            /*defaultImplementation=*/""
        >
    ];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<SymbolOpInterface::Trait>(),
            "`DeclarationOp` is only applicable to `SymbolOpInterface` ops."
        );

        return success();
    }];
}

//===----------------------------------------------------------------------===//
// DefinitionOp
//===----------------------------------------------------------------------===//

def CFDlang_DefinitionOp : OpInterface<"DefinitionOp"> {
    let cppNamespace = "::mlir::cfdlang";

    let description = [{
        Interface for a CFDlang operation that defines an atom.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Returns the symbol name of this definition.
            }],
            /*retTy=*/"StringRef",
            /*methodName=*/"getName",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<DeclarationOp>($_op.getOperation()).getName();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom type of this definition.
            }],
            /*retTy=*/"AtomType",
            /*methodName=*/"getAtomType",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<DeclarationOp>($_op.getOperation()).getAtomType();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom shape of this definition.
            }],
            /*retTy=*/"shape_t",
            /*methodName=*/"getAtomShape",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<DeclarationOp>($_op.getOperation()).getAtomShape();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom value of this definition.
            }],
            /*retTy=*/"Atom",
            /*methodName=*/"getAtom",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return $_op->getRegion(0).back().getTerminator()->getOperand(0)
                    .template cast<Atom>();
            }]
        >
    ];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<DeclarationOp::Trait>(),
            "`DefinitionOp` is only applicable to `DeclarationOp` ops."
        );

        return interface_defaults::verifyDefinitionOp($_op);
    }];
}

//===----------------------------------------------------------------------===//
// AtomOp
//===----------------------------------------------------------------------===//

def CFDlang_AtomOp : OpInterface<"AtomOp"> {
    let cppNamespace = "::mlir::cfdlang";

    let description = [{
        Interface for a CFDlang operation that computes an atom.
    }];

    let methods = [
        StaticInterfaceMethod<
            /*desc=*/[{
                Given the build arguments to an operation of this type, infers
                the shape of the result atom.
            }],
            /*retTy=*/"LogicalResult",
            /*methodName=*/"inferAtomShape",
            /*args=*/!con(
                InferenceMethod.arguments,
                (ins "teil::ShapeBuilder&":$atomShape)
            ),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return failure();
            }]
        >,
        StaticInterfaceMethod<
            /*desc=*/[{
                Given the build arguments to an operation of this type, infers
                the type of the result atom, or nullptr on error.
            }],
            /*retTy=*/"AtomType",
            /*methodName=*/"inferAtomType",
            /*args=*/InferenceMethod.arguments,
            /*methodBody=*/InferenceMethodImpl<[{
                return interface_defaults::inferAtomType<ConcreteOp>($_args);
            }]>.body,
            /*defaultImplementation=*/""
        >,

        InterfaceMethod<
            /*desc=*/[{
                Returns the atom type.
            }],
            /*retTy=*/"AtomType",
            /*methodName=*/"getAtomType",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<AtomOp>($_op.getOperation()).getAtom().getType();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom shape.
            }],
            /*retTy=*/"shape_t",
            /*methodName=*/"getAtomShape",
            /*args=*/(ins),
            /*methodBody=*/[{
                return cast<AtomOp>($_op.getOperation()).getAtomType()
                    .getShape();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the atom value.
            }],
            /*retTy=*/"Atom",
            /*methodName=*/"getAtom",
            /*args=*/(ins),
            /*methodBody=*/[{
                return $_op.getResult().template cast<Atom>();
            }],
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{
                Obtains the AtomSize of the result atom, potentially inserting
                new operations using the provided builder.
            }],
            /*retTy=*/"FailureOr<teil::AtomSize>",
            /*methodName=*/"reifyAtomSize",
            /*args=*/(ins "OpBuilder&":$builder),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return failure();
            }]
        >
    ];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::OneResult>(),
            "`AtomOp` is only applicable to `OneResult` ops."
        );

        // Everything else is verified by the inference traits.
        return success();
    }];
}

// Native trait that implements the InferTypeOpInterface,
// InferShapedTypeOpInterface and ReifyRankedShapedTypeOpInterface interfaces
// using the AtomOp interface.
def CFDlang_AtomInference : NativeOpTrait<"CFDlang_AtomInference">;

#endif
