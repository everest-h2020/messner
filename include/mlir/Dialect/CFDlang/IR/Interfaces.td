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

#endif
