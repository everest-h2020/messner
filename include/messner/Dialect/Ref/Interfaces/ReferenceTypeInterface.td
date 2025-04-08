//===- ReferenceTypeInterface.td ---------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref ReferenceTypeInterface.
//
//===----------------------------------------------------------------------===//

#ifndef REF_REFERENCETYPEINTERFACE
#define REF_REFERENCETYPEINTERFACE

include "mlir/IR/OpBase.td"

def Ref_ReferenceTypeInterface : TypeInterface<"ReferenceTypeInterface"> {
    let cppNamespace = "::mlir::ref";

    let description = [{
        Interface for a type that holds a reference to a cell.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Get the type of the referenced cell.
            }],
            /*retTy=*/"::mlir::Type",
            /*methodName=*/"getCellType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the ReferenceKind of this reference.
            }],
            /*retTy=*/"::mlir::ref::ReferenceKind",
            /*methodName=*/"getKind",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/"">,
        InterfaceMethod<
            /*desc=*/[{
                Gets the Resource associated with this reference.
            }],
            /*retTy=*/"::mlir::SideEffects::Resource *",
            /*methodName=*/"getResource",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return ::mlir::SideEffects::DefaultResource::get();
            }]>];

    code extraSharedClassDeclaration = [{
        /// Determines whether this reference is readable.
        auto isReadable() const -> bool
        {
            return ::mlir::ref::isReadable($_type.getKind());
        }

        /// Determines whether this reference is writable.
        auto isWritable() const -> bool
        {
            return ::mlir::ref::isWritable($_type.getKind());
        }
    }];
}

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

def Ref_ReadableRef : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ref::ReadableRefType>($_self)">,
        /*summary=*/"Any readable reference type",
        /*cppClassName=*/"::mlir::ref::ReadableRefType"> {
    let description = [{
        Implements a type constraint on ReferenceTypeInterface that requires it
        to be readable.
    }];
}

def Ref_WritableRef : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ref::WritableRefType>($_self)">,
        /*summary=*/"Any writable reference type",
        /*cppClassName=*/"::mlir::ref::WritableRefType"> {
    let description = [{
        Implements a type constraint on ReferenceTypeInterface that requires it
        to be writable.
    }];
}

#endif // REF_REFERENCETYPEINTERFACE
