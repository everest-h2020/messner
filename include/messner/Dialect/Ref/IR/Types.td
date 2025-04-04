//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef REF_TYPES
#define REF_TYPES

include "messner/Dialect/Ref/IR/Base.td"

//===----------------------------------------------------------------------===//
// Type families
//===----------------------------------------------------------------------===//

def Ref_ReferenceType : Ref_Type<"Reference"> {
    let mnemonic = "type";
    let summary = "Type of a reference to a cell";
    let description = [{
        The `ref.type` type represents a reference to a cell.

        Examples:

        ```
        // These two name the same type:
        !ref.ref<p | r, f64>
        !ref<pr, f64>
        ```
    }];

    let parameters = (ins
        AttrParameter<"Type", "Type stored in the cell">:$cellType,
        AttrParameter<"ReferenceKind", "Reference kind">:$kind);

    let assemblyFormat = [{
        `<` $kind `,` $cellType `>`
    }];

    let extraClassDeclaration = [{
        /// Determines whether the reference is readable.
        [[nodiscard]] bool isReadable() const;
        /// Determines whether the reference is writable.
        [[nodiscard]] bool isWritable() const;

        /// Obtains a copy of this reference type with @p cellType .
        ///
        /// @param              cellType Replacement cell type.
        ///
        /// @pre    `cellType`
        [[nodiscard]] ReferenceType cloneWith(Type cellType) const;
        /// Obtains a copy of this reference type with @p kind .
        ///
        /// @param              kind    Replacement ReferenceKind.
        [[nodiscard]] ReferenceType cloneWith(ReferenceKind kind) const;
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
        Implements a type constraint on reference types requiring them to be
        readable.

        Examples:

        ```
        !ref<r, i32>
        ```
    }];
}

def Ref_WritableRef : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ref::WritableRefType>($_self)">,
        /*summary=*/"Any writable reference type",
        /*cppClassName=*/"::mlir::ref::WritableRefType"> {
    let description = [{
        Implements a type constraint on reference types requiring them to be
        writable.

        Examples:

        ```
        !ref<w, i32>
        ```
    }];
}

#endif // REF_TYPES
