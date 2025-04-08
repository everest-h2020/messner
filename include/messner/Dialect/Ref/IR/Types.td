//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef REF_TYPES
#define REF_TYPES

include "messner/Dialect/Ref/IR/Dialect.td"
include "messner/Dialect/Ref/Interfaces/ReferenceTypeInterface.td"

//===----------------------------------------------------------------------===//
// Type families
//===----------------------------------------------------------------------===//

def Ref_ReferenceType : Ref_Type<"Reference", [Ref_ReferenceTypeInterface]> {
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
        /// Obtains a copy of this reference type with @p cellType .
        ///
        /// @param              cellType Replacement cell type.
        ///
        /// @pre    `cellType`
        auto cloneWith(Type cellType) const -> ReferenceType;
        /// Obtains a copy of this reference type with @p kind .
        ///
        /// @param              kind    Replacement ReferenceKind.
        auto cloneWith(ReferenceKind kind) const -> ReferenceType;
    }];
}

#endif // REF_TYPES
