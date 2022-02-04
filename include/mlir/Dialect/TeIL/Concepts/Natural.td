//===- Natural.td - TeIL Natural number concept ------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL natural number concept.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_CONCEPTS_NATURAL
#define TEIL_CONCEPTS_NATURAL

include "mlir/Concepts/Concepts.td"

def TeIL_Nat
        : NativeTypeConcept<"::mlir::teil::NatType", "natural number">;

def TeIL_NatType
        : ConstrainedTypeAttr<TeIL_Nat>;

def TeIL_NatAttr
        : NativeAttrConcept<"::mlir::teil::NatAttr", "natural number"> {
    let valueType = TeIL_Nat;
    let constBuilderCall = [{
        ::mlir::teil::NatAttr::get($_builder.getContext(), $0)
    }];
}

def TeIL_NatArrayAttr
        : NativeAttrConcept<"::mlir::teil::NatArrayAttr", "natural numbers">;

#endif
