//===- Concepts.td - MLIR concepts support -----------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This file provides definitions for MLIR concepts.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONCEPTS
#define MLIR_CONCEPTS

include "mlir/IR/OpBase.td"

class NativeTypeConcept<string concept, string summary>
        : Type<
            /*condition=*/CPred<"$_self.isa<"#concept#">()">,
            /*descr=*/summary,
            /*cppClassName=*/concept
        >;

class NativeAttrConcept<string concept, string summary>
        : Attr<
            /*condition=*/CPred<"$_self.isa<"#concept#">()">,
            /*descr=*/summary
        > {
    let storageType = concept;
    let returnType = "typename "#concept#"::ValueType";
    let convertFromStorage = "$_self.getValue()";
    let valueType = NoneType;
}

// Template for an AttrConstraint on TypeAttr that matches the contained type.
class TypeAttrConstraint<Type type>
        : AttrConstraint<
            SubstLeaves<
                "$_self",
                "$_self.cast<::mlir::TypeAttr>().getValue()",
                type.predicate
            >,
            "that is a " # type.summary
        >;

// TypeAttribute with a constrained type.
// NOTE: TypeAttrBase does the same thing but with strings instead of tablegen
//       type constraints.
class ConstrainedTypeAttr<Type type>
        : Confined<TypeAttr, [TypeAttrConstraint<type>]>;

#endif
