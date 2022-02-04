//===- Scalar.td - TeIL Scalar concept ---------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL Scalar concept.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_CONCEPTS_SCALAR
#define TEIL_CONCEPTS_SCALAR

include "mlir/Concepts/Concepts.td"

def TeIL_Scalar
        : NativeTypeConcept<"::mlir::teil::ScalarType", "scalar">;

def TeIL_ScalarType
        : ConstrainedTypeAttr<TeIL_Scalar>;

#endif
