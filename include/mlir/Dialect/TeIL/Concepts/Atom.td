//===- Atom.td - TeIL Atom concept -------------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL Atom concept.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_CONCEPTS_ATOM
#define TEIL_CONCEPTS_ATOM

include "mlir/Concepts/Concepts.td"

def TeIL_Atom
        : NativeTypeConcept<"::mlir::teil::AtomType", "atom">;

def TeIL_AtomType
        : ConstrainedTypeAttr<TeIL_Atom>;

#endif
