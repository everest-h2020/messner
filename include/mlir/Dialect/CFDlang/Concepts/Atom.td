//===- Atom.td - CFDlang Atom concept ----------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the CFDlang Atom concept.
//
//===----------------------------------------------------------------------===//

#ifndef CFDLANG_CONCEPTS_ATOM
#define CFDLANG_CONCEPTS_ATOM

include "mlir/Concepts/Concepts.td"

def CFDlang_Atom
        : NativeTypeConcept<"::mlir::cfdlang::AtomType", "atom">;

def CFDlang_AtomType
        : NativeAttrConcept<"mlir::cfdlang::AtomTypeAttr", "atom type">;

def CFDlang_AtomTypes
        : NativeAttrConcept<"::mlir::cfdlang::AtomTypeArrayAttr", "atom types">;

#endif
