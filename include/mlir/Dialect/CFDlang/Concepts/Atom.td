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
        : ConstrainedTypeAttr<CFDlang_Atom>;

#endif
