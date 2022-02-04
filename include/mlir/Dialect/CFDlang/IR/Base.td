//===- Base.td - CFDlang dialect base ----------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the CFDlang dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef CFDLANG_BASE
#define CFDLANG_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def CFDlang_Dialect : Dialect {
    let name = "cfdlang";
    let cppNamespace = "::mlir::cfdlang";

    let summary = "The CFDlang front-end dialect";
    let description = [{
        TODO
    }];

    let dependentDialects = [
        "StandardOpsDialect",
        "tensor::TensorDialect",
        "teil::TeILDialect"
    ];

    code extraClassDeclaration = [{
    // NOTE: To allow splitting the implementation of ::initialize() across
    //       multiple translation units, we declare additional registry
    //       methods.
    private:
        /** Registers all dialect attributes. */
        /*void registerAttrs();*/
        /** Registers all dialect operations. */
        void registerOps();
        /** Registers all dialect types. */
        void registerTypes();
    }];

    let hasConstantMaterializer = 1;
}

// Template for attributes.
/*
class CFDlang_Attr<string name, list<Trait> traits = []>
        : AttrDef<CFDlang_Dialect, name, traits>;
*/
// Template for ops.
class CFDlang_Op<string mnemonic, list<OpTrait> traits = []>
        : Op<CFDlang_Dialect, mnemonic, traits>;
// Template for types.
class CFDlang_Type<string name, list<Trait> traits = []>
        : TypeDef<CFDlang_Dialect, name, traits>;

#endif
