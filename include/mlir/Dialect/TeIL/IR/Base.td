//===- Base.td - TeIL dialect base -------------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_BASE
#define TEIL_BASE

include "mlir/Dialect/TeIL/Enums.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def TeIL_Dialect : Dialect {
    let name = "teil";
    let cppNamespace = "::mlir::teil";

    let summary = "The Tensor Intermediate Language dialect";
    let description = [{
        TODO
    }];

    let dependentDialects = [
        "tensor::TensorDialect"
    ];

    code extraClassDeclaration = [{
    // NOTE: To allow splitting the implementation of ::initialize() across
    //       multiple translation units, we declare additional registry
    //       methods.
    private:
        /** Registers all dialect attributes. */
        void registerAttrs();
        /** Registers all dialect operations. */
        void registerOps();
        /** Registers all dialect types. */
        void registerTypes();
    }];

    let hasConstantMaterializer = 1;
}

// Template for attributes.
class TeIL_Attr<string name, list<Trait> traits = []>
        : AttrDef<TeIL_Dialect, name, traits>;
// Template for ops.
class TeIL_Op<string mnemonic, list<Trait> traits = []>
        : Op<TeIL_Dialect, mnemonic, traits>;
// Template for types.
class TeIL_Type<string name, list<Trait> traits = []>
        : TypeDef<TeIL_Dialect, name, traits>;

#endif
