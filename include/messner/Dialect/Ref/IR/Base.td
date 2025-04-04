//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef REF_BASE
#define REF_BASE

include "messner/Dialect/Ref/Enums.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Ref_Dialect : Dialect {
    let name = "ref";
    let cppNamespace = "::mlir::ref";

    let summary = "Memory reference cell dialect";
    let description = [{
        The `ref` dialect provides a common framework for dealing with reference
        cells, i.e., strongly-typed references to locations in heterogeneous
        memory. This is different from `memref` and `ptr` in that it does not
        require or guarantee a machine representation.

        References in the `ref` dialect are inteded to be used as a lowering
        step in a memory ownership and consistency model. For this purpose,
        references defined through this dialect are qualified by their access
        behaviors.
    }];

    let dependentDialects = [];

    let useDefaultTypePrinterParser = 0;

    code extraClassDeclaration = [{
        Type parseType(DialectAsmParser &parser) const override;
        void printType(Type type, DialectAsmPrinter &os) const override;

    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for ops.
class Ref_Op<string mnemonic, list<Trait> traits = []>
        : Op<Ref_Dialect, mnemonic, traits>;
// Template for types.
class Ref_Type<string name, list<Trait> traits = []>
        : TypeDef<Ref_Dialect, name, traits>;

#endif // REF_BASE
