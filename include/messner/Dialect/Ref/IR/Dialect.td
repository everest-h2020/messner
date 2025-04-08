//===- Dialect.td ------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect.
//
//===----------------------------------------------------------------------===//

#ifndef REF_DIALECT
#define REF_DIALECT

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
        memory. This is different from `memref` and `ptr` in that it carries
        effects in types instead of operations.

        References in the `ref` dialect are inteded to be used as a lowering
        step in a memory ownership and consistency model. For this purpose,
        references defined through this dialect are qualified by their access
        effects.
    }];

    let dependentDialects = [];

    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 1;

    code extraClassDeclaration = [{
        auto parseType(DialectAsmParser &parser) const -> Type override;
        void printType(Type type, DialectAsmPrinter &os) const override;

    private:
        void registerTypes();
        void registerAttributes();
        void registerOps();
    }];
}

// Template for types.
class Ref_Type<string name, list<Trait> traits = []>
        : TypeDef<Ref_Dialect, name, traits>;
// Template for attributes.
class Ref_Attr<string name, list<Trait> traits = []>
        : AttrDef<Ref_Dialect, name, traits>;
// Template for ops.
class Ref_Op<string mnemonic, list<Trait> traits = []>
        : Op<Ref_Dialect, mnemonic, traits>;

#endif // REF_DIALECT
