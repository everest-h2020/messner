//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_BASE
#define EKL_BASE

include "messner/Dialect/EKL/Enums.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def EKL_Dialect : Dialect {
    let name = "ekl";
    let cppNamespace = "::mlir::ekl";

    let summary = "The EVEREST kernel language dialect";
    let description = [{
        TODO
    }];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 0;
    let useDefaultTypePrinterParser = 0;

    code extraClassDeclaration = [{
        Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
        void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

        Type parseType(DialectAsmParser &parser) const override;
        void printType(Type type, DialectAsmPrinter &os) const override;

    private:
        void registerAttributes();
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class EKL_Attr<string name, list<Trait> traits = []>
        : AttrDef<EKL_Dialect, name, traits>;
// Template for ops.
class EKL_Op<string mnemonic, list<Trait> traits = []>
        : Op<EKL_Dialect, mnemonic, traits>;
// Template for types.
class EKL_Type<string name, list<Trait> traits = []>
        : TypeDef<EKL_Dialect, name, traits>;

#endif
