//===- Dialect.td ------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_DIALECT
#define EKL_DIALECT

include "messner/Dialect/EKL/Enums.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def EKL_Dialect : Dialect {
    let name = "ekl";
    let cppNamespace = "::mlir::ekl";

    let summary = "EVEREST kernel language dialect";
    let description = [{
        The `ekl` dialect implements the syntax and semantics of the EVEREST
        kernel language (EKL) DSL.

        This dialect makes use of the mlir-type-system proposal, meaning that it
        implements the DialectTypeSystemInterface and attaches the
        TypeCheckOpInterface to its operations. Programs in the `ekl` dialect
        can thus be in syntactically typed or semantically typed form, the
        latter of which is established using gradual-refinement type checking.
    }];

    let dependentDialects = [
        "ref::RefDialect"];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 0;

    code extraClassDeclaration = [{
        /// Gets the DialectTypeSystemInterface implementation.
        auto getTypeSystem() const -> const TypeSystem & { return *_typeSystem; }

        auto parseType(DialectAsmParser &parser) const -> Type override;
        void printType(Type type, DialectAsmPrinter &os) const override;

        auto parseAttribute(DialectAsmParser &parser, Type type) const -> Attribute override;
        void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

    private:
        void registerTypes();
        void registerAttributes();
        void registerOps();

        TypeSystem *_typeSystem = nullptr;
    }];
}

// Template for types.
class EKL_Type<string name, list<Trait> traits = []>
        : TypeDef<EKL_Dialect, name, traits>;
// Template for attributes.
class EKL_Attr<string name, list<Trait> traits = []>
        : AttrDef<EKL_Dialect, name, traits>;
// Template for ops.
class EKL_Op<string mnemonic, list<Trait> traits = []>
        : Op<EKL_Dialect, mnemonic, traits>;

#endif // EKL_DIALECT
