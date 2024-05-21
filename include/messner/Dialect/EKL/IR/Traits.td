//===- Traits.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect traits.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_TRAITS
#define EKL_TRAITS

include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/Traits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for a NativeTrait implemented in the `ekl` namespace.
class EKL_Trait<string name, string entityType, list<Trait> traits = []>
        : NativeTrait<name, entityType> {
    string trait = name;
    string cppNamespace = "::mlir::ekl::" # entityType # "Trait";

    list<Trait> dependentTraits = traits;
}

// Template for type traits.
class EKL_TypeTrait<string name, list<Trait> traits = []>
        : EKL_Trait<name, "Type", traits>;
// Template for attribute traits.
class EKL_AttributeTrait<string name, list<Trait> traits = []>
        : EKL_Trait<name, "Attribute", traits>;
// Template for op traits.
class EKL_OpTrait<string name, list<Trait> traits = []>
        : EKL_Trait<name, "Op", traits>;

//===----------------------------------------------------------------------===//
// Op traits
//===----------------------------------------------------------------------===//

// Trait that identifies a symbol declaration.
def EKL_SymbolOp : TraitList<[
    Symbol,
    IsolatedFromAbove,
    HasParent<"::mlir::ekl::ProgramOp">,
    EKL_OpTrait<"IsSymbol">]>;

// Trait that marks an operation that has functor regions.
def EKL_HasFunctorsTrait
        : EKL_OpTrait<"HasFunctors", [SingleBlock]>,
          StructuralOpTrait;
def EKL_HasFunctors : TraitList<[
    RecursiveMemoryEffects,
    SingleBlock,
    EKL_HasFunctorsTrait]>;

// Trait that identifies a statement.
def EKL_StatementOp : TraitList<[
    DeclareOpInterfaceMethods<EKL_TypeCheckOpInterface>,
    EKL_OpTrait<"IsStatement">]>;

// Trait that identifies an expression.
def EKL_ExpressionOp : TraitList<[
    ConditionallySpeculatable,
    DeclareOpInterfaceMethods<EKL_TypeCheckOpInterface>,
    EKL_OpTrait<"IsExpression">]>;

// Trait that identifies a generator expression.
def EKL_GeneratorOp : TraitList<[
    EKL_HasFunctors,
    EKL_ExpressionOp,
    EKL_OpTrait<"IsGenerator">]>;

// Trait that identifies a relational operator expression.
def EKL_RelationalOp : TraitList<[
    EKL_ExpressionOp,
    EKL_OpTrait<"IsRelational">]>;

// Trait that identifies a logical operator expression.
def EKL_LogicalOp : TraitList<[
    EKL_ExpressionOp,
    EKL_OpTrait<"IsLogical">]>;

// Trait that identifies an arithmetic operator expression.
def EKL_ArithmeticOp : TraitList<[
    EKL_ExpressionOp,
    EKL_OpTrait<"IsArithmetic">]>;

#endif
