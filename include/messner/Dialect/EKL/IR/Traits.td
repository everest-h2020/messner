//===- Traits.td -------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect traits.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_TRAITS
#define EKL_TRAITS

include "messner/Dialect/EKL/Interfaces/SemaOpInterface.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for a NativeTrait implemented in the `ekl` namespace.
class EKL_Trait<string name, string entityType, list<Trait> traits = []>
        : NativeTrait<name, entityType> {
    string trait = name;
    string cppNamespace = "::mlir::ekl::" # entityType # "Trait";

    list<Trait> dependentTraits = traits;
}

// Template for op traits.
class EKL_OpTrait<string name, list<Trait> traits = []>
        : EKL_Trait<name, "Op", traits>;

//===----------------------------------------------------------------------===//
// Op traits
//===----------------------------------------------------------------------===//

// Trait for top-level program declarations.
def EKL_Declaration : TraitList<[
        IsolatedFromAbove,
        EKL_OpTrait<"Declaration">,
        HasParent<"ProgramOp">]>;

// Trait for ops whose regions are functors.
def EKL_FunctorRegionsImpl
        : EKL_OpTrait<"FunctorRegions", [
            SingleBlock,
            SingleBlockImplicitTerminatorImpl<"YieldOp">]>,
          StructuralOpTrait;

def EKL_FunctorRegions : TraitList<[
        SingleBlockImplicitTerminator<"YieldOp">,
        RecursiveMemoryEffects,
        EKL_FunctorRegionsImpl]>;

// Trait for ops that participate in semantic analysis.
def EKL_SemaOp : TraitList<[
        ConditionallySpeculatable,
        EKL_SemaOpInterface,
        EKL_OpTrait<"SemaOpImpl">]>;

// Trait that identifies a statement.
def EKL_Statement : TraitList<[
        EKL_SemaOp,
        EKL_OpTrait<"Statement">]>;

// Trait that identifies an expression.
def EKL_Expression : TraitList<[
        EKL_SemaOp,
        EKL_OpTrait<"Expression">]>;

// Trait that identifies a cast.
def EKL_Cast : TraitList<[
        EKL_Expression,
        EKL_OpTrait<"CastOpImpl">,
        CastOpInterface]>;

// Trait that identifies a generator expression.
def EKL_Generator : TraitList<[
        EKL_FunctorRegions,
        EKL_Expression,
        EKL_OpTrait<"Generator">]>;

// Trait that identifies a relational expression.
def EKL_Relational : TraitList<[
        NoMemoryEffect,
        EKL_Expression,
        EKL_OpTrait<"Relational">]>;

// Trait that identifies a logical expression.
def EKL_Logical : TraitList<[
        NoMemoryEffect,
        EKL_Expression,
        EKL_OpTrait<"Logical">]>;

// Trait that identifies an arithmetic expression.
def EKL_Arithmetic : TraitList<[
        NoMemoryEffect,
        EKL_Expression,
        EKL_OpTrait<"Arithmetic">]>;

#endif // EKL_TRAITS
