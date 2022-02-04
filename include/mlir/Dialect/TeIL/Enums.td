//===- Enums.td - TeIL enumerations ------------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_ENUMS
#define TEIL_ENUMS

include "mlir/IR/OpBase.td"

def TeIL_Reduction_Add : I64EnumAttrCase<"Add", 0, "add">;
def TeIL_Reduction_Mul : I64EnumAttrCase<"Mul", 1, "mul">;
def TeIL_Reduction_Min : I64EnumAttrCase<"Min", 2, "min">;
def TeIL_Reduction_Max : I64EnumAttrCase<"Max", 3, "max">;

// Any kind of (conceptually) commutative reduction operation.
def TeIL_ReductionKindAttr : I64EnumAttr<
    "ReductionKind",
    "Kind of reduction operation to apply",
    [
        TeIL_Reduction_Add,
        TeIL_Reduction_Mul,
        TeIL_Reduction_Min,
        TeIL_Reduction_Max
    ]
> {
    let cppNamespace = "::mlir::teil";
}

#endif
