//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect enums.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_ENUMS
#define EKL_ENUMS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

// Helper to declare optional attribute/type parameters based on enums.
class OptionalEnumParameter<EnumAttrInfo enumInfo, EnumAttrCaseInfo default>
        : DefaultValuedParameter<
            enumInfo.cppNamespace # "::" # enumInfo.className,
            enumInfo.cppNamespace # "::" # enumInfo.className # "::" # default.symbol,
            "an enum of type " # enumInfo.className> {
    let parser = enumInfo.parameterParser;
    let printer = enumInfo.parameterPrinter;
}

//===----------------------------------------------------------------------===//
// ReferenceKind
//===----------------------------------------------------------------------===//

def EKL_InReference    : I64BitEnumAttrCaseBit<"In", 0, "in">;
def EKL_OutReference   : I64BitEnumAttrCaseBit<"Out", 1, "out">;
def EKL_InOutReference : I64BitEnumAttrCaseGroup<
        "InOut",
        [EKL_InReference, EKL_OutReference],
        "inout">;

def EKL_ReferenceKind : I64BitEnumAttr<
        "ReferenceKind",
        "reference directionality kind",
        [
            EKL_InReference,
            EKL_OutReference,
            EKL_InOutReference]> {
    let cppNamespace = "::mlir::ekl";

    let printBitEnumPrimaryGroups = 1;
}

//===----------------------------------------------------------------------===//
// RelationKind
//===----------------------------------------------------------------------===//

def EKL_Equivalent     : I64BitEnumAttrCaseBit<"Equivalent", 0b001, "eq">;
def EKL_LessThan       : I64BitEnumAttrCaseBit<"LessThan", 0b010, "lt">;
def EKL_LessOrEqual    : I64BitEnumAttrCaseBit<"LessOrEqual", 0b011, "le">;
def EKL_GreaterThan    : I64BitEnumAttrCaseBit<"GreaterThan", 0b100, "gt">;
def EKL_GreaterOrEqual : I64BitEnumAttrCaseBit<"GreaterOrEqual", 0b101, "ge">;
def EKL_Antivalent     : I64BitEnumAttrCaseBit<"Antivalent", 0b110, "ne">;

def EKL_RelationKind : I64BitEnumAttr<
        "RelationKind",
        "relational operator kind",
        [
            EKL_Equivalent,
            EKL_LessThan,
            EKL_LessOrEqual,
            EKL_GreaterThan,
            EKL_GreaterOrEqual,
            EKL_Antivalent]> {
    let cppNamespace = "::mlir::ekl";
}

#endif
