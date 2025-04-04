//===- Enums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect enums.
//
//===----------------------------------------------------------------------===//

#ifndef REF_ENUMS
#define REF_ENUMS

include "mlir/IR/EnumAttr.td"

def Ref_Read : I64BitEnumAttrCaseBit<"Read", 0, "r">;
def Ref_Pure : I64BitEnumAttrCaseBit<"Pure", 1, "p">;
def Ref_Write : I64BitEnumAttrCaseBit<"Write", 2, "w">;
def Ref_Idempotent : I64BitEnumAttrCaseBit<"Idempotent", 3, "i">;
def Ref_Exclusive : I64BitEnumAttrCaseBit<"Exclusive", 4, "e">;

def Ref_PureRead : I64BitEnumAttrCaseGroup<
        "PureRead", [Ref_Pure, Ref_Read], "pr">;
def Ref_IdempotentWrite : I64BitEnumAttrCaseGroup<
        "IdempotentWrite", [Ref_Idempotent, Ref_Write], "iw">;
def Ref_ReadOnly : I64BitEnumAttrCaseGroup<
        "ReadOnly", [Ref_Exclusive, Ref_PureRead], "ro">;
def Ref_Shared : I64BitEnumAttrCaseGroup<
        "Shared", [Ref_PureRead, Ref_IdempotentWrite], "shared">;
def Ref_Unique : I64BitEnumAttrCaseGroup<
        "Unique", [Ref_Exclusive, Ref_IdempotentWrite, Ref_PureRead], "unique">;

def Ref_ReferenceKind : I64BitEnumAttr<
        "ReferenceKind",
        "reference cell access behavior",
        [
            Ref_Pure,
            Ref_Read,
            Ref_PureRead,
            Ref_Idempotent,
            Ref_Write,
            Ref_IdempotentWrite,
            Ref_Exclusive,
            Ref_ReadOnly,
            Ref_Shared,
            Ref_Unique]> {
    let cppNamespace = "::mlir::ref";
    let printBitEnumPrimaryGroups = 1;
}

#endif // REF_ENUMS
