//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef REF_OPS
#define REF_OPS

include "messner/Dialect/Ref/IR/Attributes.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ref_ReadOp : Ref_Op<
        "read",
        [
            // Models read effects, and write effects if volatile.
            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
            // Depending on the reference kind, may be speculatable.
            DeclareOpInterfaceMethods<ConditionallySpeculatable>,
            // The result type is given by the reference cell type.
            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Reads the value in a cell";
    let description = [{
        The `ref.read` operation reads a value from a cell indicated by a
        reference value.

        The reference must be readable. The reference kind can also indicate
        additional side-effects that happen during a read. The result type must
        match the reference cell type.

        Example:

        ```
        // Reads an i32 value from a reference to pure read-only memory.
        ref.read %ref : !ref<ro, i32>

        // Reads from the same reference, but explicitly pessimizes it first.
        ref.read %ref volatile : !ref<ro, i32>
        ref.read %ref impure : !ref<ro, i32>
        ref.read %ref volatile impure : !ref<ro, i32>
        ```
    }];

    let arguments = (ins
        Ref_ReadableRef:$reference,
        OptionalAttr<UnitAttr>:$isImpure,
        OptionalAttr<UnitAttr>:$isVolatile);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins
            "ReadableRef":$reference,
            CArg<"bool", "false">:$isImpure,
            CArg<"bool", "false">:$isVolatile)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $reference (`volatile` $isVolatile^)? (`impure` $isImpure^)?
        `:` type($reference) attr-dict
    }];

    code extraClassDeclaration = [{
        /// Determines if this read is volatile.
        auto isVolatile() -> bool;
        /// Determines if this read is impure.
        auto isImpure() -> bool;
    }];
}

def Ref_WriteOp : Ref_Op<
        "write",
        [
            // Models write effects.
            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
            // Depending on the reference kind, may be speculatable.
            DeclareOpInterfaceMethods<ConditionallySpeculatable>,
            // The value type is given by the reference cell type.
            TypesMatchWith<
                "type of 'value' matches cell type of 'reference'",
                "reference", "value",
                "::llvm::cast<ReferenceType>($_self).getCellType()">]> {
    let summary = "Writes the value behind a reference";
    let description = [{
        The `ref.write` operation writes a value to a cell indicated by a
        reference value.

        The reference must be writable. The reference kind can also indicate
        additional side-effects that happen during a write. The value type must
        match the reference cell type.

        Example:

        ```
        // Writes an i32 value to a unique reference.
        ref.write %val to %ref : !ref<unique, i32>

        // Writes to the same reference, but explicitly pessimizes it first.
        ref.write %val to %ref volatile : !ref<unique, i32>
        ```
    }];

    let arguments = (ins
        AnyType:$value,
        Ref_WritableRef:$reference,
        OptionalAttr<UnitAttr>:$isVolatile);

    let builders = [
        OpBuilder<(ins
            "Value":$value,
            "WritableRef":$reference,
            CArg<"bool", "false">:$isVolatile)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $value `to` $reference (`volatile` $isVolatile^)?
        `:` type($reference) attr-dict
    }];

    code extraClassDeclaration = [{
        /// Determines if this write is volatile.
        auto isVolatile() -> bool;
    }];
}

#endif // REF_OPS
