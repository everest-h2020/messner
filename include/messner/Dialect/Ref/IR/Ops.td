//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef REF_OPS
#define REF_OPS

include "messner/Dialect/Ref/IR/Types.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ref_ReadOp : Ref_Op<
        "read",
        [
            // Reads from the default memory resource.
            MemoryEffects<[MemRead]>,
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
        ref.read %ref : !ref<pr, i32>
        ```
    }];

    let arguments = (ins Ref_ReadableRef:$reference);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $reference `:` type($reference) attr-dict
    }];
}

def Ref_WriteOp : Ref_Op<
        "write",
        [
            // Writes to the default memory resource.
            MemoryEffects<[MemWrite]>,
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
        ref.write %val to %ref : !ref<iw, i32>
        ```
    }];

    let arguments = (ins
        AnyType:$value,
        Ref_WritableRef:$reference);

    let assemblyFormat = [{
        $value `to` $reference `:` type($reference) attr-dict
    }];
}

#endif // REF_OPS
