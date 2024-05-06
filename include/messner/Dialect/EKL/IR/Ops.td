//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_OPS
#define EKL_OPS

include "messner/Dialect/EKL/IR/Attributes.td"
include "messner/Dialect/EKL/Interfaces/ExpressionOp.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class EKL_SymbolBase<string mnemonic, list<Trait> traits = []> : EKL_Op<
        mnemonic,
        [
            IsolatedFromAbove,
            Symbol,
            HasParent<"ProgramOp">] # traits> {
    let arguments = (ins SymbolNameAttr:$sym_name);

    let assemblyFormat = [{
        $sym_name attr-dict
    }];
}

class EKL_StatementBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [OpAsmOpInterface] # traits> {
    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
            for (auto res : (*this)->getResults()) setNameFn(res, "expr");
        }

        static StringRef getDefaultDialect() {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

class EKL_ExpressionBase<string mnemonic, list<Trait> traits = []>
        : EKL_StatementBase<
            mnemonic,
            [DeclareOpInterfaceMethods<EKL_ExpressionOp>] # traits> {
    let results = (outs EKL_ExpressionType:$result);
}

class EKL_CastBase<string name, list<Trait> traits = []>
        : EKL_ExpressionBase<name, [Pure, CastOpInterface] # traits> {
    let arguments = (ins EKL_ExpressionType:$input);

    let assemblyFormat = [{
        custom<ExprOperand>($input, type($input))
        attr-dict
        custom<ExprResult>(type($result))
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // CastOpInterface
        //===--------------------------------------------------------------===//

        static bool areCastCompatible(TypeRange, TypeRange)
        {
            // This is decided by the type checker.
            return true;
        }
    }];
}

class EKL_Operator<string name, list<Trait> traits = []>
        : EKL_ExpressionBase<"op_" # name, [Pure] # traits> {
}

class EKL_UnaryOperator<string name, list<Trait> traits = []>
        : EKL_Operator<name, traits> {
    let arguments = (ins EKL_ExpressionType:$operand);

    let assemblyFormat = [{
        custom<ExprOperand>($operand, type($operand))
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

class EKL_BinaryOperator<string name, list<Trait> traits = []>
        : EKL_Operator<name, traits> {
    let arguments = (ins EKL_ExpressionType:$lhs, EKL_ExpressionType:$rhs);

    let assemblyFormat = [{
        custom<ExprOperand>($lhs, type($lhs))```,`
        custom<ExprOperand>($rhs, type($rhs))
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Containers
//===----------------------------------------------------------------------===//

def EKL_ProgramOp : EKL_Op<
        "program",
        [
            IsolatedFromAbove,
            SymbolTable,
            SingleBlock,
            NoRegionArguments,
            NoTerminator,
            OpAsmOpInterface]> {
    let summary = "Contains an EKL compilation unit";
    let description = [{
        The `ekl.program` operation is a container for an EKL compilation unit,
        which will result in a module.

        Operations nested under this operation can either produce constant
        expressions, or declare or define symbols. These are:

            - `ekl.global`
            - `ekl.kernel`

        Examples:

        ```
        ekl.program {
            // ...
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        attr-dict-with-keyword $body
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect() {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Symbols
//===----------------------------------------------------------------------===//

def EKL_GlobalOp : EKL_SymbolBase<"global"> {
    let summary = "Introduces a globally available reference symbol";
    let description = [{
        The `ekl.global` operation introduces a symbol that constitutes a
        globally available reference to some fixed-size segment of contiguous
        data, i.e., an `!ekl.array`.

        The type of the global reference must be an ABI-compatible `!ekl.ref`
        type, meaning that it must point to an ABI-compatible scalar type. The
        reference type may have an access qualifier that indicates whether it
        can be read from or written to.

        Public symbols must have an initializer, while non-public symbols may
        be used to refer to externally initialized data. Initializers on private
        symbols which do not allow reading are pointless and will be removed.

        Examples:

        ```
        ekl.global private @G1 : !ekl.ref<in i1>

        ekl.global @G2 : !ekl.ref<inout f32[2, 2]>
            = #ekl.init<array<f32: 0.0f, 1.0f, -1.0f, 0.0f>, [2, 2]>
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<EKL_ABIReferenceType>:$type,
        OptionalAttr<EKL_InitializerAttr>:$initializer,
        OptionalAttr<StrAttr>:$sym_visibility);

    let assemblyFormat = [{
        custom<SymbolDeclaration>($sym_visibility, $sym_name)
        `:` $type ( `=` $initializer^ )?
        attr-dict
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // SymbolOpInterface
        //===--------------------------------------------------------------===//

        bool isDeclaration() { return !getInitializer(); }
    }];
}

def EKL_KernelOp : EKL_SymbolBase<
        "kernel",
        [SingleBlock, NoTerminator, OpAsmOpInterface]> {
    let summary = "Introduces an exported kernel symbol";
    let description = [{
        The `ekl.kernel` operation defines a public callable symbol that will be
        implemented by this EKL program.

        The kernel callable may not produce any return value, but accepts any
        number of ABI-compatible arguments, which include references. The name
        of the symbol will be used to provide an unmangled C-ABI entry point for
        external callers. The calling convention is determined by the LLVM
        lowering.

        Examples:

        ```
        ekl.kernel @my_kernel(%arg0: !ekl.ref<out f32>) {
            // ...
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $sym_name``custom<KernelBody>($body, attr-dict)
    }];

    let hasVerifier = 1;

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect() {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def EKL_ReadOp : EKL_StatementBase<
        "read",
        [Pure, DeclareOpInterfaceMethods<EKL_ExpressionOp>]> {
    let summary = "Dereferences a reference to obtain its value";
    let description = [{
        The `ekl.read` operation reads a value from a location indicated by a
        reference value.

        Example:

        ```
        ekl.read %ref : !ekl.ref<in si32> -> si32
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$reference);
    let results = (outs EKL_ExpressionType:$result);

    let assemblyFormat = [{
        custom<ExprOperand>($reference, type($reference))
        custom<ExprResult>(type($result))
        attr-dict
    }];
}

def EKL_WriteOp : EKL_StatementBase<
        "write",
        [DeclareOpInterfaceMethods<EKL_ExpressionOp>]> {
    let summary = "Dereferences a reference to update its value";
    let description = [{
        The `ekl.write` operation writes a value to a location indicated by a
        reference value.

        Example:

        ```
        ekl.write %ref : !ekl.ref<out si64>, %v : si32
        ```
    }];

    let arguments = (ins
        EKL_ExpressionType:$reference,
        EKL_ExpressionType:$value);

    let assemblyFormat = [{
        custom<ExprOperand>($reference, type($reference))```,`
        custom<ExprOperand>($value, type($value))
        attr-dict
    }];
}

def EKL_IfOp : EKL_ExpressionBase<
        "if",
        [
            RecursiveMemoryEffects,
            RecursivelySpeculatable,
            SingleBlockImplicitTerminator<"ekl::YieldOp">,
            DeclareOpInterfaceMethods<EKL_ExpressionOp>]> {
    let summary = "Conditionally evaluates statements and expressions";
    let description = [{
        The `ekl.if` operation allows conditionally executing a statement or
        determining a result value.

        Example:

        ```
        ekl.if %c then {
            write %v1, %v2
        }

        %q = ekl.if %c then {
            yield %v1
        } else {
            yield %v2
        } -> !ekl.expr
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$condition);
    let regions = (region
        SizedRegion<1>:$thenBranch,
        MaxSizedRegion<1>:$elseBranch);
    let results = (outs Optional<EKL_ExpressionType>:$result);

    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<ExprOperand>($condition, type($condition))
        attr-dict-with-keyword
        `then` custom<Functor>($thenBranch)
        custom<ElseBranch>($elseBranch, type($result))
    }];
}

// TODO: PanicOp

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

def EKL_UnifyOp : EKL_CastBase<"unify"> {
    let summary = "Upcasts a value";
    let description = [{
        The `ekl.unify` operation represents a value-preserving upcast.

        Example:

        ```
        ekl.unify %a: si32 -> si64
        ```
    }];
}

def EKL_BroadcastOp : EKL_CastBase<"bcast"> {
    let summary = "Broadcasts a value";
    let description = [{
        The `ekl.bcast` operation broadcasts a value to an array.

        Example:

        ```
        ekl.bcast %a: !ekl.array<si32[4, 1]>, [4, 4] -> !ekl.array<si32[4, 4]>
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$input, DenseI64ArrayAttr:$extents);
    let results = (outs EKL_ExpressionType:$result);

    let assemblyFormat = [{
        custom<ExprOperand>($input, type($input))`,`
        $extents
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

def EKL_CoerceOp : EKL_CastBase<"coerce"> {
    let summary = "Coerces a value";
    let description = [{
        The `ekl.coerce` operation coerces a value to a different type, which
        may not preserve the value.

        Example:

        ```
        ekl.coerce %a: !ekl.array<si32> -> !ekl<_4>
        ```
    }];
}

//===----------------------------------------------------------------------===//
// Primary expressions
//===----------------------------------------------------------------------===//

def EKL_LiteralOp : EKL_Op<
        "literal",
        [
            Pure,
            ConstantLike,
            OpAsmOpInterface,
            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Obtains a literal expression";
    let description = [{
        The `ekl.literal` operation returns the value of a literal expression.

        Example:

        ```
        %lit = ekl.literal -4 : si32
        ```
    }];

    let arguments = (ins EKL_LiteralAttr:$literal);
    let results = (outs EKL_ExpressionType:$result);

    let hasVerifier = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        $literal attr-dict
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
            setNameFn(getResult(), "lit");
        }
    }];
}

def EKL_GetGlobalOp : EKL_Op<
        "get_global",
        [
            Pure,
            DeclareOpInterfaceMethods<SymbolUserOpInterface>,
            OpAsmOpInterface]> {
    let summary = "Obtains a global symbol reference value";
    let description = [{
        The `ekl.get_global` operation allows getting a reference value to a
        previously declared global symbol.

        Example:

        ```
        ekl.get_global @G2 -> !ekl.ref<in f32[2, 2]>
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$global_name);
    let results = (outs EKL_ExpressionType:$result);

    let assemblyFormat = [{
        $global_name attr-dict custom<ExprResult>(type($result))
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
            setNameFn(getResult(), getGlobalName());
        }
    }];
}

def EKL_SubscriptOp : EKL_ExpressionBase<"subscript", [Pure]> {
    let summary = "Subscript operator";

    let arguments = (ins
        EKL_ExpressionType:$array,
        Variadic<EKL_ExpressionType>:$subscripts);

    let assemblyFormat = [{
        custom<ExprOperand>($array, type($array))
        ` ``[` custom<ExprOperand>($subscripts, type($subscripts)) `]`
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Generator expressions
//===----------------------------------------------------------------------===//

def EKL_YieldOp : EKL_Op<
        "yield",
        [Pure, Terminator, DeclareOpInterfaceMethods<EKL_ExpressionOp>]> {
    let summary = "Region terminator for scopes";
    let description = [{
        The `ekl.yield` operation terminates EKL expression scopes that yield
        some value to the parent scope.

        Example:

        ```
        ekl.yield %0: si32
        ```
    }];

    let arguments = (ins Optional<EKL_ExpressionType>:$value);

    let assemblyFormat = [{
        custom<OptionalExprOperand>($value, type($value))
        attr-dict
    }];
}

def EKL_StackOp : EKL_ExpressionBase<"stack", [Pure]> {
    let summary = "Stacking operator";
    let description = [{
        The `ekl.stack` operation produces an array by concatenating its inputs
        in a new outer extent.

        Example:

        ```
        %q = ekl.stack %0: !ekl.array<si32[3]>, %1: si32 -> !ekl.array<si32[2, 3]>
        ```
    }];

    let arguments = (ins Variadic<EKL_ExpressionType>:$operands);

    let assemblyFormat = [{
        custom<ExprOperand>($operands, type($operands))
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

def EKL_AssocOp : EKL_ExpressionBase<
        "assoc",
        [
            RecursiveMemoryEffects,
            RecursivelySpeculatable,
            DeclareOpInterfaceMethods<
                OpAsmOpInterface,
                ["getAsmBlockArgumentNames"]>,
            SingleBlockImplicitTerminator<"ekl::YieldOp">]> {
    let summary = "Index association operator";
    let description = [{
        The `ekl.assoc` operation produces an array or scalar value by
        associating every tuple of its index domain with a value.

        Contained `ekl.subscript` operations will eagerly propagate known index
        bounds to the bounds on the index domain. This means that it is not
        legal to reuse an index for dimensions of different sizes.

        Example:

        ```
        %q = ekl.assoc (%i) {
            %0 = subscript %v [%i]
            yield %0
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$map);

    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<Functor>($map)
        attr-dict-with-keyword
        custom<ExprResult>(type($result))
    }];
}

def EKL_ReduceOp : EKL_ExpressionBase<
        "reduce",
        [
            RecursiveMemoryEffects,
            RecursivelySpeculatable,
            DeclareOpInterfaceMethods<
                OpAsmOpInterface,
                ["getAsmBlockArgumentNames"]>,
            SingleBlockImplicitTerminator<"ekl::YieldOp">]> {
    let summary = "Index reduction operator";
    let description = [{
        The `ekl.reduce` operation produces an array or scalar value by reducing
        over an index domain.

        Contained `ekl.subscript` operations will eagerly propagate known index
        bounds to the bounds on the index domain. This means that it is not
        legal to reuse an index for dimensions of different sizes.

        Example:

        ```
        %q = ekl.reduce (%i) {
            %0 = subscript %v [%i]
            yield %0
        } using (%l, %r) {
            %1 = op_add %l, %r
            yield %1
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$map, SizedRegion<1>:$reduction);

    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<Functor>($map)
        `using` custom<Functor>($reduction)
        attr-dict-with-keyword
        custom<ExprResult>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Relational operators
//===----------------------------------------------------------------------===//

def EKL_CompareOp : EKL_BinaryOperator<"cmp"> {
    let summary = "Comparison operator";
    let arguments = (ins
        EKL_RelationKind:$kind,
        EKL_ExpressionType:$lhs,
        EKL_ExpressionType:$rhs);

    let assemblyFormat = [{
        $kind
        custom<ExprOperand>($lhs, type($lhs))```,`
        custom<ExprOperand>($rhs, type($rhs))
        attr-dict
        custom<ExprResult>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Logical operators
//===----------------------------------------------------------------------===//

def EKL_LogicalNotOp : EKL_UnaryOperator<"lnot"> {
    let summary = "Logical negation operator";
}
def EKL_LogicalOrOp : EKL_BinaryOperator<"lor"> {
    let summary = "Logical disjunction operator";
}
def EKL_LogicalAndOp : EKL_BinaryOperator<"land"> {
    let summary = "Logical conjunction operator";
}

//===----------------------------------------------------------------------===//
// Arithmetic operators
//===----------------------------------------------------------------------===//

def EKL_NegateOp : EKL_UnaryOperator<"neg"> {
    let summary = "Unary minus operator";
}
def EKL_AddOp : EKL_BinaryOperator<"add"> {
    let summary = "Addition operator";
}
def EKL_SubtractOp : EKL_BinaryOperator<"sub"> {
    let summary = "Subtraction operator";
}
def EKL_MultiplyOp : EKL_BinaryOperator<"mul"> {
    let summary = "Multiplication operator";
}
def EKL_DivideOp : EKL_BinaryOperator<"div"> {
    let summary = "Division operator";
}
def EKL_RemainderOp : EKL_BinaryOperator<"rem"> {
    let summary = "Remainder operator";
}
def EKL_PowerOp : EKL_BinaryOperator<"pow"> {
    let summary = "Power operator";
}

//===----------------------------------------------------------------------===//
// Tensor operators
//===----------------------------------------------------------------------===//

def EKL_TensorProductOp : EKL_BinaryOperator<"tprod"> {
    let summary = "Tensor product operator";
}

#endif
