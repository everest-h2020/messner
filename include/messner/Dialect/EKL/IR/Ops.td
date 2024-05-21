//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_OPS
#define EKL_OPS

include "messner/Dialect/EKL/IR/Attributes.td"
include "messner/Dialect/EKL/IR/Traits.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Containers
//===----------------------------------------------------------------------===//
//
// Containers are operations that only serve structural purposes in the IR.

def EKL_ProgramOp : EKL_Op<
        "program",
        [
            // Foreign values are not allowed.
            IsolatedFromAbove,
            // Programs contain symbol declarations.
            SymbolTable,
            // Body does not have function semantics.
            SingleBlock,
            NoRegionArguments,
            NoTerminator,
            // Default dialect is `ekl`.
            OpAsmOpInterface]> {
    let summary = "Introduces an EKL translation unit";
    let description = [{
        The `ekl.program` operation is a container for an EKL translation unit,
        which will be compiled to a module.

        The following ops are allowed to occur inside a translation unit:

        - `ekl.global`
        - `ekl.kernel`

        Examples:

        ```
        ekl.program {
            // ...
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$bodyRegion);
    let builders = [
        OpBuilder<(ins), [{ $_state.addRegion()->emplaceBlock(); }]>];
    let skipDefaultBuilders = 1;

    let hasRegionVerifier = 1;

    let assemblyFormat = [{ attr-dict-with-keyword $bodyRegion }];

    code extraClassDeclaration = [{
        [[nodiscard]] Block *getBody() { return &getBodyRegion().front(); }

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect() {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Interoperability
//===----------------------------------------------------------------------===//
//
// Since EKL uses its own type system, we need to define a pair of cast-like
// operations that allow introducing and evaluating expressions.

class EKL_InteropBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [
            // May not touch memory.
            NoMemoryEffect,
            // Must have cast semantics.
            DeclareOpInterfaceMethods<CastOpInterface>] # traits> {
    // Must properly propagate constant values.
    let hasFolder = 1;
}

def EKL_IntroOp : EKL_InteropBase<
        "intro",
        [
            // Never has undefined behavior.
            AlwaysSpeculatable,
            // Result expression type is fixed by the operand type.
            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Introduces an expression from an MLIR value";
    let description = [{
        The `ekl.intro` operation introduces a new EKL expression by defining it
        to be a certain MLIR value. The type of the resulting expression is
        determined by the type of the input value.

        The `ekl.intro` operation is always well-formed and can never have any
        side-effects.

        Examples:

        ```
        %expr = ekl.intro %value : si32
        ```
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs EKL_ExpressionType:$expression);
    let builders = [
        OpBuilder<(ins "Value":$value), [{
            assert(value);

            $_state.addOperands({value});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), value.getType())});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{ $value `:` type($value) attr-dict }];
}

def EKL_EvalOp : EKL_InteropBase<
        "eval",
        [
            // As long as the expression type is unknown, there may be UB.
            DeclareOpInterfaceMethods<ConditionallySpeculatable>,
            // Must participate in type checking.
            DeclareOpInterfaceMethods<EKL_TypeCheckOpInterface>]> {
    let summary = "Evaluates an expression to an MLIR value";
    let description = [{
        The `ekl.eval` operation produces an MLIR value by evaluating an EKL
        expression. Since the type of the value must be known, this places a
        contraint on the input expression type.

        A program containing an `ekl.eval` operation that binds an expression of
        some deduced type to an unrelated type is ill-formed.

        Examples:

        ```
        %value = ekl.eval %expr -> si32
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$expression);
    let results = (outs AnyType:$value);
    let builders = [
        OpBuilder<(ins "Value":$expression, "Type":$resultType), [{
            $_state.addOperands({expression});
            $_state.addTypes({resultType});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($expression, type($expression))
        attr-dict
        `->` type($value)
    }];
}

//===----------------------------------------------------------------------===//
// Symbols
//===----------------------------------------------------------------------===//
//
// EKL translation units contain declarations for symbols with unique names.

class EKL_SymbolBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [EKL_SymbolOp] # traits> {
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility);

    let assemblyFormat = [{ ($sym_visibility)? $sym_name attr-dict }];
}

def EKL_StaticOp : EKL_SymbolBase<"static"> {
    let summary = "Introduces a static variable";
    let description = [{
        The `ekl.static` operation introduces a variable with static lifetime,
        which is accessible anywhere inside the surrounding `ekl.program`. The
        value of the associated symbol is an `!ekl.ref` that locates the
        variable in pinned memory.

        A static variable can have one of the following access modifiers:

        - `local` (default)

            * Defined inside the surrounding `ekl.program`.
            * Can't be accessed outside the surrounding `ekl.program`.
            * Requires an initializer if the reference is readable.
            * Will be removed if it is write-only.

        - `import`

            * Defined outside the surrounding `ekl.program`.
            * May be accessed outside the surrounding `ekl.program`.
            * Is assumed to be initialized.
            * May not have an initializer.
            * Will be removed if it is unused.

        - `export`

            * Defined inside the surrounding `ekl.program`.
            * May be accessed outside the surrounding `ekl.program`.
            * Requires an initializer.

        Any non-local static variable must have an ABI-compatible type, i.e.,
        it must be a reference to an array of an ABI-compatible scalar type.

        Examples:

        ```
        // Some zero-initialized local buffer.
        ekl.static local @buffer : !ekl.ref<inout si32[128]>
            = #ekl.init<array<i32: 0>, [128]>

        // Some imported read-only array.
        ekl.static import @G1 : !ekl.ref<in f32[30]>

        // Some exported 2x2 matrix that we never read from.
        ekl.static export @G2 : !ekl.ref<out f32[2, 2]>
            = #ekl.init<array<f32: 0.0f, 1.0f, -1.0f, 0.0f>, [2, 2]>
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<EKL_ReferenceType>:$type,
        UnitAttr:$isOwned,
        OptionalAttr<EKL_ArrayAttr>:$initializer,
        OptionalAttr<StrAttr>:$sym_visibility);
    let builders = [
        OpBuilder<(ins
                "StringRef":$name,
                "ReferenceType":$type,
                CArg<"AccessModifier", "AccessModifier::Local">:$access,
                CArg<"ArrayAttr", "{}">:$initializer)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<SymbolDecl>($sym_visibility, $isOwned, $sym_name)
        `:` $type ( `=` $initializer^ )?
        attr-dict
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;

    code extraClassDeclaration = [{
        [[nodiscard]] bool isOwned() { return getIsOwned(); }
        [[nodiscard]] bool isReadable() { return getType().isReadable(); }
        [[nodiscard]] bool isWritable() { return getType().isWritable(); }

        [[nodiscard]] AccessModifier getAccessModifier();

        [[nodiscard]] ArrayType getDataType()
        {
            return getType().getArrayType();
        }

        //===--------------------------------------------------------------===//
        // SymbolOpInterface
        //===--------------------------------------------------------------===//

        bool isDeclaration() { return !isOwned(); }
    }];
}

def EKL_KernelOp : EKL_SymbolBase<
        "kernel",
        [NoTerminator, EKL_HasFunctors, OpAsmOpInterface]> {
    let summary = "Introduces a kernel";
    let description = [{
        The `ekl.kernel` operation introduces a callable kernel. Since EKL does
        not support function pointers, the associated symbol has no value.

        A kernel can have one of the following access modifiers:

        - `public` (default)

            * May have callsites outside the surrounding `ekl.program`.
            * Must have ABI-compatible argument types.

        - `private`

            * Can only be called from inside the surrounding `ekl.program`.

        A kernel is a procedure that manipulates data in static variables or
        argument references. Kernels do not have return values.

        Examples:

        ```
        ekl.kernel @get_answer(%arg0: !ekl.ref<out f32>) {
            %lit = literal 42.0 : f32
            write %arg0: !ekl.ref<out f32>, %lit: f32
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$bodyRegion);
    let builders = [OpBuilder<(ins
            "StringRef":$name,
            CArg<"bool", "false">:$isPrivate)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<SymbolDecl>($sym_visibility, $sym_name)``
        custom<KernelBody>($bodyRegion, attr-dict)
    }];

    let hasVerifier = 1;

    code extraClassDeclaration = [{
        [[nodiscard]] Block *getBody() { return &getBodyRegion().front(); }

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//
//
// Statements are EKL operations that may have side-effects that can't be
// eliminated by dataflow means.

class EKL_StatementBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [EKL_StatementOp] # traits> {
}

def EKL_ReadOp : EKL_StatementBase<
        "read",
        [
            // Reads from the default memory resource.
            MemoryEffects<[MemRead]>,
            // Depending on the reference kind, may not be speculatable.
            DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
    let summary = "Reads the value behind a reference";
    let description = [{
        The `ekl.read` operation reads a value from a location indicated by a
        reference value.

        The reference must be readable. The reference kind can also indicate
        additional side-effects that happen during a read.

        Example:

        ```
        ekl.read %ref : !ekl.ref<in si32> -> si32
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$reference);
    let results = (outs EKL_ExpressionType:$result);
    let builders = [
        OpBuilder<(ins "Value":$reference, CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({reference});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($reference, type($reference))
        attr-dict
        custom<Result>(type($result))
    }];
}

def EKL_WriteOp : EKL_StatementBase<
        "write",
        [
            // Writes to the default memory resource.
            MemoryEffects<[MemWrite]>,
            // Depending on the reference kind, may be speculatable.
            DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
    let summary = "Writes the value behind a reference";
    let description = [{
        The `ekl.write` operation writes a value to a location indicated by a
        reference value.

        The reference must be writable. The reference kind can also exclude
        other side-effects that happen during a write.

        Example:

        ```
        ekl.write %ref : !ekl.ref<out si64>, %v : si32
        ```
    }];

    let arguments = (ins
        EKL_ExpressionType:$reference,
        EKL_ExpressionType:$value);
    let builders = [
        OpBuilder<(ins "Value":$reference, "Value":$value), [{
            $_state.addOperands({reference, value});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($reference, type($reference))```,`
        custom<Operand>($value, type($value))
        attr-dict
    }];
}

def EKL_IfOp : EKL_StatementBase<
        "if",
        [
            // This means its blocks _may_ not have a terminator.
            NoTerminator,
            EKL_HasFunctors,
            OpAsmOpInterface,
            DeclareOpInterfaceMethods<EKL_TypeCheckOpInterface>]> {
    let summary = "Diverges kernel execution";
    let description = [{
        The `ekl.if` operation allows for diverging execution in a kernel based
        on a condition. It has the following two representations:

        - The statement-if does not produce a result expression, and so it
            is not required to have an else branch, and it must not yield an
            expression from any of its branches.

        - The expression-if produces a result expression, and so it must
            have an else branch, and it must yield an expression from both of
            its branches. The yielded expressions must unify to the result
            type.

        Example:

        ```
        // Statement if without an else branch.
        ekl.if %c then {
            write %v1, %v2
        }

        // Expression if.
        %q = ekl.if %c then {
            yield %v1
        } else {
            yield %v2
        } -> !ekl.expr
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$condition);
    let regions = (region
        SizedRegion<1>:$thenRegion,
        MaxSizedRegion<1>:$elseRegion);
    let results = (outs Optional<EKL_ExpressionType>:$result);
    let builders = [
        OpBuilder<(ins
                "Value":$condition,
                CArg<"BranchBuilderRef", "{}">:$thenBranch,
                CArg<"BranchBuilderRef", "{}">:$elseBranch)>,
        OpBuilder<(ins
                "Value":$condition,
                "Type":$resultBound,
                CArg<"BranchBuilderRef", "{}">:$thenBranch,
                CArg<"BranchBuilderRef", "{}">:$elseBranch)>];
    let skipDefaultBuilders = 1;

    // NOTE: Although we do actually want to verify region contents, we need to
    //       ensure our verifier runs before the TypeCheckOpInterface verifier,
    //       which is set to verifyWithRegions.
    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<Operand>($condition, type($condition))
        attr-dict-with-keyword
        `then` custom<Functor>($thenRegion)
        custom<ElseBranch>($elseRegion, type($result))
    }];

    code extraClassDeclaration = [{
        [[nodiscard]] bool isStatement() { return getNumResults() == 0; }
        [[nodiscard]] bool isExpression() { return !isStatement(); }
        [[nodiscard]] Expression getExpression()
        {
            return llvm::cast_if_present<Expression>(getResult());
        }

        [[nodiscard]] Block* getThenBranch()
        {
            return &getThenRegion().front();
        }
        [[nodiscard]] Expression getThenExpression();

        [[nodiscard]] Block* getElseBranch()
        {
            if (getElseRegion().empty()) return nullptr;
            return &getElseRegion().front();
        }
        [[nodiscard]] Expression getElseExpression();

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//
//
// Expressions are operations that consume and produce expression values, and
// are subject to dataflow code motion and elimination.

class EKL_ExpressionBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [EKL_ExpressionOp] # traits> {
    let arguments = (ins Variadic<EKL_ExpressionType>:$operands);
    let results = (outs EKL_ExpressionType:$result);

    let builders = [
        OpBuilder<(ins
                "ValueRange":$operands,
                CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({operands});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operands>($operands, type($operands))
        attr-dict
        custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // ConditionallySpeculatable
        //===--------------------------------------------------------------===//

        Speculation::Speculatability getSpeculatability()
        {
            // Only speculatable if fully typed.
            return isFullyTyped() ? Speculation::Speculatable
                                  : Speculation::NotSpeculatable;
        }
    }];
}

//===----------------------------------------------------------------------===//
// Primary expressions
//===----------------------------------------------------------------------===//
//
// Primary expressions generate values but don't have bodies.

def EKL_LiteralOp : EKL_Op<"literal", [Pure, ConstantLike, OpAsmOpInterface]> {
    let summary = "Obtains a literal";
    let description = [{
        The `ekl.literal` operation introduces a literal expression.

        Example:

        ```
        %lit = ekl.literal -4 : si32
        %li2 = ekl.literal -4 : si32 -> si64
        ```
    }];

    let arguments = (ins EKL_LiteralAttr:$value);
    let results = (outs EKL_ExpressionType:$result);
    let builders = [
        OpBuilder<(ins "LiteralAttr":$value), [{
            build($_builder, $_state, value, value.getType());
        }]>,
        OpBuilder<(ins "LiteralAttr":$value, "Type":$resultBound), [{
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
            $_state.addAttribute(getValueAttrName($_state.name), value);
        }]>];
    let skipDefaultBuilders = 1;

    let hasVerifier = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        $value
        attr-dict
        custom<LiteralResult>(ref($value), type($result))
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

def EKL_GetStaticOp : EKL_ExpressionBase<
        "get_static",
        [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Obtains a static variable reference";
    let description = [{
        The `ekl.get_static` operation allows getting a reference value to a
        previously declared static variable.

        Example:

        ```
        ekl.get_static @G2 -> !ekl.ref<in f32[2, 2]>
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$target_name);
    let results = (outs EKL_ExpressionType:$result);
    let builders = [OpBuilder<(ins "StaticOp":$target)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $target_name attr-dict custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
            setNameFn(getResult(), getTargetName());
        }
    }];
}

def EKL_SubscriptOp : EKL_ExpressionBase<"subscript", [Pure]> {
    let summary = "Subscript operator";
    let description = [{
        The `ekl.subscript` operator derives a value from an input array using
        some tuple of indexers. The behavior of the subscript operation depends
        on the number and type of supplied subscript operands:

        - Full subscripting is when the number of subscript operands (after
            ellipsis expansion) matches the number of extents on the input
            array. The result will be a scalar of the input array.

        - Partial subscripting is when there are less subscript operands
            (after ellipsis expansion) that there are extents on the input
            array. The result will be an array of the same scalars as the
            input, but with fewer extents.

        The subscript operation is very versatile, and equivalent to an expanded
        version of Numpy basic indexing. It is best described by treating arrays
        as unctions associating indices to scalar values:

        - Scalar index values eliminate an extent from the result,
            effectively applying the index value.

        - Arrays of indices substitute an extent in the result, effectively
            composing with the indexer function.

        - The identitiy literal `:` preserves the extent in the result,
            effectively forwarding the index parameter.

        - The extent literal `*` adds a new unitary extent to the result,
            effectively declaring and not using an `!ekl<_0>` parameter.

        - The ellipsis literal `...` may only be used once and expands to
            as many `:` literals as needed to achieve full indexing.

        Example:

        ```
        // %v = ... -> !ekl.array<si32[8, 9, 10]>
        // %i = ... -> !ekl.array<!ekl<_7>[2, 3]>
        %i0 = literal #ekl<_0>
        %id = literal #ekl<:>
        %ex = literal #ekl<*>
        %el = literal #ekl<...>

        // Full indexing returns a single scalar.
        subscript %v [%i0: #ekl<_0>, %i0: #ekl<_0>, %i0: #ekl<_0>] -> si32

        // Partial indexing returns the remaining array.
        subscript %v [%i0: #ekl<_0>] -> !ekl.array<si32[9, 10]>

        // Compound indexing substitutes extents.
        subscript %v [%i, %i] -> !ekl.array<si32[2, 3, 2, 3, 10]>

        // Identity indexing preserves an extent.
        subscript %v [%id: !ekl<:>, %i0: !ekl<_0>] -> !ekl.array<si32[8, 10]>

        // Extent indexing adds a broadcast extent.
        subscript %v [%i0: !ekl<_0>, %ex: !ekl<*>] -> !ekl.array<si32[1, 9, 10]>

        // Ellipsis indexing inserts identities.
        subscript %v [%el: !ekl<*>, %i0: !ekl<_0>] -> !ekl.array<si32[8, 9]>
        ```
    }];

    let arguments = (ins
        EKL_ExpressionType:$array,
        Variadic<EKL_ExpressionType>:$subscripts);
    let builders = [
        OpBuilder<(ins
                "Value":$array,
                "ValueRange":$subscripts,
                CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({array});
            $_state.addOperands(subscripts);
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let hasFolder = 1;

    let assemblyFormat = [{
        custom<Operand>($array, type($array))
        ` ``[` custom<Operands>($subscripts, type($subscripts)) `]`
        attr-dict
        custom<Result>(type($result))
    }];
}

def EKL_StackOp : EKL_ExpressionBase<"stack", [Pure]> {
    let summary = "Stacking operator";
    let description = [{
        The `ekl.stack` operation produces an array by concatenating its
        operands ("stacking" them in a new outer extent). For this, the operands
        must broadcast and unify to a common array type.

        Example:

        ```
        ekl.stack %0: !ekl.array<si32[3]>, %1: si32 -> !ekl.array<si32[2, 3]>
        ```
    }];

    // NOTE: Unfortunately, minSize on Variadic can't be used, so this can't be
    //       an early executed ODS verifier.
    let hasVerifier = 1;
    let hasFolder = 1;

    let assemblyFormat = [{
        `[` custom<Operands>($operands, type($operands)) `]`
        attr-dict
        custom<Result>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Generator expressions
//===----------------------------------------------------------------------===//
//
// Generator expressions generate values using some functor(s). They are the
// most essential type of expression for code generation, and are targets of the
// EKL lowering patterns.

class EKL_GeneratorBase<string mnemonic, list<Trait> traits = []>
        : EKL_ExpressionBase<
            mnemonic,
            [EKL_GeneratorOp, OpAsmOpInterface] # traits> {
}

def EKL_YieldOp : EKL_Op<
        "yield",
        [
            Pure,
            Terminator,
            DeclareOpInterfaceMethods<EKL_TypeCheckOpInterface>]> {
    let summary = "Region terminator for scopes";
    let description = [{
        The `ekl.yield` operation terminates a functor of an EKL generator
        expression to produce a value.

        When an `ekl.yield` operation is invalidated during type checking, e.g,
        because the type of its operand was refined, it propagates this event to
        its parent. Care must be taken to avoid an infinite loop.

        Example:

        ```
        ekl.yield %0: si32
        ```
    }];

    let arguments = (ins EKL_ExpressionType:$expression);
    let builders = [
        OpBuilder<(ins "Value":$expression), [{
            $_state.addOperands({expression});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($expression, type($expression)) attr-dict
    }];
}

def EKL_AssocOp : EKL_GeneratorBase<
        "assoc",
        [
            DeclareOpInterfaceMethods<
                OpAsmOpInterface,
                ["getAsmBlockArgumentNames"]>]> {
    let summary = "Index association operator";
    let description = [{
        The `ekl.assoc` operation produces an array value by defining a map from
        the array index domain to the elements.

        This mapping is described by a functor region that accepts the index
        values and yields the element value. Its arguments must have
        `!ekl.index` types, and their bound determines the prefix of the result
        extents.

        The mapping functor is allowed to return scalars or arrays, with the
        latter being stacked in the last dimension, effectively concatenating
        their extents onto the declared extents.

        Contained `ekl.subscript` operations will eagerly propagate known index
        bounds to the bounds on the index domain. This means that it is not
        legal to reuse an index for dimensions of different sizes.

        Example:

        ```
        // %v : = ... -> !ekl.array<si32[4, 7, 9]>
        %c0 = ekl.literal #ekl<_0>

        ekl.assoc (%i) {
            %0 = subscript %v[%i, %c0: !ekl<_0>]
            yield %0
        } -> !ekl.array<si32[4, 9]>
        ```
    }];

    let arguments = (ins);
    let regions = (region SizedRegion<1>:$mapRegion);
    let builders = [
        OpBuilder<(ins
                "unsigned":$numExtents,
                CArg<"FunctorBuilderRef", "{}">:$map)>,
        OpBuilder<(ins
                "ExtentRange":$extents,
                CArg<"FunctorBuilderRef", "{}">:$map)>,
        OpBuilder<(ins
                CArg<"ArrayType", "{}">:$resultBound,
                CArg<"FunctorBuilderRef", "{}">:$map)>];
    let skipDefaultBuilders = 1;

    let hasFolder = 1;

    let assemblyFormat = [{
        custom<Functor>($mapRegion)
        attr-dict-with-keyword
        custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        [[nodiscard]] Block* getMap() { return &getMapRegion().front(); }
        [[nodiscard]] Expression getMapExpression();

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

def EKL_ZipOp : EKL_GeneratorBase<"zip"> {
    let summary = "Element mapping operator";
    let description = [{
        The `ekl.zip` operation produces a value by combining the elements of
        its operands. The operands must broadcast together to some common
        extents, which are the inferred result extents.

        The combinator is described by a functor region that accepts the scalar
        element values of the operands, and produces the result element.

        The combinator functor is allowed to return scalars or arrays, with the
        latter being stacked in the last dimension, effectively concatenating
        their extents onto the inferred result extents.

        If the operand list is empty, the combinator can return any value.

        Example:

        ```
        ekl.zip (%a, %b) using (%arg0, %arg1) {
            %0 = op_add %arg0, %arg1
            yield %0
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$combinatorRegion);
    let builders = [
        OpBuilder<(ins
                "ValueRange":$operands,
                CArg<"FunctorBuilderRef", "{}">:$combinator,
                CArg<"Type", "{}">:$resultBound)>];
    let skipDefaultBuilders = 1;

    let hasVerifier = 1;

    let assemblyFormat = [{
        `(` custom<Operands>($operands, type($operands)) `)`
        `using` custom<Functor>($combinatorRegion)
        attr-dict-with-keyword
        custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        [[nodiscard]] Block* getCombinator()
        {
            return &getCombinatorRegion().front();
        }
        [[nodiscard]] Expression getCombinatorExpression();

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

def EKL_ReduceOp : EKL_GeneratorBase<"reduce"> {
    let summary = "Array reduction operator";
    let description = [{
        The `ekl.reduce` operation produces a scalar value by folding all
        elements of the input array using a reduction.

        The reduction is described by a functor that takes the accumulator and
        the next element, and produces the next accumulator value. The result
        of the reduction is the last accumulator value.

        An optional initializer can be specified, e.g., the neutral element of
        the reduction. When this initializer is present, the accumulator type
        must accept it, and the reduction must must produce a compatible type.

        Example:

        ```
        ekl.reduce %i init %c0 using (%l, %r) {
            %1 = op_add %l, %r
            yield %1
        }
        ```
    }];

    let arguments = (ins
            EKL_ExpressionType:$array,
            Optional<EKL_ExpressionType>:$init);
    let regions = (region SizedRegion<1>:$reductionRegion);
    let builders = [
        OpBuilder<(ins
                "Value":$array,
                CArg<"FunctorBuilderRef", "{}">:$reduction,
                CArg<"Type", "{}">:$resultBound,
                CArg<"Value", "{}">:$init)>];
    let skipDefaultBuilders = 1;

    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<Operand>($array, type($array))
        ( `init` $init^`` custom<TypeBound>(type($init)) )?
        `using` custom<Functor>($reductionRegion)
        attr-dict-with-keyword
        custom<Result>(type($result))
    }];

    let extraClassDeclaration = [{
        [[nodiscard]] Block *getReduction()
        {
            return &getReductionRegion().front();
        }
        [[nodiscard]] Expression getReductionExpression();
        [[nodiscard]] Expression getInitExpression()
        {
            return llvm::cast_if_present<Expression>(getInit());
        }
        [[nodiscard]] Expression getAccumulatorExpression()
        {
            return llvm::cast<Expression>(getReduction()->getArgument(0));
        }
        [[nodiscard]] Expression getElementExpression()
        {
            return llvm::cast<Expression>(getReduction()->getArgument(1));
        }

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

def EKL_ConstexprOp : EKL_GeneratorBase<
        "constexpr",
        [IsolatedFromAbove, NoRegionArguments]> {
    let summary = "Constant expression scope";
    let description = [{
        The `ekl.constexpr` operation is used by the parser to create a scope
        in which constant expression evaluation can be performed. The expression
        is isolated in the nested region, to enable rewriting.

        Constant expression evaluation requires the body to be rewritable to
        a sequence of operations that folds to a literal. Additionally, the
        evaluation may not produce any side-effects.

        Example:

        ```
        ekl.constexpr {
            %0 = literal 16 : ui16
            %1 = literal -3449 : si32
            %2 = op_add %0: ui16, %1: si32
            yield %2
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs EKL_ExpressionType:$result);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let builders = [
        OpBuilder<(ins CArg<"Type", "{}">:$resultBound), [{
            $_state.addRegion()->emplaceBlock();
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let hasFolder = 1;

    let assemblyFormat = [{
        attr-dict-with-keyword $bodyRegion custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        [[nodiscard]] Block* getBody() { return &getBodyRegion().front(); }
        [[nodiscard]] Expression getExpression();

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static StringRef getDefaultDialect()
        {
            return EKLDialect::getDialectNamespace();
        }
    }];
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//
//
// Operators are expressions that combine other expressions without observing
// or changing memory.

class EKL_Operator<string name, list<Trait> traits = []>
        : EKL_ExpressionBase<name, [NoMemoryEffect] # traits> {
}

class EKL_UnaryOperator<string name, list<Trait> traits = []>
        : EKL_Operator<name, traits> {
    let arguments = (ins EKL_ExpressionType:$operand);
    let builders = [
        OpBuilder<(ins
                "Value":$operand,
                CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({operand});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand))
        attr-dict
        custom<Result>(type($result))
    }];
}

class EKL_BinaryOperator<string name, list<Trait> traits = []>
        : EKL_Operator<name, traits> {
    let arguments = (ins EKL_ExpressionType:$lhs, EKL_ExpressionType:$rhs);
    let builders = [
        OpBuilder<(ins
                "Value":$lhs,
                "Value":$rhs,
                CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({lhs, rhs});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($lhs, type($lhs))```,`
        custom<Operand>($rhs, type($rhs))
        attr-dict
        custom<Result>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//
//
// Casts are unary operators that change the type of an expression.

class EKL_CastBase<string name, list<Trait> traits = []>
        : EKL_UnaryOperator<name, [CastOpInterface] # traits> {
    let arguments = (ins EKL_ExpressionType:$operand);
    let results = (outs EKL_ExpressionType:$result);
    let builders = [
        OpBuilder<(ins
                "Value":$operand,
                "Type":$resultBound), [{
            $_state.addOperands({operand});
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>];
    let skipDefaultBuilders = 1;

    let hasFolder = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand))
        attr-dict
        custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // CastOpInterface
        //===--------------------------------------------------------------===//

        static bool areCastCompatible(TypeRange, TypeRange outputs)
        {
            // The cast target type must be known.
            return !!llvm::cast<ExpressionType>(outputs[0]).getTypeBound();
        }

        //===--------------------------------------------------------------===//
        // ConditionallySpeculatable
        //===--------------------------------------------------------------===//

        Speculation::Speculatability getSpeculatability()
        {
            // Only speculatable if fully typed.
            return isFullyTyped() ? Speculation::Speculatable
                                  : Speculation::NotSpeculatable;
        }
    }];
}

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

def EKL_BroadcastOp : EKL_CastBase<"broadcast"> {
    let summary = "Broadcasts a value";
    let description = [{
        The `ekl.bcast` operation broadcasts a value to an array.

        Example:

        ```
        ekl.bcast %a: !ekl.array<si32[4, 1]>, [4, 4] -> !ekl.array<si32[4, 4]>
        ```
    }];

    let arguments = (ins
        EKL_ExpressionType:$operand,
        DenseI64ArrayAttr:$signedExtents);
    let builders = [
        OpBuilder<(ins
                "Value":$operand,
                "DenseI64ArrayAttr":$extents,
                CArg<"Type", "{}">:$resultBound), [{
            $_state.addOperands({operand});
            $_state.addAttribute(getSignedExtentsAttrName($_state.name), extents);
            $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
        }]>,
        OpBuilder<(ins
                "Value":$operand,
                "ExtentRange":$extents,
                CArg<"Type", "{}">:$resultBound), [{
            build(
                $_builder,
                $_state,
                operand,
                getSignedExtentsAttr($_builder.getContext(), extents),
                resultBound);
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand))`` `,` $signedExtents
        attr-dict
        custom<Result>(type($result))
    }];

    code extraClassDeclaration = [{
        [[nodiscard]] static DenseI64ArrayAttr
        getSignedExtentsAttr(MLIRContext *context, ExtentRange extents);

        [[nodiscard]] ExtentRange getExtents();

        //===--------------------------------------------------------------===//
        // CastOpInterface
        //===--------------------------------------------------------------===//

        static bool areCastCompatible(TypeRange, TypeRange outputs)
        {
            // This is decided by the type checker.
            return true;
        }

        //===--------------------------------------------------------------===//
        // ConditionallySpeculatable
        //===--------------------------------------------------------------===//

        Speculation::Speculatability getSpeculatability()
        {
            // Only speculatable if fully typed.
            return isFullyTyped() ? Speculation::Speculatable
                                  : Speculation::NotSpeculatable;
        }
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

    // TODO: Implement folder.
    let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// Choice operator
//===----------------------------------------------------------------------===//

def EKL_ChoiceOp : EKL_Operator<"choice"> {
    let summary = "Choice operator";
    let description = [{
        The `ekl.choice` operator produces a value by choosing from a tuple of
        alternatives according to a selector.

        The selector must be a scalar or array of boolean or index types. The
        alternatives must broadcast and unify. The selector is extended with
        unit extents until it has at least as many extents as the alternatives,
        and then they are broadcast together to determine the result extents.

        The scalar type of the selector determines the minimum number of
        alternatives that must be provided, which is always at least 1.

        Example:

        ```
        // %c = ... -> i1
        // %i = ... -> !ekl.array<!ekl<_2>[4]>
        // %a = ... -> !ekl.array<si32[1, 9, 2]>
        // %b = ... -> !ekl.array<si64[1, 9, 2]>
        // %c = ... -> !ekl.array<si64[4, 9, 1]>

        // Boolean choice.
        ekl.choice %c of %a, %b -> !ekl.array<si64[1, 9, 2]>

        // Indexed choice.
        ekl.choice %i of %a, %b, %c -> !ekl.array<si64[4, 9, 2]>
        ```
    }];

    let arguments = (ins
        EKL_ExpressionType:$selector,
        Variadic<EKL_ExpressionType>:$alternatives);
    let builders = [
        OpBuilder<(ins
            "Value":$selector,
            "ValueRange":$alternatives,
            CArg<"Type", "{}">:$resultBound), [{
                $_state.addOperands({selector});
                $_state.addOperands(alternatives);
                $_state.addTypes({ExpressionType::get($_builder.getContext(), resultBound)});
            }]>];
    let skipDefaultBuilders = 1;

    // NOTE: Unfortunately, minSize on Variadic can't be used, so this can't be
    //       an early executed ODS verifier.
    let hasVerifier = 1;

    let assemblyFormat = [{
        custom<Operand>($selector, type($selector))
        `of` custom<Operands>($alternatives, type($alternatives))
        attr-dict
        custom<Result>(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Relational operators
//===----------------------------------------------------------------------===//

def EKL_CompareOp : EKL_BinaryOperator<"cmp", [EKL_RelationalOp]> {
    let summary = "Comparison operator";
    let arguments = (ins
        EKL_RelationKind:$kind,
        EKL_ExpressionType:$lhs,
        EKL_ExpressionType:$rhs);
    let builders = [
        OpBuilder<(ins
                "RelationKind":$kind,
                "Value":$lhs,
                "Value":$rhs,
                CArg<"Type", "{}">:$resultBound)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $kind
        custom<Operand>($lhs, type($lhs))```,`
        custom<Operand>($rhs, type($rhs))
        attr-dict
        custom<Result>(type($result))
    }];
}

def EKL_MinOp : EKL_BinaryOperator<"min", [EKL_RelationalOp]> {
    let summary = "Minimum operator";
}

def EKL_MaxOp : EKL_BinaryOperator<"max", [EKL_RelationalOp]> {
    let summary = "Maximum operator";
}

//===----------------------------------------------------------------------===//
// Logical operators
//===----------------------------------------------------------------------===//

def EKL_LogicalNotOp : EKL_UnaryOperator<"lnot", [EKL_LogicalOp]> {
    let summary = "Logical negation operator";
}
def EKL_LogicalOrOp : EKL_BinaryOperator<"lor", [EKL_LogicalOp]> {
    let summary = "Logical disjunction operator";
}
def EKL_LogicalAndOp : EKL_BinaryOperator<"land", [EKL_LogicalOp]> {
    let summary = "Logical conjunction operator";
}

//===----------------------------------------------------------------------===//
// Arithmetic operators
//===----------------------------------------------------------------------===//

def EKL_NegateOp : EKL_UnaryOperator<"neg", [EKL_ArithmeticOp]> {
    let summary = "Unary minus operator";
}
def EKL_AddOp : EKL_BinaryOperator<"add", [EKL_ArithmeticOp]> {
    let summary = "Addition operator";
}
def EKL_SubtractOp : EKL_BinaryOperator<"sub", [EKL_ArithmeticOp]> {
    let summary = "Subtraction operator";
}
def EKL_MultiplyOp : EKL_BinaryOperator<"mul", [EKL_ArithmeticOp]> {
    let summary = "Multiplication operator";
}
def EKL_DivideOp : EKL_BinaryOperator<"div", [EKL_ArithmeticOp]> {
    let summary = "Division operator";
}
def EKL_RemainderOp : EKL_BinaryOperator<"rem", [EKL_ArithmeticOp]> {
    let summary = "Remainder operator";
}
def EKL_PowerOp : EKL_BinaryOperator<"pow", [EKL_ArithmeticOp]> {
    let summary = "Power operator";
}

//===----------------------------------------------------------------------===//
// Tensor operators
//===----------------------------------------------------------------------===//

def EKL_TensorProductOp : EKL_BinaryOperator<"tprod"> {
    let summary = "Tensor product operator";
}

#endif
