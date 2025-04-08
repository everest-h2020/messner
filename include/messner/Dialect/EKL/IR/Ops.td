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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

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

        - `ekl.kernel`
        - `ekl.func`
        - `ekl.static`

        Examples:

        ```
        ekl.program {
            // ...
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$declarations);

    let builders = [OpBuilder<(ins), [{ $_state.addRegion() ; }]>];
    let skipDefaultBuilders = 1;

    let hasRegionVerifier = 1;

    let assemblyFormat = [{
        attr-dict-with-keyword $declarations
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static auto getDefaultDialect() -> StringRef { return EKLDialect::getDialectNamespace(); }
    }];
}

//===----------------------------------------------------------------------===//
// Symbols
//===----------------------------------------------------------------------===//
//
// Symbols are uniquely named declarations in EKL translation units.

class EKL_SymbolBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [EKL_Declaration, Symbol] # traits> {
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility);
}

def EKL_FuncOp : EKL_SymbolBase<
        "func",
        [
            SingleBlock,
            EKL_SemaOpInterface,
            DeclareOpInterfaceMethods<EKL_SemaOpInterface, ["checkSemantics"]>,
            FunctionOpInterface,
            OpAsmOpInterface]> {
    let summary = "Introduces a function";
    let description = [{
        The `ekl.func` operation introduces a callable symbol. Since EKL does
        not support function pointers, the associated symbol has no value.

        A function may accept any number of parameters and return any number of
        results. The ABI of functions in EKL is undefined, and thus functions
        must be definitions, with the exception of FFI functions.

        An FFI function is an external (declaration-only) function with an
        interface that only involves ABI types and has at most one result. In
        particular, all `ekl.kernel` symbols may be treated as FFI functions.
        These functions are guaranteed to follow the platform's C ABI.

        There are no return operations, all functions must be terminated by a
        single `ekl.yield` operation that returns the result values.

        Examples:

        ```
        // Defines a double identity function.
        ekl.func @my_id(%arg0: f64) -> f64 {
            yield %arg0: f64
        }

        // Import some functions from the CRT.
        ekl.func @sqrt(%arg0: f64) -> f64
        ekl.func @sqrtf(%arg0: f32) -> f32
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DefaultValuedAttr<StrAttr, "\"private\"">:$sym_visibility,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);
    let regions = (region MaxSizedRegion<1>:$definition);

    let builders = [
        OpBuilder<(ins "StringAttr":$name, "FunctionType":$type)>,
        OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasRegionVerifier = 1;

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===--------------------------------------------------------------===//

        auto getCallableRegion() -> Region * { return isExternal() ? nullptr : &getDefinition(); }
        auto getArgumentTypes() -> ArrayRef<Type> { return getFunctionType().getInputs(); }
        auto getResultTypes() -> ArrayRef<Type> { return getFunctionType().getResults(); }

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static auto getDefaultDialect() -> StringRef { return EKLDialect::getDialectNamespace(); }

        //===--------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===--------------------------------------------------------------===//

        auto isDeclaration() -> bool { return isExternal(); }
    }];
}

def EKL_KernelOp : EKL_SymbolBase<
        "kernel",
        [
            SingleBlock,
            NoTerminator,
            EKL_SemaOpInterface,
            DeclareOpInterfaceMethods<EKL_SemaOpInterface, ["checkSemantics"]>,
            // NOTE: KernelOp doesn't gain anything from implementing this.
            // FunctionOpInterface,
            OpAsmOpInterface]> {
    let summary = "Introduces a kernel";
    let description = [{
        The `ekl.kernel` operation introduces a kernel. Since EKL does not
        support function pointers, the associated symbol has no value.

        A kernel is a procedure that manipulates data in static variables or
        argument references. Kernels do not have return values.

        Examples:

        ```
        ekl.kernel @get_answer(%arg0: !ekl.ref<w, f64>) {
            %lit = literal 42.0 : f64
            write %arg0: !ekl.ref<w, f64>, %lit: f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<DictArrayAttr>:$arg_attrs);
    let regions = (region SizedRegion<1>:$definition);

    let builders = [
        OpBuilder<(ins "StringAttr":$name)>,
        OpBuilder<(ins "StringRef":$name)>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // Argument handling
        //===--------------------------------------------------------------===//

        auto getArguments() -> Block::BlockArgListType { return getBody()->getArguments(); }

        auto insertArgument(unsigned pos, ABIType type, Location loc, DictionaryAttr attrs = {}) -> BlockArgument;
        void eraseArgument(BlockArgument arg);

        //===--------------------------------------------------------------===//
        // Argument attribute handling
        //===--------------------------------------------------------------===//

        auto getAttrs(BlockArgument arg) -> DictionaryAttr::ValueType;
        void setAttrs(BlockArgument arg, DictionaryAttr attrs);
        auto getAttr(BlockArgument arg, StringRef name) -> Attribute;
        template<messner::attr_constraint T>
        auto getAttrOfType(BlockArgument arg, StringRef name) -> T
        {
            return llvm::dyn_cast_if_present<T>(getAttr(arg, name));
        }
        void setAttr(BlockArgument arg, NamedAttribute attr);
        auto removeAttr(BlockArgument arg, StringRef name) -> Attribute;

        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        static auto getDefaultDialect() -> StringRef { return EKLDialect::getDialectNamespace(); }

        //===--------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===--------------------------------------------------------------===//

        auto isDeclaration() -> bool { return false; }
    }];
}

def EKL_StaticOp : EKL_SymbolBase<
        "static",
        [
            EKL_SemaOpInterface,
            DeclareOpInterfaceMethods<EKL_SemaOpInterface, ["checkSemantics"]>]> {
    let summary = "Introduces a static variable";
    let description = [{
        The `ekl.static` operation introduces a variable with static lifetime,
        which is accessible anywhere inside the surrounding `ekl.program`. The
        value of the associated symbol is an `!ekl.ref` that locates the
        variable in memory.

        Examples:

        ```
        // Defines a zero-initialized local buffer.
        ekl.static local @buffer : !ekl.ref<unique, si32[128]>
            = #ekl.init<array<i32: 0>, [128]>

        // Imports a read-only array.
        ekl.static import @G1 : !ekl.ref<ro, f32[30]>

        // Exports a 2x2 matrix that EKL will never read from.
        ekl.static export @G2 : !ekl.ref<iw, f32[2, 2]>
            = #ekl.init<array<f32: 0.0f, 1.0f, -1.0f, 0.0f>, [2, 2]>
        ```
    }];

    let arguments = (ins
        OptionalAttr<StrAttr>:$sym_visibility,
        SymbolNameAttr:$sym_name,
        TypeAttrOf<EKL_ReferenceType>:$type,
        OptionalAttr<EKL_ArrayAttr>:$initializer);

    let builders = [
        OpBuilder<(ins
            "StringRef":$name,
            "ReferenceType":$type,
            CArg<"ArrayAttr", "{}">:$initializer,
            CArg<"bool", "false">:$isExported)>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
        auto isLocal() -> bool { return !this->isPublic() && !isDeclaration(); }
        auto isImported() -> bool { return !this->isPublic() && isDeclaration(); }
        auto isExported() -> bool { return this->isPublic(); }

        //===--------------------------------------------------------------===//
        // SymbolOpInterface
        //===--------------------------------------------------------------===//

        auto isDeclaration() -> bool { return !getInitializer().has_value(); }
    }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//
//
// Statements are operations that imply control flow and may have side-effecting
// behavior.

// IfOp
// ReadOp
// WriteOp
// ApplyOp

//===----------------------------------------------------------------------===//
// Primary expressions
//===----------------------------------------------------------------------===//
//
// Primary expressions are expressions that aren't subject to type checking.

def EKL_LiteralOp : EKL_Op<
        "literal",
        [
            Pure,
            ConstantLike,
            OpAsmOpInterface,
            AllTypesMatch<["value", "result"]>]> {
    let summary = "Obtains a literal";
    let description = [{
        The `ekl.literal` operation introduces a literal expression.

        Example:

        ```
        %lit = ekl.literal -4 : si32
        ```
    }];

    let arguments = (ins EKL_LiteralAttr:$value);
    let results = (outs AnyType:$result);
    let builders = [OpBuilder<(ins "LiteralAttr":$value)>];
    let skipDefaultBuilders = 1;

    let hasFolder = 1;

    let assemblyFormat = [{
        $value attr-dict
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) { setNameFn(getResult(), "lit"); }
    }];
}

def EKL_GetStaticOp : EKL_Op<
        "get_static",
        [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Obtains a static variable reference";
    let description = [{
        The `ekl.get_static` operation allows getting a reference value to a
        previously declared static variable.

        Example:

        ```
        ekl.get_static @G2 : !ekl.ref<in f32[2, 2]>
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$target_name);
    let results = (outs EKL_ReferenceType:$result);
    let builders = [OpBuilder<(ins "StaticOp":$target)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        $target_name attr-dict `:` type($result)
    }];

    code extraClassDeclaration = [{
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(OpAsmSetValueNameFn setNameFn) { setNameFn(getResult(), getTargetName()); }
    }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//
//
// Expressions are operations that consume and produce expression values, and
// are subject to dataflow code motion and elimination.

class EKL_ExpressionBase<string mnemonic, list<Trait> traits = []>
        : EKL_Op<mnemonic, [EKL_Expression] # traits> {
    let arguments = (ins Variadic<AnyType>:$operands);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        custom<OperandList>($operands, type($operands))
        attr-dict
        custom<ResultType>(type($result))
    }];
}

// SubscriptOp
// StackOp
// ChooseOp

//===----------------------------------------------------------------------===//
// Generators
//===----------------------------------------------------------------------===//
//
// Generators are expressions that produce SSA values using functors.

// MapOp
// ZipOp
// FoldOp
// LambdaOp
// ConstevalOp

def EKL_LambdaOp : EKL_Op<"lambda", [Pure]> {
    let summary = "Anonymous EKL expression abstraction";
    let description = [{
        The `ekl.lambda` operation serves as a value-over-values abstraction
        that creates an anonymous function.


    }];
}

def EKL_YieldOp : EKL_Op<
        "yield",
        [
            Pure,
            Terminator,
            EKL_SemaOpInterface,
            DeclareOpInterfaceMethods<TypeCheckOpInterface, ["typeCheck"]>]> {
    let summary = "Terminator for an EKL functor";
    let description = [{
        The `ekl.yield` operation terminates a functor of an EKL generator
        expression to produce a value.

        When an `ekl.yield` operation is invalidated during type checking, e.g,
        because the type of its operand was refined, it propagates this event to
        its parent.

        Example:

        ```
        ekl.yield %0: si32
        ```
    }];

    let arguments = (ins AnyType:$operand);

    let builders = [OpBuilder<(ins "Value":$operand)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand)) attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

class EKL_CastBase<string name, list<Trait> traits = []> : EKL_Op<
        name,
        [
            EKL_Cast,
            DeclareOpInterfaceMethods<TypeCheckOpInterface, ["typeCheck"]>]
            # traits> {
    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let hasFolder = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand))
        attr-dict
        `->` type($result)
    }];
}

def EKL_PromoteOp : EKL_CastBase<"promote"> {
    let summary = "Promotes a value to a supertype";
    let description = [{
        The `ekl.promote` operation is a value-preserving promoting cast.

        Example:

        ```
        ekl.promote %a: si32 -> si64
        ```
    }];
}

def EKL_BroadcastOp : EKL_CastBase<"bcast"> {
    let summary = "Broadcasts a value to a different shape";
    let description = [{
        The `ekl.bcast` operation broadcasts a value to a specified shape, which
        always results in an array.

        Example:

        ```
        ekl.bcast %a: !ekl.array<si32[4, 1]> to [4, 4] -> !ekl.array<si32[4, 4]>
        ```
    }];

    let arguments = (ins
        AnyType:$operand,
        DenseI64ArrayAttr:$signedExtents);

    let builders = [
        OpBuilder<(ins "ArrayType":$resultType, "Value":$operand)>,
        OpBuilder<(ins "ShapeRef":$resultShape, "Value":$operand)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        custom<Operand>($operand, type($operand))`` `to` $signedExtents
        attr-dict
        custom<ResultType>(type($result))
    }];

    code extraClassDeclaration = [{
        /// Obtains a DenseI64ArrayAttr for @p shape .
        ///
        /// @pre    `context`
        static auto getSignedExtentsAttr(MLIRContext *context, ShapeRef shape) -> DenseI64ArrayAttr;

        /// Gets the result shape.
        auto getResultShape() -> ShapeRef;
    }];
}

def EKL_CoerceOp : EKL_CastBase<"coerce"> {
    let summary = "Coerces a value to a different type";
    let description = [{
        The `ekl.coerce` operation coerces a value to a different type, which
        may not preserve the value.

        Example:

        ```
        ekl.coerce %a: si64 -> si32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

// CompareOp
// MinOp
// MaxOp

// LogicalNotOp
// LogicalOrOp
// LogicalAndOp

// AddOp
// SubtractOp
// MultiplyOp
// DivideOp
// RemainderOp
// PowerOp

#endif // EKL_OPS
