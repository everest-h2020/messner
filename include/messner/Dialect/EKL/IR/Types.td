//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_TYPES
#define EKL_TYPES

include "messner/Dialect/EKL/IR/Dialect.td"
include "messner/Dialect/EKL/Interfaces/ContiguousType.td"
include "messner/Dialect/Ref/Interfaces/ReferenceTypeInterface.td"

include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Type families
//===----------------------------------------------------------------------===//

def EKL_ExpressionType : EKL_Type<"Expression"> {
    let mnemonic = "expr";
    let summary = "Type of an EKL expression";
    let description = [{
        The `ekl.expr` type is the universal type that results from any EKL
        expression. It is used in the syntactic form of an EKL program, where it
        is the type of an AST node.

        The presence of this type in the program means that it has not yet been
        concretely typed and may thus be semantically invalid. This is usually
        resolved via type checking using the `type-check` pass.

        Examples:

        ```
        // These two name the same type:
        !ekl.expr
        !ekl<?>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::ExpressionType::get($_builder.getContext())
    }];
}

def EKL_RationalType : EKL_Type<"Rational"> {
    let mnemonic = "Q";
    let summary = "Type of an EKL rational value";
    let description = [{
        The `ekl.Q` type is the multiprecision rational number type.

        Typically, this type should only be present at comptime. All values that
        are to be evaluated at runtime should be mapped to a concrete ABI type,
        such as a fixed-width integer or floating-point type.

        Examples:

        ```
        !ekl.Q
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::RationalType::get($_builder.getContext())
    }];
}

def EKL_IndexType : EKL_Type<"Index"> {
    let mnemonic = "I";
    let summary = "Type of an EKL index value";
    let description = [{
        The `ekl.I` type is the sequence indexer type.

        Indices can be relative to the start or end of a sequence, and may have
        a statically known upper bound on their distance, i.e., a lower bound on
        the length of a sequence they are in-bounds for.

        Examples:

        ```
        // These two name the same type:
        !ekl.I
        !ekl.I<?>

        !ekl.I<1>
        !ekl.I<-?>
        !ekl.I<-1>
        ```
    }];

    let parameters = (ins
        DefaultValuedParameter<
            "Extent",
            "unbounded",
            "statically known bound">:$bound,
        DefaultValuedParameter<
            "bool",
            "false",
            "relative to sequence end">:$isFromEnd
    );

    let builders = [TypeBuilder<(ins "Index":$value)>];

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
        /// Determines whether the values have a bounded offset.
        auto isBounded() const -> bool { return getBound().isBounded(); }
        /// Determines whether the values relative to the end.
        auto isFromEnd() const -> bool { return getIsFromEnd(); }
    }];
}

def EKL_StringType : EKL_Type<"String"> {
    let mnemonic = "str";
    let summary = "Type of an EKL string value";
    let description = [{
        The `ekl.string` type is the character sequence type.

        Examples:

        ```
        !ekl.str
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::StringType::get($_builder.getContext());
    }];
}

def EKL_ArrayType : EKL_Type<"Array", [EKL_ContiguousType]> {
    let mnemonic = "array";
    let summary = "Type of an EKL array value";
    let description = [{
        The `ekl.array` type is the type of an indexed family with a static
        index domain.

        The index domain of an array is part of a zero or more dimensional tuple
        space. It must be a hyperrectangle bound by the origin and an exclusive
        extents vector. It may not be empty, i.e., no extent can be zero.

        The element type of an array must be one of the allowed EKL scalar
        types. For convenience, a builder is available that automatically
        flattens nested array types.

        Examples:

        ```
        // These two name the same type:
        !ekl.array<f64>
        !ekl.array<f64[]>

        !ekl.array<f64[3,4]>
        ```
    }];

    let parameters = (ins
        "ScalarType":$scalarType,
        ArrayRefParameter<"Extent">:$shape);

    let assemblyFormat = [{
        `<` $scalarType`` custom<StaticShape>($shape) `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains an ArrayType with @p scalarTy and @p shape.
        ///
        /// @pre    `scalarTy`
        /// @pre    `isValid(shape)`
        static auto get(ScalarType scalarTy, ShapeRef shape = {}) -> ArrayType;
        /// Obtains an ArrayType with outer @p shape and inner
        /// @p scalarOrArrayTy.
        ///
        /// @pre    `scalarOrArrayTy`
        /// @pre    `llvm::isa<ScalarType, ArrayType>(scalarOrArrayTy)`
        /// @pre    `isValid(shape)`
        static auto get(Type scalarOrArrayTy, ShapeRef shape = {}) -> ArrayType;

        /// Determines if @p shape is a valid array shape.
        ///
        /// All extents in @p shape must be bounded and non-zero for the shape
        /// to be valid. In addition, the flattened shape extents (i.e., the
        /// total number of elements) may not exceed Extents::value_max.
        static auto isValid(ShapeRef shape) -> bool;

        /// Obtains an ArrayType with @p scalarOrArrayTy as its inner type.
        ///
        /// If @p scalarOrArrayTy is a ScalarType, this method does the same as
        /// cloneWith(ScalarType). If @p scalarOrArrayTy is another ArrayType ,
        /// the extents are appended and the scalar type is replaced.
        ///
        /// @pre    `scalarOrArrayTy`
        auto cloneWith(Type scalarOrArrayTy) const -> ArrayType;
        /// Obtains an ArrayType with @p scalarTy and the same shape.
        ///
        /// @pre    `scalarTy`
        auto cloneWith(ScalarType scalarTy) const -> ArrayType;
        /// Obtains an ArrayType with @p shape and the same scalar type.
        ///
        /// @pre    `isValid(shape)`
        auto cloneWith(ShapeRef shape) const -> ArrayType;
    }];
}

def EKL_ReferenceType : EKL_Type<
        "Reference",
        [EKL_ContiguousType, Ref_ReferenceTypeInterface]> {
    let mnemonic = "ref";
    let summary = "Type of an EKL array reference";
    let description = [{
        The `ekl.ref` type is the type of a reference to a contiguous array, or
        the first element in a contiguous range of such arrays.

        Examples:

        ```
        // These two name the same type:
        !ekl.ref<ro, f64>
        !ekl.ref<ro, f64[]>

        !ekl.ref<ro, f64[...]>
        !ekl.ref<ro, f64[3,4]>
        !ekl.ref<ro, f64[...,3,4]>
        ```
    }];

    let parameters = (ins
        "ref::ReferenceKind":$kind,
        "ScalarType":$scalarType,
        ArrayRefParameter<"Extent">:$shape);

    let assemblyFormat = [{
        `<` $kind `,` $scalarType`` custom<ReferenceShape>($shape) `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains a ReferenceType for @p scalarTy with @p shape .
        ///
        /// @pre    `scalarTy`
        /// @pre    `isValid(shape)`
        static auto get(ref::ReferenceKind kind, ScalarType scalarTy, ShapeRef shape = {}) -> ReferenceType;
        /// Obtains a ReferenceType for @p arrayTy that @p isScalable .
        ///
        /// @pre    `arrayTy`
        static auto get(ref::ReferenceKind kind, ArrayType arrayTy, bool isScalable = false) -> ReferenceType;

        /// Determines if @p shape is a valid reference shape.
        ///
        /// The leading extent may be unbounded, but the remaining extents must
        /// be a valid array shape.
        static auto isValid(ShapeRef shape) -> bool;

        /// Determines whether this reference has a leading unbounded extent.
        auto isScalable() const -> bool;

        /// Obtains a ReferenceType with @p kind and the same scalar type and shape.
        auto cloneWith(ref::ReferenceKind kind) const -> ReferenceType;
        /// Obtains a ReferenceType with @p arrayTy and the same kind.
        ///
        /// @pre    `arrayTy`
        auto cloneWith(ArrayType arrayTy) const -> ReferenceType;

        //===--------------------------------------------------------------===//
        // ReferenceTypeInterface
        //===--------------------------------------------------------------===//

        /// Gets the underlying array type.
        auto getCellType() const -> ArrayType;
    }];
}

def EKL_SliceType : EKL_Type<"Slice"> {
    let mnemonic = "slice";
    let summary = "Type of an EKL slice indexer";
    let description = [{
        The `ekl.slice` type is the

        Example:

        ```
        // These three name the same type:
        !ekl.slice
        !ekl.slice<:>
        !ekl.slice<::1>

        !ekl.slice<0:>
        !ekl.slice<:end-1>

        // These two name the same type:
        !ekl.slice<0:end-1>
        !ekl.slice<0:end-1:1>

        !ekl.slice<1:>
        !ekl.slice<end-2:>
        !ekl.slice<:end-2>
        !ekl.slice<:1>
        !ekl.slice<0::2>
        ```
    }];

    let parameters = (ins
        OptionalParameter<"std::optional<Index>", "optional begin index">:$begin,
        OptionalParameter<"std::optional<Index>", "optional end index">:$end,
        DefaultValuedParameter<"Offset", "1", "non-zero stride">:$stride);

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains the SliceType of @p value .
        ///
        /// @pre    `context`
        static auto get(MLIRContext *context, const Slice &value) -> SliceType;

        /// Obtains the Slice that inhabits this type, if any.
        auto tryGetValue() const -> std::optional<Slice>;
    }];
}

def EKL_AxisType : EKL_Type<"Axis"> {
    let mnemonic = "axis";
    let summary = "Type of the axis literal";
    let description = [{
        The `ekl.axis` type is the type of the `*` literal, which introduces a
        new unit extent in a subscript expression.

        Example:

        ```
        // These two name the same type:
        !ekl.axis
        !ekl<*>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::EllipsisType::get($_builder.getContext());
    }];
}

def EKL_EllipsisType : EKL_Type<"Ellipsis"> {
    let mnemonic = "ellipsis";
    let summary = "Type of the ellipsis literal";
    let description = [{
        The `ekl.ellipsis` type is the type of the `...` literal, which expands
        to indentity slices in a subscript expression.

        Example:

        ```
        // These two name the same type:
        !ekl.ellipsis
        !ekl<...>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::EllipsisType::get($_builder.getContext());
    }];
}

#endif // EKL_TYPES
