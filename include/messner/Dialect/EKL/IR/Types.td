//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_TYPES
#define EKL_TYPES

include "messner/Dialect/EKL/IR/Base.td"
include "messner/Dialect/EKL/Interfaces/ContiguousType.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Type families
//===----------------------------------------------------------------------===//

def EKL_ExpressionType : EKL_Type<"Expression"> {
    let mnemonic = "expr";
    let summary = "Type of an EKL expression";
    let description = [{
        The `ekl.expr` type is the kind for all types that can result from EKL
        expressions. Its representation includes a type bound, which constrains
        the type of the expression.

        If no type bound is provided, the expression type is unbounded. All
        expression types are considered subtypes of the unbounded type. If a
        type bound is provided, it indicates that any value of this type must be
        a subtype of the type bound.

        Operations that accept or produce values of this type should implement
        the `ExpressionOp` interface to participate in type checking. During
        that process, the types of value definitions may be updated by the type
        checker, which goes against usual MLIR procedures. Note that consenting
        users may also accept a change to an unrelated type.

        Examples:

        ```
        !ekl.expr
        !ekl.expr<si32>
        ```
    }];

    let parameters = (ins
        OptionalParameter<
            "Type",
            "type bound of the expression, or nullptr">:$typeBound);

    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, Type{});
        }]>];

    let assemblyFormat = [{
        ( `<` $typeBound^ `>` )?
    }];

    code extraClassDeclaration = [{
        /// Determines whether this expression type is unbounded.
        [[nodiscard]] bool isUnbounded() const { return !getTypeBound(); }
    }];
}

def EKL_NumberType : EKL_Type<"Number"> {
    let mnemonic = "number";
    let summary = "Type of an EKL numeric value";
    let description = [{
        The `ekl.number` type is the generic unspecified numeral type of EKL. It
        is the type of a numeric literal, and results from expressions on these
        literals.

        Typically, this type should only be present at comptime. All values that
        are to be evaluated at runtime should be mapped to a concrete ABI type,
        such as a fixed-width integer or floating-point type.

        Examples:

        ```
        !ekl.number
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::NumberType::get($_builder.getContext())
    }];
}

def EKL_IndexType : EKL_Type<"Index"> {
    let mnemonic = "index";
    let summary = "Type of an EKL index value";
    let description = [{
        The `ekl.index` type is the type of an index expression with a known
        upper bound on the result value.

        Indices are assumed to be no wider than 64 bits, so the maximum upper
        bound value is 2^64 - 1, which is treated as the unbounded value.

        Examples:

        ```
        // These two name the same type:
        !ekl.index
        !ekl.index<18446744073709551617>

        // These two name the same type:
        !ekl<_4>
        !ekl.index<4>
        ```
    }];

    let parameters = (ins
        DefaultValuedParameter<
            "extent_t",
            "kUnbounded",
            "statically known upper bound">:$upperBound
    );

    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, kUnbounded);
        }]>];

    let assemblyFormat = [{
        ( `<` $upperBound^ `>` )?
    }];

    code extraClassDeclaration = [{
        /// The highest possible upper bound.
        static constexpr uint64_t kUnbounded = max_extent;

        /// Determines whether this index type is unbounded.
        [[nodiscard]] bool isUnbounded() const
        {
            return getUpperBound() == kUnbounded;
        }

        /// Gets the minimum number of bits needed to store the index value.
        [[nodiscard]] unsigned getBitWidth() const;

        /// Obtains the smallest unsigned IntegerType that stores the index.
        [[nodiscard]] IntegerType getIntegerType() const
        {
            return IntegerType::get(
                getContext(),
                getBitWidth(),
                IntegerType::SignednessSemantics::Unsigned);
        }
    }];
}

def EKL_StringType : EKL_Type<"String"> {
    let mnemonic = "string";
    let summary = "Type of an EKL string value";
    let description = [{
        The `ekl.string` type is the type of a string expression.

        Examples:

        ```
        !ekl.string
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
        The `ekl.array` type is the type of an indexed family with a regular
        index domain.

        The index domain of an array is part of a zero or more dimensional tuple
        space. It must be a hyperrectangle bound by the origin and an exclusive
        extents vector. It may not be empty, i.e., no extent can be zero.

        The element type of an array must be one of the allowed EKL scalar
        types. For convenience, a builder is available that automatically
        flattens nested array types.

        Examples:

        ```
        !ekl.array<f64>
        !ekl.array<f64[3,4]>
        ```
    }];

    let parameters = (ins
        "ScalarType":$scalarType,
        ArrayRefParameter<"extent_t">:$extents);

    let builders = [
        TypeBuilderWithInferredContext<(ins
                "ScalarType":$scalarType,
                CArg<"ExtentRange", "{}">:$extents), [{
            assert(scalarType);
            return $_get(scalarType.getContext(), scalarType, extents);
        }]>,
        TypeBuilderWithInferredContext<(ins
                "Type":$scalarOrArrayType,
                CArg<"ExtentRange", "{}">:$extents)>];

    let assemblyFormat = [{
        `<` $scalarType`` custom<Extents>($extents) `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains a copy with different @p extents .
        ///
        /// @pre    @p extents does not contain @c 0UL .
        [[nodiscard]] ArrayType cloneWith(ExtentRange extents) const
        {
            return get(getScalarType(), extents);
        }
        /// Obtains a copy with a different @p scalarType .
        ///
        /// @pre    `scalarType`
        [[nodiscard]] ArrayType cloneWith(ScalarType scalarType) const
        {
            return get(scalarType, getExtents());
        }
        /// Obtains a copy with different @p scalarOrArrayType .
        ///
        /// @pre    `llvm::isa_and_present<ScalarType, ArrayType>(scalarOrArrayType)`
        [[nodiscard]] ArrayType cloneWith(Type scalarOrArrayType) const;
    }];
}

def EKL_ReferenceType : EKL_Type<"Reference", [EKL_ContiguousType]> {
    let mnemonic = "ref";
    let summary = "Type of an EKL array reference";
    let description = [{
        The `ekl.ref` type represents the type of a value that stores a
        reference to a contiguous segment of memory (an `ekl.array`).

        Examples:

        ```
        // These two name the same type:
        !ekl.ref<si32>
        !ekl.ref<in si32>

        !ekl.ref<out f64[11,11,11]>
        ```
    }];

    let parameters = (ins
        EKL_ArrayType:$arrayType,
        DefaultValuedParameter<
            "ReferenceKind",
            "ReferenceKind::In",
            "reference directionality">:$kind);

    let builders = [
        TypeBuilderWithInferredContext<(ins
                "ArrayType":$arrayType,
                CArg<"ReferenceKind", "ReferenceKind::In">:$kind), [{
            assert(arrayType);
            return $_get(arrayType.getContext(), arrayType, kind);
        }]>];

    // NOTE: DefaultValuedParameter doesn't actually work in this case :(
    let assemblyFormat = [{
        `<` custom<ReferenceKind>($kind)`` custom<ArrayType>($arrayType) `>`
    }];

    code extraClassDeclaration = [{
        /// Determines whether the reference is readable.
        [[nodiscard]] bool isReadable() const
        {
            return ekl::isReadable(getKind());
        }
        /// Determines whether the reference is writable.
        [[nodiscard]] bool isWritable() const
        {
            return ekl::isWritable(getKind());
        }

        /// Obtains a copy with a different @p arrayType .
        ///
        /// @pre    `arrayType`
        [[nodiscard]] ReferenceType cloneWith(ArrayType arrayType) const
        {
            return get(arrayType, getKind());
        }
        /// Obtains a copy with a different @p kind .
        [[nodiscard]] ReferenceType cloneWith(ReferenceKind kind) const
        {
            return get(getArrayType(), kind);
        }

        //===--------------------------------------------------------------===//
        // ContiguousType
        //===--------------------------------------------------------------===//

        ScalarType getScalarType() const
        {
            return getArrayType().getScalarType();
        }
        ExtentRange getExtents() const
        {
            return getArrayType().getExtents();
        }
    }];
}

def EKL_IdentityType : EKL_Type<"Identity"> {
    let mnemonic = "id";
    let summary = "Type of the identity literal";
    let description = [{
        The `ekl.identity` type is the type of the `:` literal, which can be
        used in a subscript expression to produce an identity indexer.

        Example:

        ```
        !ekl.id
        !ekl<:>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::IdentityType::get($_builder.getContext());
    }];
}

def EKL_ExtentType : EKL_Type<"Extent"> {
    let mnemonic = "extent";
    let summary = "Type of the extent literal";
    let description = [{
        The `ekl.extent` type is the type of the `*` literal, which can be used
        in a subscript expression to produce a new unit extent.

        Example:

        ```
        !ekl.extent
        !ekl<*>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::ExtentType::get($_builder.getContext());
    }];
}

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

def EKL_IntegerType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::IntegerType>($_self)">,
        /*summary=*/"Type of an EKL integral value",
        /*cppClassName=*/"::mlir::ekl::IntegerType"> {
    let description = [{
        Implements a type constraint on builtin integer types such that only
        signed or unsigned integers are accepted.

        Examples:

        ```
        si32
        ui16
        ```
    }];
}

def EKL_BoolType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::BoolType>($_self)">,
        /*summary=*/"Type of an EKL boolean value",
        /*cppClassName=*/"::mlir::ekl::BoolType"> {
    let description = [{
        Implements a type constraint on builtin integer types such that only
        the `i1` type is accepted.

        Examples:

        ```
        i1
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::BoolType::get($_builder.getContext());
    }];
}

def EKL_ScalarType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::ScalarType>($_self)">,
        /*summary=*/"Type of an EKL scalar value",
        /*cppClassName=*/"::mlir::ekl::ScalarType"> {
    let description = [{
        Implements a type constraint that matches the allowed EKL scalar types:

            - NumberType
            - IntegerType
            - FloatType
            - IndexType
            - BoolType

        Examples:

        ```
        !ekl.number
        si32
        ui64
        f32
        !ekl.index
        !ekl.index<9>
        i1
        !ekl.string
        ```
    }];
}

def EKL_LiteralType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::LiteralType>($_self)">,
        /*summary=*/"Type of an EKL literal",
        /*cppClassName=*/"::mlir::ekl::LiteralType"> {
    let description = [{
        Implements a type constraint that matches any EKL literal type:

            - ScalarType
            - ArrayType
            - IdentityType
            - ExtentType
    }];
}

//===----------------------------------------------------------------------===//
// ABI type constraints
//===----------------------------------------------------------------------===//

def EKL_ABIScalarType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::ABIScalarType>($_self)">,
        /*summary=*/"Type of an ABI-compatible EKL scalar value",
        /*cppClassName=*/"::mlir::ekl::ABIScalarType"> {
    let description = [{
        Implements a type constraint on ScalarType that ensures the type has a
        defined ABI layout.

            - Any IntegerType wider than 0 bits
            - Any FloatType known to LLVM
            - Any BoolType

        Examples:

        ```
        si32
        f32
        i1
        ```
    }];
}

def EKL_ABIType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::ABIType>($_self)">,
        /*summary=*/"Type of an ABI-compatible EKL value",
        /*cppClassName=*/"::mlir::ekl::ABIType"> {
    let description = [{
        Implements a type constraint that matches any EKL type that has a
        defined ABI layout.

            - Any ABIScalarType
            - Any ArrayType with less than 2^32 - 1 elements per dimension and
              an underlying ABIScalarType
            - Any ReferenceType to an ABI-compatible array.

        Examples:

        ```
        ui9
        f64
        i1
        !ekl.array<f32>
        !ekl.array<si32[2, 2]>
        !ekl.ref<inout f32[11]>
        ```
    }];
}

def EKL_ABIReferenceType : Type<
        /*predicate=*/CPred<"::llvm::isa<::mlir::ekl::ABIReferenceType>($_self)">,
        /*summary=*/"Type of an ABI-compatible EKL reference value",
        /*cppClassName=*/"::mlir::ekl::ABIReferenceType"> {
    let description = [{
        Implements a type constraint that matches any EKL reference type that
        has an ABI-compatible target type.

        Examples:

        ```
        !ekl.ref<inout f32[11]>
        ```
    }];
}

#endif
