//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EKL dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef EKL_ATTRIBUTES
#define EKL_ATTRIBUTES

include "messner/Dialect/EKL/IR/Types.td"

//===----------------------------------------------------------------------===//
// Attribute classes
//===----------------------------------------------------------------------===//

def EKL_NumberAttr : EKL_Attr<"Number"> {
    let mnemonic = "number";
    let summary = "Numeric literal value attribute";
    let description = [{
        The `ekl.number` attribute holds the value of an EKL number literal. It
        models rational numbers using a (signed) integer mantissa and a signed
        exponent of base 2 with at most 64 bits.

        Examples:

        ```
        // These two name the same value:
        #ekl.number<1>
        #ekl.number<4, -2>

        // This is the value 3.141592653589793:
        #ekl.number<884279719003555, -48>
        ```
    }];

    let parameters = (ins "Number":$value);

    let builders = [
        AttrBuilder<(ins "Number":$value), [{
            // Ensure that the value is in its canonical, smallest form.
            value.shrinkToFit();
            return $_get($_ctxt, value);
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        `<` custom<Number>($value) `>`
    }];

    code extraClassDeclaration = [{
        /// Gets the NumberType singleton instance.
        [[nodiscard]] NumberType getType() const
        {
            return NumberType::get(getContext());
        }
    }];
}

def EKL_IndexAttr : EKL_Attr<"Index"> {
    let mnemonic = "index";
    let summary = "Index literal value attribute";
    let description = [{
        The `ekl.index` attribute holds a value of the IndexType.

        Example:

        ```
        // These two name the same value:
        #ekl.index<3>
        #ekl<_3>
        ```
    }];

    let parameters = (ins "extent_t":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];

    code extraClassDeclaration = [{
        using ValueType = extent_t;

        /// Gets the matching IndexType for the contained value.
        [[nodiscard]] ekl::IndexType getType() const
        {
            return ekl::IndexType::get(getContext(), getValue());
        }
    }];
}

def EKL_ArrayAttr : EKL_Attr<"Array", [TypedAttrInterface]> {
    let mnemonic = "array";
    let summary = "Comptime EKL array value";
    let description = [{
        The `ekl.array` attribute holds a value of an ArrayType.

        The storage of the `ekl.array` attribute is very inefficient (it uses
        `mlir::ArrayAttr`), but covariant. It also supports a splat
        representation, where only one value is stored, and repeated for the
        entire domain.

        Examples:

        ```
        #ekl.array<[1 : si32]> : !ekl.array<si32>
        #ekl.array<[0 : ui4, -1 : si2, 2.0 : f32, #ekl.index<2>]>
            : !ekl.array<!ekl.number [2, 2]>

        #ekl.array<[0.0 : f64]> : !ekl.array<f64 [100, 100]>
        ```
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"", "ArrayType">:$arrayType,
        "mlir::ArrayAttr":$flattened);

    let builders = [
        AttrBuilderWithInferredContext<(ins
                "ArrayType":$arrayType,
                "mlir::ArrayAttr":$flattened), [{
            assert(arrayType);
            return $_get(arrayType.getContext(), arrayType, flattened);
        }]>];

    let assemblyFormat = [{
        `<` $flattened `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Determines whether the value is stored as a splat.
        [[nodiscard]] bool isSplat() const
        {
            return getFlattened().size() == 1;
        }
        /// Obtains the stored splat value, if any.
        [[nodiscard]] Attribute getSplatValue() const
        {
            return isSplat() ? getFlattened().getValue().front() : Attribute{};
        }

        //===--------------------------------------------------------------===//
        // TypedAttrInterface
        //===--------------------------------------------------------------===//

        [[nodiscard]] ArrayType getType() const
        {
            return getArrayType();
        }
    }];
}

def EKL_InitializerAttr : EKL_Attr<"Initializer"> {
    let mnemonic = "init";
    let summary = "Initializer literal value attribute";
    let description = [{
        The `ekl.init` attribute holds the value of an initializable ABI array
        constant.

        Since the backing storage for the array elements is a DenseArrayAttr,
        the only allowed scalar types are the following:

            - Booleans
            - Integers up to 64 bits wide
            - Floating-point subtypes of f64

        Examples:

        ```
        #ekl.init<array<si32: 0>>
        #ekl.init<array<si32: 0, -1, 2, 3>, [2, 2]>
        ```
    }];

    let parameters = (ins
        "DenseArrayAttr":$flattened,
        OptionalArrayRefParameter<"extent_t">:$extents);

    let builders = [
        AttrBuilderWithInferredContext<(ins
                "DenseArrayAttr":$flattened,
                CArg<"ExtentRange", "{}">:$extents), [{
            assert(flattened);
            return $_get(flattened.getContext(), flattened, extents);
        }]>];

    let assemblyFormat = [{
        `<` $flattened ( `,` custom<Extents>($extents)^ )? `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Gets the matching ArrayType for the contained value.
        [[nodiscard]] ArrayType getType() const
        {
            return ArrayType::get(
                getFlattened().getElementType(),
                getExtents());
        }
    }];
}

def EKL_IdentityAttr : EKL_Attr<"Identity"> {
    let mnemonic = "id";
    let summary = "Identity literal attribute";
    let description = [{
        The `ekl.identity` attribute represents the `:` literal, which can be
        used in a subscript expression to produce an identity indexer.

        Example:

        ```
        #ekl.id
        #ekl<:>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::IdentityAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        /// Gets the IdentityType singleton instance.
        [[nodiscard]] IdentityType getType() const
        {
            return IdentityType::get(getContext());
        }
    }];
}

def EKL_ExtentAttr : EKL_Attr<"Extent"> {
    let mnemonic = "extent";
    let summary = "Extent insertion literal attribute";
    let description = [{
        The `ekl.extent` attribute represents the `*` literal, which can be used
        in a subscript expression to produce a new unit extent.

        Example:

        ```
        #ekl.extent
        #ekl<*>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::ExtentAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        // Gets the ExtentType singleton instance.
        [[nodiscard]] ExtentType getType() const
        {
            return ExtentType::get(getContext());
        }
    }];
}

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

def EKL_IntegerAttr : Attr<
        /*predicate=*/CPred<"llvm::isa<::mlir::ekl::IntegerAttr>($_self)">,
        /*summary=*/"Comptime EKL integer value"> {
    let description = [{
        Implements an attribute constraint on builtin integer attributes such
        that only signed or unsigned integers are accepted.

        Examples:

        ```
        -13 : si32
        4 : ui16
        ```
    }];

    let storageType = "::mlir::ekl::IntegerAttr";
    let returnType = "::llvm::APSInt";
}

def EKL_ScalarAttr : Attr<
        /*predicate=*/CPred<"llvm::isa<::mlir::ekl::ScalarAttr>($_self)">,
        /*summary=*/"Comptime EKL scalar value"> {
    let description = [{
        Implements an attribute constraint that accepts any of the valid EKL
        comptime scalar attributes:

            - NumberAttr
            - IntegerAttr
            - FloatAttr
            - IndexAttr
            - BoolAttr

        Examples:

        ```
        #ekl.number<9>
        #ekl.number<12, -4>
        -13 : si32
        4 : ui16
        1.0 : f32
        #ekl.index<0>
        true
        ```
    }];

    let storageType = "::mlir::ekl::ScalarAttr";
    let returnType = "::mlir::ekl::ScalarAttr";
    let convertFromStorage = "$_self";
}

def EKL_LiteralAttr : Attr<
        /*predicate=*/CPred<"llvm::isa<::mlir::ekl::LiteralAttr>($_self)">,
        /*summary=*/"Comptime EKL value"> {
    let description = [{
        Implements an attribute constraint that accepts any of the valid EKL
        literal attributes:

            - ScalarAttr
            - ArrayAttr
            - InitializerAttr
            - IdentityAttr
            - ExtentAttr
    }];

    let storageType = "::mlir::ekl::LiteralAttr";
    let returnType = "::mlir::ekl::LiteralAttr";
    let convertFromStorage = "$_self";
}

#endif
