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

def EKL_RationalAttr : EKL_Attr<"Rational"> {
    let mnemonic = "Q";
    let summary = "Rational number literal attribute";
    let description = [{
        The `ekl.Q` attribute holds the value of the RationalType. It models
        rational numbers using a (signed) integer mantissa and a signed exponent
        of base 2 with at most 64 bits.

        Examples:

        ```
        // These two name the same value:
        #ekl.Q<1>
        #ekl.Q<"4p-2">

        // These two name the value 3.141592653589793:
        #ekl.Q<"3.141592653589793">
        #ekl.Q<"884279719003555p-48">
        ```
    }];

    let parameters = (ins "Rational":$value);

    let builders = [AttrBuilder<(ins "Rational":$value)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        `<` custom<Rational>($value) `>`
    }];

    code extraClassDeclaration = [{
        /// Gets the RationalType singleton instance.
        auto getType() const -> RationalType;
    }];
}

def EKL_IndexAttr : EKL_Attr<"Index"> {
    let mnemonic = "I";
    let summary = "Index literal attribute";
    let description = [{
        The `ekl.index` attribute holds a value of the IndexType.

        Example:

        ```
        #ekl.I<3>
        #ekl.I<end-3>
        ```
    }];

    let parameters = (ins "Index":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];

    code extraClassDeclaration = [{
        using ValueType = Index;

        /// Gets the corresponding IndexType.
        auto getType() const -> IndexType;
    }];
}

def EKL_ArrayAttr : EKL_Attr<"Array", [TypedAttrInterface]> {
    let mnemonic = "array";
    let summary = "Array literal attribute";
    let description = [{
        The `ekl.array` attribute holds a value of an ArrayType.

        The internal representation of this attribute uses an array of stack
        expression literals, which are also implicitly broadcasting and
        covariant. In addition, a splat representation is possible, where a
        single-valued array can be stored in a shallow form.

        Examples:

        ```
        // A scalar-valued array attribute.
        #a1 = #ekl.array<[1 : si32]> : !ekl.array<si32>

        // A covariant splat array attribute.
        #a2 = #ekl.array<[0 : si32]> : !ekl.array<si64[3, 1024]>

        // A stacked array attribute with splat elements.
        #a3 = #ekl.array<[1 : si32, 2 : si32, 3 : si32]> : !ekl.array<si64[3, 1]>

        // A stacked array attribute with broadcasted elements.
        #a4 = #ekl.array<[#a1, #a2, #a3]> : !ekl.array<si64[3, 3, 1024]>
        ```
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"", "ArrayType">:$arrayType,
        "TupleAttr":$stack);

    let builders = [
        AttrBuilderWithInferredContext<(ins
            "ArrayType":$arrayType,
            "TupleAttr":$stack)>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{
        `<` $stack `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Obtains an ArrayAttr for @p arrayTy from @p stack .
        ///
        /// @pre    `arrayTy`
        /// @pre    @p stack is valid for @p arrayTy
        static auto get(ArrayType arrayTy, TupleAttr::ValueType stack) -> ArrayAttr;
        /// Obtains an ArrayAttr for @p arrayTy from @p splatValue .
        ///
        /// @pre    `arrayTy`
        /// @pre    `splatValue`
        /// @pre    @p splatValue is valid for @p arrayTy
        static auto get(ArrayType arrayTy, ScalarAttr splatValue) -> ArrayAttr;
        /// Obtains an ArrayAttr from @p splatValue with @p shape .
        ///
        /// @pre    `splatValue`
        /// @pre    `ArrayType::isValid(shape)`
        static auto get(ScalarAttr splatValue, ShapeRef shape = {}) -> ArrayAttr;
        /// Obtains an ArrayAttr from @p bcastValue and outer @p extent .
        ///
        /// @pre    `bcastValue`
        /// @pre    `ArrayType::isValid(concat(extent, bcastValue.getType().getShape()))`
        static auto get(ArrayAttr bcastValue, Extent extent) -> ArrayAttr;

        /// Determines whether the value is stored as a broadcast.
        auto isBroadcast() const -> bool;
        /// Obtains the stored broadcast value, if any.
        auto getBroadcastValue() const -> BroadcastAttr;

        /// Determines whether the value is stored as a splat.
        auto isSplat() const -> bool;
        /// Obtains the stored splat value, if any.
        auto getSplatValue() const -> ScalarAttr;

        //===--------------------------------------------------------------===//
        // TypedAttrInterface
        //===--------------------------------------------------------------===//

        /// Gets the corresponding ArrayType.
        auto getType() const -> ArrayType { return getArrayType(); }
    }];
}

def EKL_SliceAttr : EKL_Attr<"Slice"> {
    let mnemonic = "slice";
    let summary = "Slice literal attribute";
    let description = [{
        The `ekl.slice` attribute holds a value of a SliceType.

        Example:

        ```
        // These all name the identity slice.
        #ekl<:>
        #ekl.slice<:>
        #ekl.slice<0:>
        #ekl.slice<:end>
        #ekl.slice<0:end>
        #ekl.slice<0:end:1>
        #ekl.slice<-1:end+1>

        // Get the first element.
        #ekl.slice<:1>
        // Get the last element.
        #ekl.slice<end-1:>
        // End at the second-to-last element.
        #ekl.slice<:end-1>
        // Start at the second element.
        #ekl.slice<1:>

        // These two get every odd element:
        #ekl.slice<0::2>
        #ekl.slice<::2>
        ```
    }];

    let parameters = (ins "Slice":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Gets the corresponding SliceType.
        auto getType() const -> SliceType;
    }];
}

def EKL_AxisAttr : EKL_Attr<"Axis"> {
    let mnemonic = "axis";
    let summary = "Extent insertion literal attribute";
    let description = [{
        The `ekl.axis` attribute represents the `*` literal, which introduces a
        new unit extent in a subscript expression.

        Example:

        ```
        // These two name the same attribute:
        #ekl.axis
        #ekl<*>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::ExtentAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        /// Gets the ExtentType singleton instance.
        auto getType() const -> AxisType;
    }];
}

def EKL_EllipsisAttr : EKL_Attr<"Ellipsis"> {
    let mnemonic = "ellipsis";
    let summary = "Ellipsis literal attribute";
    let description = [{
        The `ekl.ellipsis` attribute represents the `...` literal, which expands
        to identity slices in a subscript expression.

        Example:

        ```
        // These two name the same attribute:
        #ekl.ellipsis
        #ekl<...>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::EllipsisAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        /// Gets the EllipsisType singleton instance.
        auto getType() const -> EllipsisType;
    }];
}

def EKL_MetadataAttr : EKL_Attr<"Metadata"> {
    let mnemonic = "meta";
    let summary = "Metadata constructor literal attribute";
    let description = [{
        The `ekl.meta` attibute represents a literal constructor invocation,
        which is used to represent program metadata.

        Example:

        ```
        // These two name the same attriubte:
        #ekl.meta<host>
        #ekl.meta<host()>

        #ekl.meta<HDL("my_top.v", "my_top")>
        ```
    }];

    let parameters = (ins
        "StringAttr":$name,
        "TupleAttr":$arguments);

    let assemblyFormat = [{
        `<` custom<KeywordOrString>($name)`` custom<OptionalTuple>($arguments) `>`
    }];
}

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

def EKL_LiteralAttr : Attr<
        /*predicate=*/CPred<"llvm::isa<::mlir::ekl::LiteralAttr>($_self)">,
        /*summary=*/"Comptime EKL value"> {
    let description = [{
        Implements an attribute constraint that accepts any of the valid EKL
        literal attributes:

            - ScalarAttr
            - StringAttr
            - ArrayAttr
            - SliceAttr
            - AxisAttr
            - EllipsisAttr
    }];

    let storageType = "::mlir::ekl::LiteralAttr";
    let returnType = "::mlir::ekl::LiteralAttr";
    let convertFromStorage = "$_self";
}

#endif // EKL_ATTRIBUTES
