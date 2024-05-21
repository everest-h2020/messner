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
    let summary = "Array literal value attribute";
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
        "mlir::ArrayAttr":$stack);

    let builders = [
        AttrBuilderWithInferredContext<(ins
                "ArrayType":$arrayType,
                "mlir::ArrayAttr":$stack), [{
            return $_get(arrayType.getContext(), arrayType, stack);
        }]>,
        AttrBuilderWithInferredContext<(ins
                "ArrayType":$arrayType,
                "mlir::ArrayAttr::ValueType":$stack), [{
            return $_get(
                arrayType.getContext(),
                arrayType,
                mlir::ArrayAttr::get(arrayType.getContext(), stack));
        }]>,
        AttrBuilderWithInferredContext<(ins
                "ArrayType":$arrayType,
                "ScalarAttr":$splatValue), [{
            return $_get(
                arrayType.getContext(),
                arrayType,
                mlir::ArrayAttr::get(arrayType.getContext(), {splatValue}));
        }]>,
        AttrBuilderWithInferredContext<(ins
                "ScalarAttr":$splatValue,
                CArg<"ExtentRange", "{}">:$extents), [{
            return $_get(
                splatValue.getContext(),
                ArrayType::get(
                    splatValue.getContext(),
                    splatValue.getType(),
                    extents),
                mlir::ArrayAttr::get(splatValue.getContext(), {splatValue}));
        }]>,
        AttrBuilderWithInferredContext<(ins
                "ArrayAttr":$bcastValue,
                "extent_t":$extent), [{
            return $_get(
                bcastValue.getContext(),
                bcastValue.getType().cloneWith(
                    concat({extent}, bcastValue.getType().getExtents())),
                mlir::ArrayAttr::get(bcastValue.getContext(), {bcastValue}));
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = [{ `<` $stack `>` }];
    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
        /// Determines whether the value is stored as a broadcast.
        [[nodiscard]] bool isBroadcast() const
        {
            return getStack().size() == 1;
        }
        /// Obtains the stored broadcast value, if any.
        [[nodiscard]] Attribute getBroadcastValue() const
        {
            return isBroadcast() ? *getStack().begin() : Attribute{};
        }

        /// Determines whether the value is stored as a splat.
        [[nodiscard]] bool isSplat() const
        {
            return !!getSplatValue();
        }
        /// Obtains the stored splat value, if any.
        [[nodiscard]] ScalarAttr getSplatValue() const
        {
            return llvm::dyn_cast_if_present<ScalarAttr>(getBroadcastValue());
        }

        /// Applies a basic (partial) index subscript to this array.
        ///
        /// @pre    `getType().isInBounds(indices)`
        [[nodiscard]] Attribute subscript(ExtentRange indices) const;

        /// Broadcasts this array to the @p extents .
        ///
        /// @pre    `getType().isBroadcastableTo(extents)`
        [[nodiscard]] ArrayAttr broadcastTo(ExtentRange extents) const
        {
            assert(getType().isBroadcastableTo(extents));

            return get(getType().cloneWith(extents), getStack());
        }

        //===--------------------------------------------------------------===//
        // TypedAttrInterface
        //===--------------------------------------------------------------===//

        [[nodiscard]] ArrayType getType() const { return getArrayType(); }
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

def EKL_EllipsisAttr : EKL_Attr<"Ellipsis"> {
    let mnemonic = "ellipsis";
    let summary = "Ellipsis literal attribute";
    let description = [{
        The `ekl.ellipsis` attribute represents the `...` literal, which expands
        to identity indexers in a subscript expression.

        Example:

        ```
        #ekl.ellipsis
        #ekl<...>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::EllipsisAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        // Gets the EllipsisType singleton instance.
        [[nodiscard]] EllipsisType getType() const
        {
            return EllipsisType::get(getContext());
        }
    }];
}

def EKL_ErrorAttr : EKL_Attr<"Error"> {
    let mnemonic = "error";
    let summary = "Error literal attribute";
    let description = [{
        The `ekl.error` attribute represents the result of an erroneous
        expression.

        Example:

        ```
        #ekl.error
        #ekl<?>
        ```
    }];

    code builderCall = [{
        ::mlir::ekl::ErrorAttr::get($_builder.getContext());
    }];

    code extraClassDeclaration = [{
        // Gets the ErrorType singleton instance.
        [[nodiscard]] ErrorType getType() const
        {
            return ErrorType::get(getContext());
        }
    }];
}

//===----------------------------------------------------------------------===//
// Debug information
//===----------------------------------------------------------------------===//

def EKL_SourceLocation : EKL_Attr<"SourceLocation"> {
    let mnemonic = "loc";
    let summary = [{
        The `ekl.loc` attribute is a workaround for an MLIR limitation that does
        not allow the user to specify new derived `LocationAttr` attributes,
        because the printer will not handle them correctly.

        Instead, we use the attribute storage uniquer to store location info
        in a `SourceLocationAttr`, and attach it to an `OpaqueLoc`. This is not
        round-trippable, but at least it works at all.

        Example:

        ```
        #ekl.loc<"file.ekl" 1:1>
        #ekl.loc<"file.ekl" 1:1:10>
        #ekl.loc<"file.ekl" 1:1:4:10>
        ```
    }];

    let parameters = (ins
        "StringAttr":$filename,
        "unsigned":$startLine,
        "unsigned":$startColumn,
        "unsigned":$endLine,
        "unsigned":$endColumn);

    let builders = [
        AttrBuilderWithInferredContext<(ins
                "StringAttr":$filename,
                "unsigned":$startLine,
                "unsigned":$startColumn,
                "unsigned":$endLine,
                "unsigned":$endColumn), [{
            assert(filename);
            assert(startLine <= endLine);
            assert(startLine != endLine || startColumn <= endColumn);

            return $_get(
                filename.getContext(),
                filename,
                startLine,
                startColumn,
                endLine,
                endColumn);
        }]>,
        AttrBuilder<(ins
                "StringRef":$filename,
                "unsigned":$startLine,
                "unsigned":$startColumn,
                "unsigned":$endLine,
                "unsigned":$endColumn), [{
            return get(
                StringAttr::get($_ctxt, filename),
                startLine,
                startColumn,
                endLine,
                endColumn);
        }]>,
        AttrBuilderWithInferredContext<(ins
                "StringAttr":$filename,
                "unsigned":$line,
                "unsigned":$column), [{
            return get(filename, line, column, line, column);
        }]>,
        AttrBuilder<(ins
                "StringRef":$filename,
                "unsigned":$line,
                "unsigned":$column), [{
            return get(StringAttr::get($_ctxt, filename), line, column);
        }]>
    ];

    let assemblyFormat = [{
        `<`
        $filename
        custom<SourceRange>($startLine, $startColumn, $endLine, $endColumn)
        `>`
    }];

    code extraClassDeclaration = [{
        /// Obtains the SourceLocationAttr inside @p loc , if any.
        [[nodiscard]] static SourceLocationAttr fromLocation(OpaqueLoc loc);

        /// Obtains a FileLineColLoc that encapsulates the start location.
        [[nodiscard]] FileLineColLoc toStartLocation() const
        {
            return FileLineColLoc::get(
                getFilename(),
                getStartLine(),
                getStartColumn());
        }

        /// Obtains an OpaqueLoc that encapsulates this SourceLocationAttr.
        [[nodiscard]] OpaqueLoc toLocation() const;
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
            - IdentityAttr
            - ExtentAttr
            - EllipsisAttr
            - ErrorAttr
    }];

    let storageType = "::mlir::ekl::LiteralAttr";
    let returnType = "::mlir::ekl::LiteralAttr";
    let convertFromStorage = "$_self";
}

#endif
