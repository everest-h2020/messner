/// Declaration of the EKL dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Base.h"
#include "messner/Dialect/EKL/Interfaces/ContiguousType.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

/// Parses a list of array extents, or nothing.
///
/// This parser implements the following grammar:
///
/// ```
/// extents ::= [ `[` [ uint { `,` uint } ] `]` ]
/// ```
///
/// @param  [in]        parser      OpAsmParser.
/// @param  [out]       extents     Extents.
///
/// @return ParseResult.
ParseResult parseExtents(AsmParser &parser, SmallVectorImpl<extent_t> &extents);

/// Prints a list of array extents, or nothing.
///
/// This printer implements the following grammar:
///
/// ```
/// extents ::= [ `[` [ uint { `,` uint } ] `]` ]
/// ```
///
/// @param  [in]        printer     OpAsmPrinter.
/// @param              extents     Extents.
void printExtents(AsmPrinter &printer, ExtentRange extents);

//===----------------------------------------------------------------------===//
// ODS types
//===----------------------------------------------------------------------===//
//
// Forward declarations of the ODS-generated types, so that we can use them to
// declare the named constraints.

class ExpressionType;
class NumberType;
class IndexType;
class StringType;
class ArrayType;
class ReferenceType;
class IdentityType;
class ExtentType;

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

/// Implements the IntegerType named constraint.
struct IntegerType : mlir::IntegerType {
    using mlir::IntegerType::IntegerType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(mlir::IntegerType type)
    {
        return !type.isSignless();
    }
    /// Determines whether @p type is an IntegerType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(type))
            return classof(intTy);
        return false;
    }

    /// Obtains the canonical IntegerType for @p bitWidth and @p isSigned .
    ///
    /// @pre    `context`
    /// @pre    `bitWidth < IntegerType::kMaxWidth`
    [[nodiscard]] static IntegerType
    get(MLIRContext *context, unsigned bitWidth, bool isSigned = true)
    {
        return llvm::cast<IntegerType>(mlir::IntegerType::get(
            context,
            bitWidth,
            isSigned ? SignednessSemantics::Signed
                     : SignednessSemantics::Unsigned));
    }

    [[nodiscard]] bool isSignless() const { return false; }
};

/// Implements the BoolType named constraint.
struct BoolType : mlir::IntegerType {
    using mlir::IntegerType::IntegerType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(mlir::IntegerType type)
    {
        return type.isSignless() && type.getWidth() == 1;
    }
    /// Determines whether @p type is a BoolType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(type))
            return classof(intTy);
        return false;
    }

    /// Obtains the canonical BoolType.
    ///
    /// @pre    `context`
    [[nodiscard]] static BoolType get(MLIRContext *context)
    {
        return llvm::cast<BoolType>(mlir::IntegerType::get(context, 1));
    }

    [[nodiscard]] unsigned getWidth() const { return 1U; }
    [[nodiscard]] SignednessSemantics getSignedness() const
    {
        return SignednessSemantics::Signless;
    }
    [[nodiscard]] bool isSignless() const { return true; }
    [[nodiscard]] bool isSigned() const { return false; }
    [[nodiscard]] bool isUnsigned() const { return false; }
};

/// Implements the ScalarType named constraint.
struct ScalarType : Type {
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(NumberType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ekl::IntegerType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(FloatType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ekl::IndexType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(BoolType) { return true; }
    /// Determines whether @p type is a ScalarType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type);

    /*implicit*/ ScalarType(NumberType type);
    /*implicit*/ ScalarType(ekl::IntegerType type)
            : Type(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ ScalarType(FloatType type)
            : Type(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ ScalarType(ekl::IndexType type);
    /*implicit*/ ScalarType(BoolType type)
            : Type(static_cast<Type>(type).getImpl())
    {}
};

/// Implements the NumericType named constraint.
struct NumericType : ScalarType {
    using ScalarType::ScalarType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(NumberType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(IntegerType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(FloatType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ekl::IndexType);
    /// Determines whether @p type is a NumericType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type);

    /*implicit*/ NumericType(NumberType);
    /*implicit*/ NumericType(IntegerType type)
            : ScalarType(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ NumericType(FloatType type)
            : ScalarType(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ NumericType(ekl::IndexType);
};

/// Implements the TensorType named constraint.
struct TensorType : Type {
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(NumericType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ArrayType);
    /// Determines whether @p type is a TensorType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type);

    /*implicit*/ TensorType(NumericType type)
            : Type(static_cast<Type>(type).getImpl())
    {}

    /// Gets the underlying ScalarType.
    [[nodiscard]] NumericType getScalarType() const;
    /// Gets the extents.
    [[nodiscard]] ExtentRange getExtents() const;
};

/// Implements the LiteralType named constraint.
struct LiteralType : Type {
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ScalarType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(StringType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ArrayType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(IdentityType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ExtentType);
    /// Determines whether @p type is a LiteralType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type);

    /*implicit*/ LiteralType(ScalarType type)
            : Type(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ LiteralType(StringType);
    /*implicit*/ LiteralType(ArrayType);
    /*implicit*/ LiteralType(IdentityType);
    /*implicit*/ LiteralType(ExtentType);
};

//===----------------------------------------------------------------------===//
// ABI type constraints
//===----------------------------------------------------------------------===//

/// Implements the ABIScalarType named constraint.
struct ABIScalarType : ScalarType {
    using ScalarType::ScalarType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ekl::IntegerType type)
    {
        return type.getWidth() > 0U;
    }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(FloatType type);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(BoolType) { return true; }
    /// Determines whether @p type is an ABIScalarType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type)
    {
        return llvm::TypeSwitch<Type, bool>(type)
            .Case([](ekl::IntegerType intTy) { return classof(intTy); })
            .Case([](FloatType floatTy) { return classof(floatTy); })
            .Case([](BoolType) { return true; })
            .Default(false);
    }

    /*implicit*/ ABIScalarType(BoolType type) : ScalarType(type) {}

    /// Gets the corresponding LLVM type.
    [[nodiscard]] Type getLLVMType() const;
};

// Constraint is derived and thus must be declared later.
struct ABIReferenceType;

/// Implements the ABIType named constraint.
struct ABIType : Type {
    using Type::Type;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ABIScalarType) { return true; }
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ABIReferenceType);
    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ArrayType);
    /// Determines whether @p type is an ABIType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type);

    /*implicit*/ ABIType(ABIScalarType type)
            : Type(static_cast<Type>(type).getImpl())
    {}
    /*implicit*/ ABIType(ABIReferenceType type);

    /// Gets the corresponding LLVM type.
    [[nodiscard]] Type getLLVMType() const;
};

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/EKL/IR/Types.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ScalarType implementation
//===----------------------------------------------------------------------===//

inline bool ScalarType::classof(NumberType) { return true; }

inline bool ScalarType::classof(ekl::IndexType) { return true; }

inline ScalarType::ScalarType(NumberType type)
        : Type(static_cast<Type>(type).getImpl())
{}

inline ScalarType::ScalarType(ekl::IndexType type)
        : Type(static_cast<Type>(type).getImpl())
{}

inline bool ScalarType::classof(Type type)
{
    return llvm::TypeSwitch<Type, bool>(type)
        .Case([](NumberType) { return true; })
        .Case([](ekl::IntegerType) { return true; })
        .Case([](FloatType) { return true; })
        .Case([](ekl::IndexType) { return true; })
        .Case([](BoolType) { return true; })
        .Default(false);
}

//===----------------------------------------------------------------------===//
// LiteralType implementation
//===----------------------------------------------------------------------===//

inline bool LiteralType::classof(ArrayType) { return true; }

inline bool LiteralType::classof(StringType) { return true; }

inline bool LiteralType::classof(IdentityType) { return true; }

inline bool LiteralType::classof(ExtentType) { return true; }

inline bool LiteralType::classof(Type type)
{
    return llvm::TypeSwitch<Type, bool>(type)
        .Case([](ScalarType) { return true; })
        .Case([](StringType) { return true; })
        .Case([](ArrayType) { return true; })
        .Case([](IdentityType) { return true; })
        .Case([](ExtentType) { return true; })
        .Default(false);
}

inline LiteralType::LiteralType(StringType type)
        : Type(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(ArrayType type)
        : Type(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(IdentityType type)
        : Type(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(ExtentType type)
        : Type(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// NumericType implementation
//===----------------------------------------------------------------------===//

inline bool NumericType::classof(NumberType) { return true; }

inline bool NumericType::classof(ekl::IndexType) { return true; }

inline bool NumericType::classof(Type type)
{
    return llvm::TypeSwitch<Type, bool>(type)
        .Case([](NumberType) { return true; })
        .Case([](IntegerType) { return true; })
        .Case([](FloatType) { return true; })
        .Case([](ekl::IndexType) { return true; })
        .Default(false);
}

inline NumericType::NumericType(NumberType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

inline NumericType::NumericType(ekl::IndexType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// TensorType implementation
//===----------------------------------------------------------------------===//

inline bool TensorType::classof(ArrayType type)
{
    return llvm::isa<NumericType>(type.getScalarType());
}

inline bool TensorType::classof(Type type)
{
    return llvm::TypeSwitch<Type, bool>(type)
        .Case([](NumericType) { return true; })
        .Case([](ArrayType arrayTy) { return classof(arrayTy); })
        .Default(false);
}

inline NumericType TensorType::getScalarType() const
{
    if (const auto numericTy = llvm::dyn_cast<NumericType>(*this))
        return numericTy;
    return llvm::cast<NumericType>(
        llvm::cast<ArrayType>(*this).getScalarType());
}

inline ExtentRange TensorType::getExtents() const
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(*this))
        return arrayTy.getExtents();
    return {};
}

//===----------------------------------------------------------------------===//
// ABIReferenceType implementation
//===----------------------------------------------------------------------===//

/// Implements the ABIReferenceType named constraint.
struct ABIReferenceType : ReferenceType {
    using ReferenceType::ReferenceType;

    /// @copydoc classof(Type)
    [[nodiscard]] inline static bool classof(ReferenceType type)
    {
        return llvm::isa<ABIScalarType>(type.getScalarType());
    }
    /// Determines whether @p type is an ABIReferenceType.
    ///
    /// @pre    `type`
    [[nodiscard]] inline static bool classof(Type type)
    {
        if (const auto refTy = llvm::dyn_cast<ReferenceType>(type))
            return classof(refTy);
        return false;
    }

    /// Gets the corresponding LLVM type.
    [[nodiscard]] Type getLLVMType() const;
};

//===----------------------------------------------------------------------===//
// ABIType implementation
//===----------------------------------------------------------------------===//

inline bool ABIType::classof(ABIReferenceType) { return true; }

inline bool ABIType::classof(Type type)
{
    return llvm::TypeSwitch<Type, bool>(type)
        .Case([](ABIScalarType) { return true; })
        .Case([](ABIReferenceType) { return true; })
        .Case([](ArrayType arrayTy) { return classof(arrayTy); })
        .Default(false);
}

inline ABIType::ABIType(ABIReferenceType type)
        : Type(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// Derived named constraints
//===----------------------------------------------------------------------===//

/// Implements the ReadableRefType named constraint.
struct ReadableRefType : ReferenceType {
    using ReferenceType::ReferenceType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ReferenceType type)
    {
        return type.isReadable();
    }
    /// Determines whether @p type is a ReadableRefType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto refTy = llvm::dyn_cast<ReferenceType>(type))
            return classof(refTy);
        return false;
    }

    /// @copydoc ReferenceType::isReadable()
    [[nodiscard]] bool isReadable() const { return true; }
};

/// Implements the WritableRefType named constraint.
struct WritableRefType : ReferenceType {
    using ReferenceType::ReferenceType;

    /// @copydoc classof(Type)
    [[nodiscard]] static bool classof(ReferenceType type)
    {
        return type.isWritable();
    }
    /// Determines whether @p type is a WritableRefType.
    ///
    /// @pre    `type`
    [[nodiscard]] static bool classof(Type type)
    {
        if (const auto refTy = llvm::dyn_cast<ReferenceType>(type))
            return classof(refTy);
        return false;
    }

    /// @copydoc ReferenceType::isReadable()
    [[nodiscard]] bool isWritable() const { return true; }
};

} // namespace mlir::ekl
