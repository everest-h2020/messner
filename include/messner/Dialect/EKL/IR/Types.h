/// Declaration of the EKL dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h" // IWYU pragma: keep
#include "messner/Dialect/EKL/Analysis/Shape.h"
#include "messner/Dialect/EKL/IR/Assembly.h"               // IWYU pragma: keep
#include "messner/Dialect/EKL/IR/Dialect.h"                // IWYU pragma: keep
#include "messner/Dialect/EKL/Interfaces/ContiguousType.h" // IWYU pragma: keep
#include "messner/Dialect/Ref/Enums.h"
#include "messner/Dialect/Ref/Interfaces/ReferenceTypeInterface.h" // IWYU pragma: keep

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ODS types
//===----------------------------------------------------------------------===//
//
// Forward declarations of the ODS-generated types, so that we can use them to
// declare the named constraints.

class ExpressionType;
class RationalType;
class IndexType;
class StringType;
class ArrayType;
class ReferenceType;
class SliceType;
class AxisType;
class EllipsisType;

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

struct IntegerType : mlir::IntegerType {
    static auto classof(mlir::IntegerType type) -> bool;
    static auto classof(Type type) -> bool;

    /// Obtains the canonical IntegerType for @p bitWidth and @p isSigned .
    ///
    /// @pre    `context`
    /// @pre    `bitWidth < IntegerType::kMaxWidth`
    static auto
    get(MLIRContext *context, unsigned bitWidth, bool isSigned = true)
        -> IntegerType;

    /*implicit*/ IntegerType() = default;
    /*implicit*/ IntegerType(const mlir::Type::ImplType *impl);

    auto isSignless() const -> bool;
};

struct BoolType : mlir::IntegerType {
    static auto classof(mlir::IntegerType type) -> bool;
    static auto classof(Type type) -> bool;

    /// Obtains the canonical BoolType.
    ///
    /// @pre    `context`
    static BoolType get(MLIRContext *context);

    /*implicit*/ BoolType() = default;
    /*implicit*/ BoolType(const mlir::Type::ImplType *impl);

    auto getWidth() const -> unsigned;
    auto getSignedness() const -> SignednessSemantics;
    auto isSignless() const -> bool;
    auto isSigned() const -> bool;
    auto isUnsigned() const -> bool;
};

struct ScalarType : Type {
    static auto classof(FloatType) -> bool;
    static auto classof(IntegerType) -> bool;
    static auto classof(BoolType) -> bool;
    static auto classof(RationalType) -> bool;
    static auto classof(IndexType) -> bool;
    static auto classof(Type type) -> bool;

    using Type::Type;
    /*implicit*/ ScalarType(FloatType type);
    /*implicit*/ ScalarType(IntegerType type);
    /*implicit*/ ScalarType(BoolType type);
    /*implicit*/ ScalarType(RationalType type);
    /*implicit*/ ScalarType(IndexType type);
};

struct LiteralType : Type {
    static auto classof(ScalarType) -> bool;
    static auto classof(StringType) -> bool;
    static auto classof(ArrayType) -> bool;
    static auto classof(SliceType) -> bool;
    static auto classof(AxisType) -> bool;
    static auto classof(EllipsisType) -> bool;
    static auto classof(Type type) -> bool;

    using Type::Type;
    /*implicit*/ LiteralType(ScalarType type);
    /*implicit*/ LiteralType(StringType type);
    /*implicit*/ LiteralType(ArrayType type);
    /*implicit*/ LiteralType(SliceType type);
    /*implicit*/ LiteralType(AxisType type);
    /*implicit*/ LiteralType(EllipsisType type);
};

struct NumberType : ScalarType {
    static auto classof(FloatType) -> bool;
    static auto classof(IntegerType) -> bool;
    static auto classof(RationalType) -> bool;
    static auto classof(Type type) -> bool;

    /*implicit*/ NumberType() = default;
    /*implicit*/ NumberType(const ImplType *impl);
    /*implicit*/ NumberType(FloatType type);
    /*implicit*/ NumberType(IntegerType type);
    /*implicit*/ NumberType(RationalType type);
};

struct BroadcastType : Type {
    static auto classof(ScalarType) -> bool;
    static auto classof(ArrayType) -> bool;
    static auto classof(Type type) -> bool;

    using Type::Type;
    /*implicit*/ BroadcastType(ScalarType type);
    /*implicit*/ BroadcastType(ArrayType type);

    /// Gets the underlying ScalarType.
    auto getScalarType() const -> ScalarType;
    /// Gets the underlying shape, which may be empty.
    auto getShape() const -> ShapeRef;

    /// Obtains a BroadcastType with the same shape and @p scalarTy .
    ///
    /// @pre    `scalarTy`
    auto cloneWith(ScalarType scalarTy) const -> BroadcastType;
    /// Obtains a BroadcastType with the same scalar type and @p shape .
    auto cloneWith(ShapeRef shape) const -> ArrayType;
};

struct LogicType : BroadcastType {
    static auto classof(BoolType) -> bool;
    static auto classof(ArrayType type) -> bool;
    static auto classof(Type type) -> bool;

    /*implicit*/ LogicType() = default;
    /*implicit*/ LogicType(const ImplType *impl);
    /*implicit*/ LogicType(BoolType type);

    /// Gets the underlying BoolType.
    auto getScalarType() const -> BoolType;

    /// Obtains a LogicType with @p shape .
    auto cloneWith(ShapeRef shape) const -> LogicType;
};

struct ArithmeticType : BroadcastType {
    static auto classof(NumberType) -> bool;
    static auto classof(ArrayType type) -> bool;
    static auto classof(Type type) -> bool;

    /*implicit*/ ArithmeticType() = default;
    /*implicit*/ ArithmeticType(const ImplType *impl);
    /*implicit*/ ArithmeticType(NumberType type);

    /// Gets the underlying NumberType.
    auto getScalarType() const -> NumberType;

    /// Obtains a ArithmeticType with the same shape and @p scalarTy .
    ///
    /// @pre    `scalarTy`
    auto cloneWith(NumberType scalarTy) const -> ArithmeticType;
    /// Obtains a ArithmeticType with @p shape .
    auto cloneWith(ShapeRef shape) const -> ArithmeticType;
};

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/EKL/IR/Types.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// IntegerType implementation
//===----------------------------------------------------------------------===//

inline auto IntegerType::classof(mlir::IntegerType type) -> bool
{
    return !type.isSignless() && type.getWidth() > 0U;
}

inline auto IntegerType::classof(Type type) -> bool
{
    if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(type); intTy)
        return classof(intTy);
    return false;
}

inline IntegerType::IntegerType(const mlir::Type::ImplType *impl)
        : mlir::IntegerType(impl)
{}

inline auto
IntegerType::get(MLIRContext *context, unsigned int bitWidth, bool isSigned)
    -> IntegerType
{
    return llvm::cast<IntegerType>(mlir::IntegerType::get(
        context,
        bitWidth,
        isSigned ? SignednessSemantics::Signed
                 : SignednessSemantics::Unsigned));
}

inline auto IntegerType::isSignless() const -> bool { return false; }

//===----------------------------------------------------------------------===//
// BoolType implementation
//===----------------------------------------------------------------------===//

inline auto BoolType::classof(mlir::IntegerType type) -> bool
{
    return type.isSignless() && type.getWidth() == 1U;
}

inline auto BoolType::classof(Type type) -> bool
{
    if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(type); intTy)
        return classof(intTy);
    return false;
}

inline auto BoolType::get(MLIRContext *context) -> BoolType
{
    return llvm::cast<BoolType>(mlir::IntegerType::get(context, 1U));
}

inline BoolType::BoolType(const mlir::Type::ImplType *impl)
        : mlir::IntegerType(impl)
{}

inline auto BoolType::getWidth() const -> unsigned { return 1U; }

inline auto BoolType::getSignedness() const -> SignednessSemantics
{
    return SignednessSemantics::Signless;
}

inline auto BoolType::isSignless() const -> bool { return true; }

inline auto BoolType::isSigned() const -> bool { return false; }

inline auto BoolType::isUnsigned() const -> bool { return false; }

//===----------------------------------------------------------------------===//
// ScalarType implementation
//===----------------------------------------------------------------------===//

inline auto ScalarType::classof(FloatType) -> bool { return true; }

inline auto ScalarType::classof(IntegerType) -> bool { return true; }

inline auto ScalarType::classof(BoolType) -> bool { return true; }

inline auto ScalarType::classof(RationalType) -> bool { return true; }

inline auto ScalarType::classof(IndexType) -> bool { return true; }

inline auto ScalarType::classof(Type type) -> bool
{
    return llvm::isa<FloatType, IntegerType, BoolType, RationalType, IndexType>(
        type);
}

inline ScalarType::ScalarType(FloatType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

inline ScalarType::ScalarType(IntegerType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

inline ScalarType::ScalarType(BoolType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

inline ScalarType::ScalarType(RationalType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

inline ScalarType::ScalarType(IndexType type)
        : ScalarType(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// LiteralType implementation
//===----------------------------------------------------------------------===//

inline auto LiteralType::classof(ScalarType) -> bool { return true; }

inline auto LiteralType::classof(StringType) -> bool { return true; }

inline auto LiteralType::classof(ArrayType) -> bool { return true; }

inline auto LiteralType::classof(SliceType) -> bool { return true; }

inline auto LiteralType::classof(AxisType) -> bool { return true; }

inline auto LiteralType::classof(EllipsisType) -> bool { return true; }

inline auto LiteralType::classof(Type type) -> bool
{
    return llvm::isa<
        ScalarType,
        StringType,
        ArrayType,
        SliceType,
        AxisType,
        EllipsisType>(type);
}

inline LiteralType::LiteralType(ScalarType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(StringType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(ArrayType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(SliceType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(AxisType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

inline LiteralType::LiteralType(EllipsisType type)
        : LiteralType(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// NumberType implementation
//===----------------------------------------------------------------------===//

inline auto NumberType::classof(FloatType) -> bool { return true; }

inline auto NumberType::classof(IntegerType) -> bool { return true; }

inline auto NumberType::classof(RationalType) -> bool { return true; }

inline auto NumberType::classof(Type type) -> bool
{
    return llvm::isa<FloatType, IntegerType, IndexType, RationalType>(type);
}

inline NumberType::NumberType(const ImplType *impl) : ScalarType(impl) {}

inline NumberType::NumberType(FloatType type)
        : NumberType(static_cast<Type>(type).getImpl())
{}

inline NumberType::NumberType(IntegerType type)
        : NumberType(static_cast<Type>(type).getImpl())
{}

inline NumberType::NumberType(RationalType type)
        : NumberType(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// BroadcastType implementation
//===----------------------------------------------------------------------===//

inline auto BroadcastType::classof(ScalarType) -> bool { return true; }

inline auto BroadcastType::classof(ArrayType) -> bool { return true; }

inline auto BroadcastType::classof(Type type) -> bool
{
    return llvm::isa<ScalarType, ArrayType>(type);
}

inline BroadcastType::BroadcastType(ScalarType type)
        : BroadcastType(static_cast<Type>(type).getImpl())
{}

inline BroadcastType::BroadcastType(ArrayType type)
        : BroadcastType(static_cast<Type>(type).getImpl())
{}

inline auto BroadcastType::getScalarType() const -> ScalarType
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(*this); arrayTy)
        return arrayTy.getScalarType();
    return llvm::cast<ScalarType>(*this);
}

inline auto BroadcastType::getShape() const -> ShapeRef
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(*this); arrayTy)
        return arrayTy.getShape();
    return {};
}

inline auto BroadcastType::cloneWith(ScalarType scalarTy) const -> BroadcastType
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(*this); arrayTy)
        return arrayTy.cloneWith(scalarTy);
    return scalarTy;
}

inline auto BroadcastType::cloneWith(ShapeRef shape) const -> ArrayType
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(*this); arrayTy)
        return arrayTy.cloneWith(shape);
    return ArrayType::get(getContext(), llvm::cast<ScalarType>(*this), shape);
}

//===----------------------------------------------------------------------===//
// LogicType implementation
//===----------------------------------------------------------------------===//

inline auto LogicType::classof(BoolType) -> bool { return true; }

inline auto LogicType::classof(ArrayType type) -> bool
{
    return llvm::isa<BoolType>(type.getScalarType());
}

inline auto LogicType::classof(Type type) -> bool
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(type); arrayTy)
        return classof(arrayTy);
    return llvm::isa<BoolType>(type);
}

inline LogicType::LogicType(const ImplType *impl) : BroadcastType(impl) {}

inline LogicType::LogicType(BoolType type)
        : LogicType(static_cast<Type>(type).getImpl())
{}

inline auto LogicType::getScalarType() const -> BoolType
{
    return BoolType::get(getContext());
}

inline auto LogicType::cloneWith(ShapeRef shape) const -> LogicType
{
    return llvm::cast<LogicType>(BroadcastType::cloneWith(shape));
}

//===----------------------------------------------------------------------===//
// ArithmeticType implementation
//===----------------------------------------------------------------------===//

inline auto ArithmeticType::classof(NumberType) -> bool { return true; }

inline auto ArithmeticType::classof(ArrayType type) -> bool
{
    return llvm::isa<NumberType>(type.getScalarType());
}

inline auto ArithmeticType::classof(Type type) -> bool
{
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(type); arrayTy)
        return classof(arrayTy);
    return llvm::isa<NumberType>(type);
}

inline ArithmeticType::ArithmeticType(const ImplType *impl)
        : BroadcastType(impl)
{}

inline ArithmeticType::ArithmeticType(NumberType type)
        : ArithmeticType(static_cast<Type>(type).getImpl())
{}

inline auto ArithmeticType::getScalarType() const -> NumberType
{
    return llvm::cast<NumberType>(BroadcastType::getScalarType());
}

inline auto ArithmeticType::cloneWith(NumberType scalarTy) const
    -> ArithmeticType
{
    return llvm::cast<ArithmeticType>(BroadcastType::cloneWith(scalarTy));
}

inline auto ArithmeticType::cloneWith(ShapeRef shape) const -> ArithmeticType
{
    return llvm::cast<ArithmeticType>(BroadcastType::cloneWith(shape));
}

//===----------------------------------------------------------------------===//
// IndexType implementation
//===----------------------------------------------------------------------===//

inline auto IndexType::get(MLIRContext *context, Index value) -> IndexType
{
    return get(
        context,
        value.getBound().value_or(unbounded),
        value.isFromEnd());
}

//===----------------------------------------------------------------------===//
// ArrayType implementation
//===----------------------------------------------------------------------===//

inline auto ArrayType::get(Type scalarOrArrayTy, ShapeRef shape) -> ArrayType
{
    assert(scalarOrArrayTy);
    assert(isValid(shape));

    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(scalarOrArrayTy);
        arrayTy) {
        get(arrayTy.getScalarType(), concat(shape, arrayTy.getShape()));
    }

    return get(llvm::cast<ScalarType>(scalarOrArrayTy), shape);
}

inline auto ArrayType::get(ScalarType scalarTy, ShapeRef shape) -> ArrayType
{
    assert(scalarTy);
    assert(isValid(shape));

    return get(scalarTy.getContext(), scalarTy, shape);
}

inline auto ArrayType::isValid(ShapeRef shape) -> bool
{
    const auto flattened = flatten(shape);
    return succeeded(flattened) && flattened->isBounded()
        && *flattened != Extent{};
}

inline auto ArrayType::cloneWith(Type scalarOrArrayTy) const -> ArrayType
{
    assert(scalarOrArrayTy);

    return get(scalarOrArrayTy, getShape());
}

inline auto ArrayType::cloneWith(ScalarType scalarTy) const -> ArrayType
{
    assert(scalarTy);

    return get(scalarTy, getShape());
}

inline auto ArrayType::cloneWith(ShapeRef shape) const -> ArrayType
{
    assert(isValid(shape));

    return get(getScalarType(), shape);
}

//===----------------------------------------------------------------------===//
// ReferenceType implementation
//===----------------------------------------------------------------------===//

inline auto
ReferenceType::get(ref::ReferenceKind kind, ScalarType scalarTy, ShapeRef shape)
    -> ReferenceType
{
    assert(scalarTy);
    assert(isValid(shape));

    return get(scalarTy.getContext(), kind, scalarTy, shape);
}

inline auto
ReferenceType::get(ref::ReferenceKind kind, ArrayType arrayTy, bool isScalable)
    -> ReferenceType
{
    assert(arrayTy);

    if (!isScalable)
        return get(kind, arrayTy.getScalarType(), arrayTy.getShape());
    return get(
        kind,
        arrayTy.getScalarType(),
        concat({unbounded}, arrayTy.getShape()));
}

inline auto ReferenceType::isValid(ShapeRef shape) -> bool
{
    if (!shape.empty() && shape.front() == unbounded)
        shape = shape.drop_front();
    return ArrayType::isValid(shape);
}

inline auto ReferenceType::isScalable() const -> bool
{
    return !getShape().empty() && getShape().front() == unbounded;
}

inline auto ReferenceType::getCellType() const -> ArrayType
{
    auto shape = getShape();
    if (!shape.empty() && shape.front() == unbounded)
        shape = shape.drop_front();
    return ArrayType::get(getContext(), getScalarType(), shape);
}

//===----------------------------------------------------------------------===//
// SliceType implementation
//===----------------------------------------------------------------------===//

inline auto SliceType::get(MLIRContext *context, const Slice &slice)
    -> SliceType
{
    return get(context, slice.getBegin(), slice.getEnd(), slice.getStride());
}

inline auto SliceType::tryGetValue() const -> std::optional<Slice>
{
    if (getBegin() && getEnd()) return Slice(getBegin(), getEnd(), getStride());
    return std::nullopt;
}

} // namespace mlir::ekl
