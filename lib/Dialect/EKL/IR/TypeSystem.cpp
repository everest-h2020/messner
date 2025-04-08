/// Implementation of the EKL dialect type system.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/TypeSystem.h"

#include "messner/Dialect/EKL/IR/Types.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::ekl;

/// Determines whether every value of @p sub is contained in @p super .
///
/// @pre    `sub`
static auto isSubtype(ekl::IntegerType sub, const llvm::fltSemantics &super)
    -> bool
{
    assert(sub);

    // Check if the mantissa can represent any value of the subtype.
    if (sub.isSigned() && !APFloat::semanticsHasSignedRepr(super)) return false;
    const auto subWidth = sub.getWidth() - (sub.isSigned() ? 1U : 0U);
    return subWidth <= APFloat::semanticsPrecision(super);
}

/// Determines whether every value of @p sub is contained in @p super .
///
/// This does not compare NaN and Inf behavior, because EKL does not care about
/// any irrational values.
static auto
isSubtype(const llvm::fltSemantics &sub, const llvm::fltSemantics &super)
    -> bool
{
    // Compare signedness first.
    if (APFloat::semanticsHasSignedRepr(sub)
        && !APFloat::semanticsHasSignedRepr(super))
        return false;

    // Compare exponent value range and mantissa precision in bits.
    if (!APFloat::isRepresentableAsNormalIn(sub, super)) return false;

    if (APFloat::semanticsHasZero(sub) || !APFloat::semanticsHasZero(super))
        return false;
    // NOTE: In EKL, these types of values are UB.
    /*
    if (APFloat::semanticsHasNaN(subSema)
        || !APFloat::semanticsHasNaN(superSema))
        return false;
    if (APFloat::semanticsHasInf(subSema)
        || !APFloat::semanticsHasInf(superSema))
        return false;
    */
    return true;
}

/// Obtains the FloatType for @p sema .
///
/// @pre    `context`
///
/// @retval FloatType   MLIR FloatType for @p sema.
/// @retval nullptr     @p sema has no MLIR type equivalent.
static auto getFloatType(MLIRContext *context, APFloat::Semantics sema)
    -> FloatType
{
    assert(context);

    switch (sema) {
#define FLOAT_TYPE_SEMANTICS(TYPE, SEM)                                        \
    case APFloat::S_##SEM: return TYPE::get(context);

        FLOAT_TYPE_SEMANTICS(Float4E2M1FNType, Float4E2M1FN)
        FLOAT_TYPE_SEMANTICS(Float6E2M3FNType, Float6E2M3FN)
        FLOAT_TYPE_SEMANTICS(Float6E3M2FNType, Float6E3M2FN)
        FLOAT_TYPE_SEMANTICS(Float8E5M2Type, Float8E5M2)
        FLOAT_TYPE_SEMANTICS(Float8E4M3Type, Float8E4M3)
        FLOAT_TYPE_SEMANTICS(Float8E4M3FNType, Float8E4M3FN)
        FLOAT_TYPE_SEMANTICS(Float8E5M2FNUZType, Float8E5M2FNUZ)
        FLOAT_TYPE_SEMANTICS(Float8E4M3FNUZType, Float8E4M3FNUZ)
        FLOAT_TYPE_SEMANTICS(Float8E4M3B11FNUZType, Float8E4M3B11FNUZ)
        FLOAT_TYPE_SEMANTICS(Float8E3M4Type, Float8E3M4)
        FLOAT_TYPE_SEMANTICS(Float8E8M0FNUType, Float8E8M0FNU)
        FLOAT_TYPE_SEMANTICS(BFloat16Type, BFloat)
        FLOAT_TYPE_SEMANTICS(Float16Type, IEEEhalf)
        FLOAT_TYPE_SEMANTICS(FloatTF32Type, FloatTF32)
        FLOAT_TYPE_SEMANTICS(Float32Type, IEEEsingle)
        FLOAT_TYPE_SEMANTICS(Float64Type, IEEEdouble)
        FLOAT_TYPE_SEMANTICS(Float80Type, x87DoubleExtended)
        FLOAT_TYPE_SEMANTICS(Float128Type, IEEEquad)

#undef FLOAT_TYPE_SEMANTICS
    default: return {};
    }
}

/// Finds the smallest IEEE float with semantics satisfying @p pred and returns
/// its MLIR type.
///
/// @pre    `context`
///
/// @retval FloatType   Smallest IEEE float satisfying @p pred .
/// @retval nullptr     No IEEE float satisfying @p pred found.
static auto findIEEEFloat(
    MLIRContext *context,
    function_ref<bool(const llvm::fltSemantics &sema)> pred) -> FloatType
{
    static const std::array order{
        &APFloat::IEEEhalf(),
        &APFloat::IEEEsingle(),
        &APFloat::IEEEdouble(),
        &APFloat::IEEEquad()};

    for (auto *sema : order)
        if (pred(*sema))
            return getFloatType(context, APFloat::SemanticsToEnum(*sema));

    return {};
}

//===----------------------------------------------------------------------===//
// TypeSystem implementation
//===----------------------------------------------------------------------===//

auto TypeSystem::isSubtype(FloatType sub, FloatType super) const -> bool
{
    assert(sub && super);

    const auto &superSema = super.getFloatSemantics();
    const auto &subSema   = sub.getFloatSemantics();
    return ::isSubtype(subSema, superSema);
}

auto TypeSystem::isSubtype(IntegerType sub, FloatType super) const -> bool
{
    assert(sub && super);

    return ::isSubtype(sub, super.getFloatSemantics());
}

auto TypeSystem::isSubtype(ekl::IntegerType sub, ekl::IntegerType super) const
    -> bool
{
    assert(sub && super);

    if (sub.isSigned()) {
        // Signed types can not fit into unsigned types.
        if (super.isUnsigned()) return false;

        // With same signedness, we can just compare bit widths.
        return sub.getWidth() <= super.getWidth();
    }

    // If the supertype is signed, we have to subtract the sign bit from the
    // width, but then we can just compare widths as normal.
    const auto superWidth = super.getWidth() - (super.isSigned() ? 1U : 0U);
    return sub.getWidth() <= superWidth;
}

auto TypeSystem::isSubtype(Type sub, Type super) const -> bool
{
    // T :> T
    if (sub == super) return true;

    // Expr :> T
    if (llvm::isa<ExpressionType>(super)) return true;
    // T :> Expr <=> T = Expr
    if (llvm::isa<ExpressionType>(sub)) return false;

    // Rational :> Float, Rational :> Integer, Rational :> Index
    if (llvm::isa<RationalType>(super)) return llvm::isa<NumberType>(sub);

    // ... -> Float :> T
    if (const auto supFloat = llvm::dyn_cast<FloatType>(super); supFloat) {
        return llvm::TypeSwitch<Type, bool>(sub)
            .Case([&](FloatType subFloat) {
                return isSubtype(subFloat, supFloat);
            })
            .Case([&](ekl::IntegerType subInt) {
                return isSubtype(subInt, supFloat);
            })
            .Default(false);
    }

    // ... -> Integer :> T
    if (const auto supInt = llvm::dyn_cast<ekl::IntegerType>(super); supInt) {
        return llvm::TypeSwitch<Type, bool>(sub)
            .Case([&](ekl::IntegerType subInt) {
                return isSubtype(subInt, supInt);
            })
            .Default(false);
    }

    // ... -> Index :> T
    if (const auto supIndex = llvm::dyn_cast<ekl::IndexType>(super); supIndex) {
        return llvm::TypeSwitch<Type, bool>(super)
            .Case([&](ekl::IndexType subIndex) {
                return isSubtype(subIndex, supIndex);
            })
            .Default(false);
    }

    // ... -> Array(T, s) :> U
    if (const auto supArray = llvm::dyn_cast<ArrayType>(super); supArray) {
        return llvm::TypeSwitch<Type, bool>(sub)
            .Case([&](ArrayType subArray) {
                return isSubtype(subArray, supArray);
            })
            .Default(false);
    }

    return false;
}

auto TypeSystem::promote(FloatType lhs, FloatType rhs) const -> FloatType
{
    assert(lhs && rhs);

    const auto &lhsSema = lhs.getFloatSemantics();
    const auto &rhsSema = rhs.getFloatSemantics();

    // Only promote when some IEEE float is involved.
    if (!APFloat::isIEEELikeFP(lhsSema) || !APFloat::isIEEELikeFP(rhsSema))
        return {};

    // Find the smallest IEEE float that fits both.
    return findIEEEFloat(lhs.getContext(), [&](const llvm::fltSemantics &sema) {
        return ::isSubtype(lhsSema, sema) && ::isSubtype(rhsSema, sema);
    });
}

auto TypeSystem::promote(FloatType lhs, ekl::IntegerType rhs) const -> FloatType
{
    assert(lhs && rhs);

    const auto &rhsSema = lhs.getFloatSemantics();

    // Only promote when some IEEE float is involved.
    if (!APFloat::isIEEELikeFP(rhsSema)) return {};

    // Find the smallest IEEE float that fits both.
    return findIEEEFloat(lhs.getContext(), [&](const llvm::fltSemantics &sema) {
        return ::isSubtype(rhs, sema) && ::isSubtype(rhsSema, sema);
    });
}

auto TypeSystem::promote(ekl::IntegerType lhs, ekl::IntegerType rhs) const
    -> ekl::IntegerType
{
    using std::max;

    assert(lhs && rhs);

    if (lhs.isSigned() ^ rhs.isSigned()) {
        // If the signs differ, the result must be signed, and the unsigned max
        // value demands an extra bit.
        return ekl::IntegerType::get(
            lhs.getContext(),
            max(lhs.getWidth() + (lhs.isUnsigned() ? 1U : 0U),
                rhs.getWidth() + (rhs.isUnsigned() ? 1U : 0U)),
            true);
    }

    // If the signs are the same, we can just find the maximum bit width.
    return ekl::IntegerType::get(
        lhs.getContext(),
        max(lhs.getWidth(), rhs.getWidth()),
        lhs.isSigned());
}

auto TypeSystem::promote(Type lhs, Type rhs) const -> Type
{
    const auto get = [&]<class T>(T) {
        using std::swap;
        if (llvm::isa<T>(rhs)) {
            swap(lhs, rhs);
            return llvm::cast<T>(lhs);
        }
        return llvm::dyn_cast<T>(lhs);
    };

    // T |_| T = T
    if (lhs == rhs) return lhs;
    // Expr |_| T = Expr
    if (get(ExpressionType{})) return lhs;

    // Rational |_| Float = Rational |_| Integer = Rational |_| Index = Rational
    if (get(RationalType{})) return llvm::isa<NumberType>(rhs) ? lhs : Type{};

    // Float |_| T = ...
    if (const auto lhsFloat = get(FloatType{}); lhsFloat) {
        return llvm::TypeSwitch<Type, Type>(rhs)
            .Case(
                [&](FloatType rhsFloat) { return promote(lhsFloat, rhsFloat); })
            .Case([&](ekl::IntegerType rhsInt) {
                return promote(lhsFloat, rhsInt);
            })
            .Default({});
    }

    // Integer |_| T = ...
    if (const auto lhsInt = get(ekl::IntegerType{}); lhsInt) {
        return llvm::TypeSwitch<Type, Type>(rhs)
            .Case([&](ekl::IntegerType rhsInt) {
                return promote(lhsInt, rhsInt);
            })
            .Default({});
    }

    // Index |_| T = ...
    if (const auto lhsIndex = get(ekl::IndexType{}); lhsIndex) {
        return llvm::TypeSwitch<Type, Type>(rhs)
            .Case([&](ekl::IndexType rhsIndex) {
                return promote(lhsIndex, rhsIndex);
            })
            .Default({});
    }

    // Array |_| T = ...
    if (const auto lhsArray = get(ekl::ArrayType{}); lhsArray) {
        return llvm::TypeSwitch<Type, Type>(rhs)
            .Case(
                [&](ArrayType rhsArray) { return promote(lhsArray, rhsArray); })
            .Default({});
    }

    return {};
}

auto TypeSystem::promote(OpBuilder &, Location, Value input, Type superTy) const
    -> Operation *
{
    assert(input && superTy);
    assert(isSubtype(input.getType(), superTy));

    // TODO: Implement PromoteOp.
    return nullptr;
}
