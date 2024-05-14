/// Provides some utilities for working with EKL types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Types.h"

namespace mlir::ekl {

template<class T>
concept type_constraint = std::is_same_v<Type, T> || std::is_base_of_v<Type, T>;

template<class T>
concept value_constraint =
    std::is_same_v<Value, T> || std::is_base_of_v<Value, T>;

/// Gets the underlying scalar type of @p type , if any.
///
/// If @p type is a scalar matching @p ResultType , returns it. If @p type is a
/// ContiguousType over some @p ResultType , returns that type. Otherwise,
/// returns @c nullptr .
///
/// @tparam ResultType  Additional constraint on the scalar type.
///
/// @param              type    The type.
///
/// @retval nullptr     @p type is neither a scalar nor an aggregate.
/// @retval ResultType  The scalar type of @p type .
template<type_constraint ResultType = ScalarType>
[[nodiscard]] inline ResultType getScalarType(type_constraint auto type)
{
    if (!type) return nullptr;
    if (const auto resultTy = llvm::dyn_cast<ResultType>(type)) return resultTy;
    if (const auto contiguousTy = llvm::dyn_cast<ContiguousType>(type))
        return llvm::dyn_cast<ResultType>(contiguousTy.getScalarType());
    return nullptr;
}

/// Gets the aggregate extents of @p type , if any.
///
/// If @p type is a ContiguousType, returns its extents. If @p type is a
/// ScalarType, returns the empty ExtentRange. Otherwise, fails.
///
/// @param              type    The type.
///
/// @retval failure     @p type is not contiguous or scalar.
/// @retval ExtentRange The extents of @p type .
inline FailureOr<ExtentRange> getExtents(type_constraint auto type)
{
    if (const auto contiguousTy =
            llvm::dyn_cast_if_present<ContiguousType>(type))
        return contiguousTy.getExtents();
    if (llvm::isa_and_present<ScalarType>(type)) return ExtentRange{};
    return failure();
}

/// Gets the type bound on @p type .
///
/// If @p type is an ExpressionType, returns its type bound. Otherwise, returns
/// @p type .
///
/// @param  [in]        type    The type.
///
/// @return The type bound on @p type .
[[nodiscard]] inline Type getTypeBound(type_constraint auto type)
{
    if (const auto exprTy = llvm::dyn_cast_if_present<ExpressionType>(type))
        return exprTy.getTypeBound();
    return type;
}
/// Gets the type bound on @p value .
///
/// If @p value has an ExpressionType, returns its type bound. Otherwise,
/// returns the type of @p value .
///
/// @param  [in]        value   The value.
///
/// @return The type bound on @p value .
[[nodiscard]] inline Type getTypeBound(value_constraint auto value)
{
    return getTypeBound(value.getType());
}

/// Decays @p type to a ScalarType if possible.
///
/// If @p type is an ArrayType with no extents, returns its scalar type.
/// Otherwise, returns @p type .
///
/// @param              type    The type.
///
/// @return The scalar type of @p type if it has no extents, or @p type .
[[nodiscard]] inline Type decayToScalar(type_constraint auto type)
{
    if (const auto arrayTy = llvm::dyn_cast_if_present<ArrayType>(type))
        if (arrayTy.isScalar()) return arrayTy.getScalarType();
    return type;
}

/// Applies the aggregate extents of @p from to @p to .
///
/// If @p from is an ArrayType, returns the ArrayType over @p to with the same
/// extents as @p from . Otherwise, returns @p to .
///
/// @param              from    The source type.
/// @param              to      The target type.
///
/// @return @p to or an ArrayType over @p to with the extents of @p from .
[[nodiscard]] inline Type applyExtents(type_constraint auto from, ScalarType to)
{
    if (const auto arrayTy = llvm::dyn_cast_if_present<ArrayType>(from))
        return arrayTy.cloneWith(to);
    return to;
}

/// Applies the reference kind of @p from to @p to .
///
/// If @p from is a ReferenceType, returns the ReferenceType to @p to with the
/// same kind as @p from . Otherwise, returns @p to .
///
/// @param              from    The source type.
/// @param              to      The target type.
///
/// @return @p to or a ReferenceType to @p to with the kind of @p from .
[[nodiscard]] inline Type
applyReference(type_constraint auto from, ArrayType to)
{
    if (const auto refTy = llvm::dyn_cast_if_present<ReferenceType>(from))
        return refTy.cloneWith(to);
    return to;
}

} // namespace mlir::ekl
