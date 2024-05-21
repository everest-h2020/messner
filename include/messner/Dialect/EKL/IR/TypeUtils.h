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
concept broadcast_type_constrait = std::is_same_v<BroadcastType, T> || std::is_base_of_v<BroadcastType, T>;

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
    if (llvm::isa_and_present<ScalarType>(type)) return ExtentRange{};
    if (const auto contiguousTy =
            llvm::dyn_cast_if_present<ContiguousType>(type))
        return contiguousTy.getExtents();
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

} // namespace mlir::ekl
