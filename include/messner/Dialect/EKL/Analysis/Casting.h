/// Declares the casting analyses.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Unification
//===----------------------------------------------------------------------===//

/// Attempts to unify @p lhs and @p rhs .
///
/// Two types can be unified to a supertype of both of them iff one of them is
/// the supertype of the other.
///
/// @param              lhs First type.
/// @param              rhs Second type.
///
/// @retval failure     @p lhs and @p rhs can't be unified.
/// @retval nullptr     @p lhs or @p rhs is unbounded.
/// @retval Type        Unified type of @p lhs and @p rhs .
inline FailureOr<Type> unify(Type lhs, Type rhs)
{
    if (!lhs || !rhs) return success(Type{});
    if (isSubtype(lhs, rhs)) return success(rhs);
    if (isSubtype(rhs, lhs)) return success(lhs);
    return failure();
}

/// Attempts to unify all @p types .
///
/// Unifies as many types in @p types as possible according to the rules of
/// unify(Type, Type). The caller can examine the exact set of types that are
/// not unifiable in case of failure.
///
/// @param  [in,out]    types   The types.
///
/// @retval failure     The types remaining in @p types can't be unified.
/// @retval nullptr     @p types contained the unbounded type.
/// @retval Type        Unified type of @p types .
FailureOr<Type> unify(SmallVectorImpl<Type> &types);

//===----------------------------------------------------------------------===//
// Broadcasting
//===----------------------------------------------------------------------===//

/// Attempts to broadcast @p lhs and @p rhs together.
///
/// Obtains an ArrayType with the same scalar type as @p lhs and the broadcasted
/// extents of @p lhs and @p rhs . See
/// broadcast(SmallVectorImpl<extent_t> &, ExtentRange) for more details.
///
/// @param              lhs ArrayType.
/// @param              rhs ExtentRange.
///
/// @retval failure     @p lhs and @p rhs can't be broadcast together.
/// @retval nullptr     @p lhs is @c nullptr .
/// @retval ArrayType   Broadcasted ArrayType of @p lhs and @p rhs .
inline FailureOr<ArrayType> broadcast(ArrayType lhs, ExtentRange rhs)
{
    if (!lhs) return success(ArrayType{});
    auto extents = llvm::to_vector(lhs.getExtents());
    if (failed(broadcast(extents, rhs))) return failure();
    return success(lhs.cloneWith(extents));
}

/// Broadcasts @p lhs to @p rhs .
///
/// Obtains an ArrayType using @p lhs as the scalar type and @p rhs as the
/// extents.
///
/// @param              lhs ScalarType.
/// @param              rhs ExtentRange.
///
/// @return ArrayType, or @c nullptr of @p lhs was @c nullptr .
[[nodiscard]] inline ArrayType broadcast(ScalarType lhs, ExtentRange rhs)
{
    if (!lhs) return nullptr;
    return ArrayType::get(lhs, rhs);
}

/// Attempts to broadcast @p lhs and @p rhs together.
///
/// See broadcast(ArrayType, ExtentRange) and broadcast(ScalarType, ExtentRange)
/// for more details.
///
/// @retval failure     @p lhs and @p rhs can't be broadcast together.
/// @retval nullptr     @p lhs is @c nullptr .
/// @retval ArrayType   Broadcasted ArrayType of @p lhs and @p rhs .
inline FailureOr<ArrayType> broadcast(Type lhs, ExtentRange rhs)
{
    if (!lhs) return success(ArrayType{});
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(lhs))
        return broadcast(arrayTy, rhs);
    if (const auto scalarTy = llvm::dyn_cast<ScalarType>(lhs))
        return success(broadcast(scalarTy, rhs));
    return failure();
}

/// Distinguishes between possible outcomes of broadcasting.
enum class BroadcastResult {
    // The inputs are not broadcast comaptible.
    Failure,
    // The extents were broadcast together.
    Success,
    // Some input is unbounded.
    Unbounded
};

/// Attempts to obtain the broadcasted extents of @p types .
///
/// Obtains the @p extents that are the result of broadcasting together all
/// the @p types extents. If @p types contains a non-ScalarType without extents,
/// broadcasting fails.
///
/// @param              types   The types.
/// @param  [out]       extents The result extents.
///
/// @retval Failure     @p types can't be broadcast together.
/// @retval Unbounded   @p types contained the unbounded type.
/// @retval Success     @p extents contains the broadcasted extents.
[[nodiscard]] BroadcastResult
broadcast(ArrayRef<Type> types, SmallVectorImpl<extent_t> &extents);

/// Attempts to broadcast @p type to an ArrayType.
///
/// If @p type is an ArrayType, it is returned. If @p type is a ScalarType, it
/// is wrapped in an ArrayType. If @p type is @c nullptr , it is propagated.
/// Fails otherwise.
///
/// @param              type    Type.
///
/// @retval failure     @p type can not be wrapped in an ArrayType.
/// @retval nullptr     @p type is @c nullptr .
/// @retval ArrayType   The ArrayType equivalent to @p type .
inline FailureOr<ArrayType> broadcast(type_constraint auto type)
{
    if (!type) return success(ArrayType{});
    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(type))
        return success(arrayTy);
    if (const auto scalarTy = llvm::dyn_cast<ScalarType>(type))
        return success(ArrayType::get(scalarTy));
    return failure();
}

/// Attempts to broadcast @p lhs and @p rhs together.
///
/// Broadcasting succeeds when @p lhs or @p rhs is unbounded, or both could be
/// broadcasted to the same extents.
///
/// @param  [in,out]    lhs First ArrayType.
/// @param  [in,out]    rhs Second ArrayType.
///
/// @return Whether broadcasting succeeded.
inline LogicalResult broadcast(ArrayType &lhs, ArrayType &rhs)
{
    if (!lhs || !rhs || lhs == rhs) return success();
    auto extents = llvm::to_vector(lhs.getExtents());
    if (failed(broadcast(extents, rhs.getExtents()))) return failure();
    lhs = lhs.cloneWith(extents);
    rhs = rhs.cloneWith(extents);
    return success();
}

/// Attempts to broadcast @p lhs and @p rhs together.
///
/// Broadcasting succeeds when @p lhs or @p rhs is unbounded, or both could be
/// broadcasted to the same extents. Also fails if @p rhs can't participate in
/// broadcasting.
///
/// @param  [in,out]    lhs First ArrayType.
/// @param  [in,out]    rhs Second Type.
///
/// @return Whether broadcasting succeeded.
///
/// @post   `failed(result) || !lhs || !rhs || llvm::isa<ArrayType>(rhs)`
inline LogicalResult broadcast(ArrayType &lhs, Type &rhs)
{
    auto arrayTy = broadcast(rhs);
    if (failed(arrayTy) || failed(broadcast(lhs, *arrayTy))) return failure();
    rhs = *arrayTy;
    return success();
}

/// Attempts to broadcast @p lhs and @p rhs together.
///
/// Broadcasting succeeds when @p lhs or @p rhs is unbounded, or both could be
/// broadcasted to the same extents. Also fails if @p lhs or @p rhs can't
/// participate in broadcasting.
///
/// @param  [in,out]    lhs First Type.
/// @param  [in,out]    rhs Second Type.
///
/// @return Whether broadcasting succeeded.
///
/// @post   `failed(result) || !lhs || !rhs || llvm::isa<ArrayType>(lhs)`
/// @post   `failed(result) || !lhs || !rhs || llvm::isa<ArrayType>(rhs)`
inline LogicalResult broadcast(Type &lhs, Type &rhs)
{
    auto arrayTy = broadcast(lhs);
    if (failed(arrayTy) || failed(broadcast(*arrayTy, rhs))) return failure();
    lhs = *arrayTy;
    return success();
}

/// Attempts to broadcast all @p types together.
///
/// Broadcasting requires all @p types to be broadcastable to the same extent.
///
/// @param  [in,out]    types   The types.
///
/// @return Whether broadcasting succeeded.
///
/// @post   If successful and no @c nullptr type, all @p types are ArrayType.
BroadcastResult broadcast(MutableArrayRef<Type> types);

//===----------------------------------------------------------------------===//
// Coersion
//===----------------------------------------------------------------------===//

/// Determines whether @p from can be coerced to @p to .
///
/// The following coersions are allowed:
///
///     - Any type to one of its supertypes.
///     - Any number type to any other number type.
///     - Any array to an array of the same extents and a coercible scalar type.
///     - Any scalar to a pseudo-scalar array of a coercible scalar type.
///
/// @pre    `from && to`
[[nodiscard]] bool canCoerce(Type from, Type to);

} // namespace mlir::ekl
