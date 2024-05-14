/// Implements the casting analyses.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Casting.h"

using namespace mlir;
using namespace mlir::ekl;

/// Tries to reduce @p types to a single value via fallible reduction @p fn .
///
/// @param  [in,out]    types   The types to reduce.
/// @param              fn      The fallible binary reduction function.
///
/// @retval failure     @p types is irreducible, but longer than @c 1 .
/// @retval nullptr     @p types is empty or @p fn produced @c nullptr .
/// @retval Type        The single type that @p types reduced to.
static FailureOr<Type> reducePairwise(SmallVectorImpl<Type> &types, auto fn)
{
    using std::swap;

    if (types.empty()) {
        // The empty worklist reduces to the unbounded type.
        return Type{};
    }

    // Reduce the list until it is irreducible.
    auto head = types.begin();
    while (types.size() > 1) {
        // Find some element to combine with the head.
        auto it = std::next(head);
        for (; it != types.end(); ++it) {
            const auto maybeUnified = fn(*head, *it);
            if (succeeded(maybeUnified)) {
                if (!*maybeUnified) {
                    // We were able to produce the unbounded type, abort.
                    return Type{};
                }

                // Update the head element and stop searching.
                *head = *maybeUnified;
                break;
            }
        }

        if (it == types.end()) {
            // We could not reduce the head with any type we have. There may be
            // a pair not involving the head that is reducible.
            if (++head == types.end()) {
                // We have tried all pairs, there isn't.
                return failure();
            }

            // The head was changed and we will now search new pairs.
            continue;
        }

        // The element at it was reduced to the new head.
        types.erase(it);

        if (head != types.begin()) {
            // We recovered from a head without reducible pairs. To avoid
            // searching all the known-irreducible pairs again, make the head
            // the first element and search from there.
            swap(types.front(), *head);
            head = types.begin();
        }
    }

    // No types we need to unify are left.
    return types.front();
}

//===----------------------------------------------------------------------===//
// Unification
//===----------------------------------------------------------------------===//

FailureOr<Type> mlir::ekl::unify(SmallVectorImpl<Type> &types)
{
    return reducePairwise(types, [](Type lhs, Type rhs) {
        return unify(lhs, rhs);
    });
}

//===----------------------------------------------------------------------===//
// Broadcasting
//===----------------------------------------------------------------------===//

BroadcastResult
mlir::ekl::broadcast(ArrayRef<Type> types, SmallVectorImpl<extent_t> &extents)
{
    if (types.empty()) {
        // The empty worklist broadcasts to the scalar shape.
        return BroadcastResult::Success;
    }

    // The result is initialized using the first type's extents.
    if (!types.front()) return BroadcastResult::Unbounded;
    const auto frontExtents = getExtents(types.front());
    if (failed(frontExtents)) return BroadcastResult::Failure;
    concat(extents, *frontExtents);

    // Broadcast the result together with all remaining type's extents.
    for (auto type : types.drop_front()) {
        if (!type) return BroadcastResult::Unbounded;
        const auto rhsExtents = getExtents(type);
        if (failed(rhsExtents) || failed(broadcast(extents, *rhsExtents)))
            return BroadcastResult::Failure;
    }

    return BroadcastResult::Success;
}

BroadcastResult mlir::ekl::broadcast(MutableArrayRef<Type> types)
{
    // Determine the extents of the broadcasted result.
    SmallVector<extent_t> extents;
    const auto result = broadcast(types, extents);
    if (result != BroadcastResult::Success) {
        // Propagate failure or unbounded.
        return result;
    }

    // Update all the types to their broadcasted type.
    for (auto &type : types) {
        if (const auto arrayTy = llvm::dyn_cast<ArrayType>(type)) {
            type = arrayTy.cloneWith(extents);
            continue;
        }

        type = ArrayType::get(llvm::cast<ScalarType>(type), extents);
    }

    return BroadcastResult::Success;
}

//===----------------------------------------------------------------------===//
// Coersion
//===----------------------------------------------------------------------===//

bool mlir::ekl::canCoerce(Type from, Type to)
{
    // Upcasting (unifcation) is trivial.
    if (isSubtype(from, to)) return true;

    // Can coerce any number type to any other number type.
    if (llvm::isa<NumericType>(from) && llvm::isa<NumericType>(to)) return true;

    // Arrays can be coerced if they have the same shape and coercible scalar
    // types.
    const auto fromArray = llvm::dyn_cast_if_present<ArrayType>(from);
    const auto toArray   = llvm::dyn_cast<ArrayType>(to);
    if (fromArray && toArray) {
        return fromArray.getExtents() == toArray.getExtents()
            && canCoerce(fromArray.getScalarType(), toArray.getScalarType());
    }

    return false;
}
