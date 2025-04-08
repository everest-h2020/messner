/// Declares the shape utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"

#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

namespace mlir::ekl {

/// Type that stores an array shape.
using Shape        = llvm::SmallVector<Extent>;
/// Mutable builder for a Shape.
using ShapeBuilder = llvm::SmallVectorImpl<Extent>;
/// Immutable reference to a Shape.
using ShapeRef     = llvm::ArrayRef<Extent>;

/// Determines whether @p shape is bounded.
///
/// Checks whether any extent in @p shape is unbounded.
auto isBounded(ShapeRef shape) -> bool;

/// Determines whether @p shape is known to have 0 elements.
///
/// Checks whether any extent in @p shape is 0.
auto isTriviallyEmpty(ShapeRef shape) -> bool;

/// Flattens @p shape into a single extent.
///
/// The extent of the flattened value is the product of the @p shape . If
/// @p shape is empty, the result is defined to be @c 1 UL . In other words,
/// the result of this function is the number of elements in @p shape .
///
/// @retval failure Result is out of range.
/// @retval Extent  Extent of the flattened shape.
auto flatten(ShapeRef shape) -> FailureOr<Extent>;

/// Appends @p rhs to the end of @p lhs .
///
/// @param  [in,out]    lhs Prefix and result.
/// @param              rhs Suffix.
void concat(ShapeBuilder &lhs, ShapeRef rhs);

/// Concatenates @p lhs and @p rhs into a single Shape.
auto concat(ShapeRef lhs, ShapeRef rhs) -> Shape;

/// Tries to broadcast the extents @p lhs and @p rhs together.
///
/// Two bounded extents can be broadcast together to the maximum of both iff
/// they are either the same, or one of them is @c 1UL .
///
/// @retval failure     @p lhs and @p rhs are not broadcast-compatible.
/// @retval extent_t    The resulting extent.
auto broadcast(Extent lhs, Extent rhs) -> FailureOr<Extent>;

/// Tries to broadcast the shapes @p lhs and @p rhs together.
///
/// Two shapes are broadcast together by broadcasting their extents together
/// pairwise, using broadcast(Extent, Extent). The shapes can only be broadcast
/// together if they are bounded and have the same number of extents.
///
/// @param  [in,out]    lhs First extent tuple and result.
/// @param              rhs Second extent tuple.
///
/// @retval failure     @p lhs and @p rhs are not broadcast-compatible.
/// @retval success     @p lhs contains the broadcasted result.
auto broadcast(ShapeBuilder &lhs, ShapeRef rhs) -> LogicalResult;

/// Tries to broadcast @p lhs and @p rhs together.
///
/// See broadcast(ShapeBuilder &, ShapeRef) for more information.
auto broadcast(ShapeRef lhs, ShapeRef rhs) -> FailureOr<Shape>;

} // namespace mlir::ekl

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// isBounded
//===----------------------------------------------------------------------===//

inline auto isBounded(ShapeRef shape) -> bool
{
    return std::find(shape.begin(), shape.end(), unbounded) != shape.end();
}

//===----------------------------------------------------------------------===//
// isTriviallyEmpty
//===----------------------------------------------------------------------===//

inline auto isTriviallyEmpty(ShapeRef shape) -> bool
{
    return std::find(shape.begin(), shape.end(), Extent{0}) != shape.end();
}

//===----------------------------------------------------------------------===//
// concat
//===----------------------------------------------------------------------===//

inline void concat(ShapeBuilder &lhs, ShapeRef rhs)
{
    const auto offset = lhs.size();
    lhs.resize_for_overwrite(offset + rhs.size());
    std::copy(rhs.begin(), rhs.end(), lhs.data() + offset);
}

inline auto concat(ShapeRef lhs, ShapeRef rhs) -> Shape
{
    Shape result;
    result.resize_for_overwrite(lhs.size() + rhs.size());
    std::copy(
        rhs.begin(),
        rhs.end(),
        std::copy(lhs.begin(), lhs.end(), result.data()));
    return result;
}

//===----------------------------------------------------------------------===//
// broadcast
//===----------------------------------------------------------------------===//

inline auto broadcast(Extent lhs, Extent rhs) -> FailureOr<Extent>
{
    if (!lhs.isBounded() || !rhs.isBounded()) return failure();
    if (lhs == 1 || lhs == rhs) return rhs;
    if (rhs == 1) return lhs;
    return failure();
}

inline auto broadcast(ShapeRef lhs, ShapeRef rhs) -> FailureOr<Shape>
{
    Shape result(lhs);
    if (failed(broadcast(result, rhs))) return failure();
    return success(std::move(result));
}

} // namespace mlir::ekl
