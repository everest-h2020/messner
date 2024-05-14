/// Declares types and utilities for working with aggregate value extents.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir::ekl {

/// Type that holds a constant array extent value.
///
/// The storage type for an array extent must be target-independent. We follow
/// the design rationale of the `index` dialect, requiring that the platform
/// type is no wider than 64 bits, and thus folding non-overflowing operations
/// with 64 bits of precision is always legal.
///
/// Contrary to MLIR built-ins, we use an unsigned integer type for consistency
/// with the intended semantics, and to have portable operational semantics.
using extent_t = std::uint64_t;

/// The largest value of the extent_t type.
static constexpr auto max_extent = std::numeric_limits<extent_t>::max();

/// Reference to a contiguous range of extent_t elements.
using ExtentRange = llvm::ArrayRef<extent_t>;

/// Determines whether @p extents contains an empty extent.
///
/// @param              extents ExtentRange.
///
/// @return Whether @p extents contains @c 0 .
[[nodiscard]] inline bool hasNoElements(ExtentRange extents)
{
    return llvm::is_contained(extents, 0UL);
}

/// Determines whether @p extents describes a scalar value.
///
/// @param              extents ExtentRange.
///
/// @return Whether @p extents is empty.
[[nodiscard]] inline bool isScalar(ExtentRange extents)
{
    return extents.empty();
}

/// Flattens @p extents into a single extent value.
///
/// The extent of the flattened value is the product of the @p extents . If
/// @p extents is empty, the result is defined to be @c 1 UL . In other words,
/// the result of this function is the number of elements in @p extents .
///
/// @param              extents ExtentRange.
///
/// @retval failure     Result is out of range of extent_t.
/// @retval extent_t    Extent of the flattened value.
FailureOr<extent_t> flatten(ExtentRange extents);

/// Appends @p rhs to the end of @p lhs .
///
/// @param  [in,out]    lhs Prefix and result.
/// @param              rhs Suffix.
inline void concat(SmallVectorImpl<extent_t> &lhs, ExtentRange rhs)
{
    const auto offset = lhs.size();
    lhs.resize_for_overwrite(offset + rhs.size());
    std::copy(rhs.begin(), rhs.end(), lhs.data() + offset);
}

/// Concatenates @p lhs and @p rhs .
///
/// @param              lhs Prefix.
/// @param              rhs Suffix.
///
/// @return A sequence of @p lhs and @p rhs .
inline SmallVector<extent_t> concat(ExtentRange lhs, ExtentRange rhs)
{
    SmallVector<extent_t> result;
    result.resize_for_overwrite(lhs.size() + rhs.size());
    std::copy(
        rhs.begin(),
        rhs.end(),
        std::copy(lhs.begin(), lhs.end(), result.data()));
    return result;
}

/// Attempts to broadcast the extents @p lhs and @p rhs together.
///
/// Two extents can be broadcast together to the maximum of both iff they are
/// either the same, or one of them is @c 1UL .
///
/// @param              lhs First extent_t.
/// @param              rhs Second extent_t.
///
/// @retval failure     @p lhs and @p rhs are not broadcast-compatible.
/// @retval extent_t    The resulting extent.
inline FailureOr<extent_t> broadcast(extent_t lhs, extent_t rhs)
{
    if (lhs == 1 || lhs == rhs) return rhs;
    if (rhs == 1) return lhs;
    return failure();
}

/// Attempts to broadcast the extent tuples @p lhs and @p rhs together.
///
/// Two extent tuples are broadcast together by broadcasting their extents
/// together pairwise, using broadcast(extent_t, extent_t). The tuples can only
/// be broadcast together if they have the same size, unless either is empty.
///
/// @param  [in,out]    lhs First extent tuple and result.
/// @param              rhs Second extent tuple.
///
/// @retval failure     @p lhs and @p rhs are not broadcast-compatible.
/// @retval success     @p lhs contains the broadcasted result.
LogicalResult broadcast(SmallVectorImpl<extent_t> &lhs, ExtentRange rhs);

} // namespace mlir::ekl
