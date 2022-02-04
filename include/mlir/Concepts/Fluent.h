/** Provides some extensions for fluent syntax.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <type_traits>
#include <utility>

namespace mlir::concepts {

/** The default capacity of an llvm::SmallVector of @p T . */
template<class T>
inline constexpr auto small_vector_capacity =
    llvm::CalculateSmallVectorDefaultInlinedElements<T>::value;

/** Alternative to llvm::to_vector that uses the default capacity. */
template<class Range>
inline auto to_vector(Range &&range)
{
    using reference = decltype(*range.begin());
    using value_type = std::remove_cv_t<std::remove_reference_t<reference>>;
    constexpr auto capacity = small_vector_capacity<value_type>;
    return llvm::to_vector<capacity>(std::forward<Range>(range));
}

/** Asserts that @p result indicate success. */
inline void cantFail(LogicalResult result)
{
    assert(succeeded(result));
}
/** Asserts that @p results indicates success and returns the value. */
template<class T>
inline T cantFail(FailureOr<T> result)
{
    assert(succeeded(result));
    return result.getValue();
}

/** Returns the first failure of @p lhs and @p rhs . */
inline LogicalResult operator||(LogicalResult lhs, LogicalResult rhs)
{
    return failed(lhs) ? lhs : rhs;
}
/** Returns the first success of @p lhs and @p rhs . */
inline LogicalResult operator&&(LogicalResult lhs, LogicalResult rhs)
{
    return failed(lhs) ? rhs : lhs;
}

} // namespace mlir::concepts
