/** Declaration of the TeIL dialect base.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Concepts/Concepts.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/TeIL/Enums.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include <cstddef>
#include <cstdint>

namespace mlir::teil {

using namespace mlir::concepts;

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//
//
// The usage of signed/unsigned types throughout MLIR is somewhat inconsistent
// when it comes to values with natural number semantics:
//
//   * Where appropriate, to stay in line with the standard library, std::size_t
//     and thus unsigned types are used to represent sizes and counts.
//
//   * Natural number values defined by the MLIR language specification such as
//     values of IndexType are represented as signed integers to avoid issues
//     with comparison and indicate undefined overflow behavior.
//
//   * Natural number values required by dialects, such as Linalg's references
//     to dimensions, are represented as unsigned integers in line with their
//     actual semantics.
//
// TeIL represents natural numbers using unsigned values wherever their use is
// purely internal, and sticks to the existing conventions where values are
// passed to the MLIR API.
//

/** Type that stores a value of mlir::IndexType.
 *
 * @note    Although index values are conceptually unsigned, all usages in
 *          MLIR are signed int64_t values (e.g., Builder::getIndexAttr()), and
 *          therefore this type is also signed.
 */
using index_t = std::int64_t;
/** Type that stores a value of teil::NatType. */
using natural_t = std::uint64_t;
/** Type that stores the size of a dimension.
 *
 * @note    Dynamic dimension sizes are indicated using the reserved value
 *          ShapedType::kDynamicSize, which requires a signed integer type.
 */
using dim_size_t = std::decay_t<decltype(ShapedType::kDynamicSize)>;
/** Type that represents an atom shape.
 *
 * @warning Commonly used throughout MLIR, shapes are passed as non-owning
 *          references to arrays of dim_size_t, which means that shape_t does
 *          not declare a storage for the actual dimension sizes!
 */
using shape_t = ArrayRef<dim_size_t>;
/** Type that stores the rank of a shape.
 *
 * @note    Ranks are natural numbers and should be implicitly convertible from
 *          and to the size of a shape, which is why rank_t is tied to shape_t.
 */
using rank_t = shape_t::size_type;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//
//
// For the sake of convenience, to maintain type correctness, and to clarify
// intent, special values of the above types are redefined as constants within
// the teil namespace.
//

/** Value that indicates the size of a dimension is determined at runtime. */
inline constexpr dim_size_t dynamic_size = ShapedType::kDynamicSize;
/** Constant representing the shape of a scalar value. */
inline constexpr shape_t scalar_shape{};
/** Constant representing the rank of a scalar value. */
inline constexpr rank_t scalar_rank{0}; // = scalar_shape.size()

} // namespace mlir::teil

//===- Generated includes -------------------------------------------------===//

#include "mlir/Dialect/TeIL/IR/Base.h.inc"

//===----------------------------------------------------------------------===//

