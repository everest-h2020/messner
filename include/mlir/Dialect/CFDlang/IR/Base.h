/** Declaration of the CFDlang dialect base.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Concepts.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/TeIL/IR/Base.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::cfdlang {

using namespace mlir::concepts;

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//

using teil::index_t;
using teil::natural_t;
using teil::dim_size_t;
using teil::shape_t;
using teil::rank_t;

using nat_indices_t = ArrayRef<natural_t>;

} // namespace mlir::cfdlang

//===- Generated includes -------------------------------------------------===//

#include "mlir/Dialect/CFDlang/IR/Base.h.inc"

//===----------------------------------------------------------------------===//
