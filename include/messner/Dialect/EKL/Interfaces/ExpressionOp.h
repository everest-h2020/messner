/// Declares the EKL ExpressionOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace mlir::ekl::detail {

/// Checks that @p op verifies according to ExpresionOp::verifyUses.
///
/// @param  [in]        op              Operation to verify.
///
/// @return Whether verification succeeded.
LogicalResult verifyExpressionOp(Operation *op);

} // namespace mlir::ekl::detail

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/ExpressionOp.h.inc"

//===----------------------------------------------------------------------===//
