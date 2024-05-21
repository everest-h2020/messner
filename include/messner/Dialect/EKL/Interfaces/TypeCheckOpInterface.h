/// Declares the EKL TypeCheckOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace mlir::ekl::impl {

/// Uses a LocalTypeChecker to determine whether @p op is valid.
///
/// @param  [in]        op  Operation.
///
/// @return Whether verification succeeded.
LogicalResult verifyTypeCheckOpInterface(Operation *op);

/// Determines whether @p op is fully typed.
///
/// An operation is fully typed if it does not have any unbounded expression
/// operands or results.
///
/// @param  [in]        op  Operation.
///
/// @return Whether @p op is fully typed.
bool isFullyTyped(Operation *op);

} // namespace mlir::ekl::impl

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h.inc"

//===----------------------------------------------------------------------===//
