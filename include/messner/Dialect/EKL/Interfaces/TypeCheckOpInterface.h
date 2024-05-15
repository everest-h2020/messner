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
/// @param  [in]        op              Operation to verify.
///
/// @return Whether verification succeeded.
LogicalResult verifyTypeCheckOpInterface(Operation *op);

} // namespace mlir::ekl::impl

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h.inc"

//===----------------------------------------------------------------------===//
