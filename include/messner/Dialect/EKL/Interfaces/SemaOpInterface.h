/// Declaration of the EKL SemaOpInterface interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Typing/TypeCheckOpInterface.h>

namespace mlir::ekl {

/// Verifies the SemaOpInterface implementation of @p op .
///
/// @pre    `llvm::isa_and_present<SemaOpInterface>(op)`
auto verifySemaOpInterface(Operation *op) -> LogicalResult;

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/SemaOpInterface.h.inc"

//===----------------------------------------------------------------------===//
