/// Declares the EKL passes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ekl {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Constructs the ekl-type-check pass.
std::unique_ptr<Pass> createTypeCheckPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl
