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

/// Perform type checking for @p root and all of its descendants.
///
/// @pre    `root`
///
/// @param              root    Operation.
///
/// @return Whether type checking suceeded.
LogicalResult typeCheck(Operation *root);

/// Constructs the ekl-type-check pass.
std::unique_ptr<Pass> createTypeCheckPass();

/// Adds the ekl-lower patterns to @p patterns .
///
/// @param  [out]       patterns    RewritePatternSet.
void populateLowerPatterns(RewritePatternSet &patterns);

/// Construct the ekl-lower pass.
std::unique_ptr<Pass> createLowerPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl
