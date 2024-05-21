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

/// Adds the ekl-decay-number patterns to @p patterns .
///
/// @param  [in,out]    patterns    RewritePatternSet.
void populateDecayNumberPatterns(RewritePatternSet &patterns);

/// Constructs the ekl-decay-number pass.
std::unique_ptr<Pass> createDecayNumberPass();

/// Adds the ekl-homogenize patterns to @p patterns .
///
/// @param  [in,out]    patterns    RewritePatternSet.
void populateHomogenizePatterns(RewritePatternSet &patterns);

/// Constructs the ekl-homogenize pass.
std::unique_ptr<Pass> createHomogenizePass();

/// Adds the ekl-lower patterns to @p patterns .
///
/// @param  [in,out]    patterns    RewritePatternSet.
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
