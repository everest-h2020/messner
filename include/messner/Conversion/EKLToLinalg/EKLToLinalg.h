/// Declares the ekl-to-linalg conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace messner {

/// Adds the ekl-to-linalg pass patterns to @p patterns .
void populateConvertEKLToLinalgPatterns(
    mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTEKLTOLINALG
#include "messner/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertEKLToLinalgPass();

} // namespace messner
