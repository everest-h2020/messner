/// Declares the ekl-to-standard conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace messner {

/// Adds the ekl-to-standard pass patterns to @p patterns .
void populateConvertEKLToStandardPatterns(
    mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTEKLTOSTANDARD
#include "messner/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertEKLToStandardPass();

} // namespace messner
