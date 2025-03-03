/// Declares the ekl-to-index conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace messner {

/// Adds the ekl-to-index pass patterns to @p patterns .
void populateConvertEKLToIndexPatterns(
    mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTEKLTOINDEX
#include "messner/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertEKLToIndexPass();

} // namespace messner
