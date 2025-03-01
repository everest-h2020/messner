/// Declares the ekl-to-arith conversion pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace messner {

/// Adds the ekl-to-arith pass patterns to @p patterns .
void populateConvertEKLToArithPatterns(
    mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTEKLTOARITH
#include "messner/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertEKLToArithPass();

} // namespace messner
