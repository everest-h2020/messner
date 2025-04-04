/// Provides helper functions to initialize the DialectRegistry.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/IR/Base.h"

#include <mlir/IR/MLIRContext.h>

namespace messner {

/// Registers all dialects added by messner at @p registry .
inline void registerAllDialects(mlir::DialectRegistry &registry)
{
    registry.insert<mlir::ref::RefDialect>();
}

/// Registers all extensions added by messner at @p registry .
inline void registerAllExtensions(mlir::DialectRegistry &) {}

/// Registers all passes added by messner at @p registry .
inline void registerAllPasses(mlir::DialectRegistry &) {}

/// Registers all translations added by messner at @p registry .
inline void registerAllTranslations(mlir::DialectRegistry &) {}

} // namespace messner
