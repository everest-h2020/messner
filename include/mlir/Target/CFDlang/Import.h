/** Declares the CFDlang import translation.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/IR/Ops.h"

#include "mlir/Concepts/Translation.h"

namespace mlir::cfdlang {

/** Register the import translation. */
void registerImport();

} // namespace mlir::cfdlang
