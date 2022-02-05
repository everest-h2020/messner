/** Declaration of the CFDlang dialect interfaces.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/Concepts/Atom.h"
#include "mlir/Dialect/CFDlang/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::cfdlang {
namespace interface_defaults {

LogicalResult verifyDefinitionOp(Operation *op);

} // namespace interface_defaults

} // namespace mlir::cfdlang

//===- Generated includes -------------------------------------------------===//

//#include "mlir/Dialect/CFDlang/IR/AttrInterfaces.h.inc"
#include "mlir/Dialect/CFDlang/IR/OpInterfaces.h.inc"
//#include "mlir/Dialect/CFDlang/IR/TypeInterfaces.h.inc"

//===----------------------------------------------------------------------===//
