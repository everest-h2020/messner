/** Declaration of the CFDlang dialect ops.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/IR/Interfaces.h"
#include "mlir/Dialect/CFDlang/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CFDlang/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//
