/** Implements the CFDlang dialect operations.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cfdlang;

//===- Generated implementation -------------------------------------------===//

#include "mlir/Dialect/CFDlang/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CFDlangDialect
//===----------------------------------------------------------------------===//

void CFDlangDialect::registerOps()
{
    addOperations<
        #define GET_OP_LIST
        #include "mlir/Dialect/CFDlang/IR/Ops.cpp.inc"
    >();
}
