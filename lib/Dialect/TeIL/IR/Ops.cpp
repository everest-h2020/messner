/** Implements the TeIL dialect operations.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/TeIL/IR/Ops.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::teil;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/TeIL/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TeILDialect
//===----------------------------------------------------------------------===//

void TeILDialect::registerOps()
{
    addOperations<
        #define GET_OP_LIST
        #include "mlir/Dialect/TeIL/IR/Ops.cpp.inc"
    >();
}
