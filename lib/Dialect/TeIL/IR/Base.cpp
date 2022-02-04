/** Implements the TeIL dialect base.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/TeIL/IR/Base.h"

using namespace mlir;
using namespace mlir::teil;

//===- Generated implementation -------------------------------------------===//

#include "mlir/Dialect/TeIL/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TeILDialect
//===----------------------------------------------------------------------===//

Operation* TeILDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location loc
)
{
    // TODO: Implement.
    return nullptr;
}

void TeILDialect::initialize()
{
    // Delegate to the registry methods.
    registerTypes();
}
