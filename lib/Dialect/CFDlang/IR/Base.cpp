/** Implements the CFDlang dialect base.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Base.h"

#include "mlir/Dialect/CFDlang/IR/Ops.h"

using namespace mlir;
using namespace mlir::cfdlang;

//===- Generated implementation -------------------------------------------===//

#include "mlir/Dialect/CFDlang/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CFDlangDialect
//===----------------------------------------------------------------------===//

Operation *CFDlangDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location loc
)
{
    if (auto symbolRef = value.dyn_cast<SymbolRefAttr>()) {
        return builder.create<EvalOp>(loc, type, symbolRef);
    }

    // TODO: Implement.
    return nullptr;
}

void CFDlangDialect::initialize()
{
    // Delegate to the registry methods.
    /*registerAttributes();*/
    registerTypes();
    registerOps();
}
