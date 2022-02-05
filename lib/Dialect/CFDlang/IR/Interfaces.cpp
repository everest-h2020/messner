/** Implements the CFDlang dialect interfaces.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Interfaces.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cfdlang;

LogicalResult mlir::cfdlang::interface_defaults::verifyDefinitionOp(
    Operation *op
)
{
    auto definition = cast<DefinitionOp>(op);

    // Ensure that the provided value actually matches the declaration.
    auto atom = definition.getAtom();
    if (definition.getAtomType() != atom.getType()) {
        return op->emitOpError()
            << "result type " << atom.getType()
            << " does not match the declared type " << definition.getAtomType()
            << " of the symbol @" << definition.getName();
    }

    return success();
}

//===- Generated implementation -------------------------------------------===//

//#include "mlir/Dialect/CFDlang/IR/AttrInterfaces.cpp.inc"
#include "mlir/Dialect/CFDlang/IR/OpInterfaces.cpp.inc"
//#include "mlir/Dialect/CFDlang/IR/TypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//

