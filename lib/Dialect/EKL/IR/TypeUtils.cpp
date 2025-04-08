/// Implementation of the EKL dialect type utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/TypeUtils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// hasConcreteType
//===----------------------------------------------------------------------===//

auto mlir::ekl::hasConcreteType(Operation *op) -> bool
{
    assert(op);

    if (!hasConcreteType(op->getOperandTypes())
        || !hasConcreteType(op->getResultTypes()))
        return false;

    for (auto &region : op->getRegions()) {
        for (auto &block : region)
            if (!hasConcreteType(block.getArgumentTypes())) return false;
    }

    return true;
}
