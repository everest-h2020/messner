/// Implements the EKL dialect TypeCheckOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h"

#include "messner/Dialect/EKL/Analysis/LocalTypeChecker.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//

LogicalResult mlir::ekl::impl::verifyTypeCheckOpInterface(Operation *op)
{
    auto iface = llvm::cast<TypeCheckOpInterface>(op);

    // Verify the IR return types of this operation using a LocalTypeChecker.
    LocalTypeChecker typeChecker(iface);
    return iface.typeCheck(typeChecker);
}

bool mlir::ekl::impl::isFullyTyped(Operation *op)
{
    const auto isUnbounded  = [](Type type) { return !getTypeBound(type); };
    const auto hasUnbounded = [&](TypeRange types) {
        return llvm::any_of(types, isUnbounded);
    };

    if (hasUnbounded(op->getOperandTypes())) return false;
    if (hasUnbounded(op->getResultTypes())) return false;

    return true;
}
