/// Implements the EKL dialect ExpressionOp interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Interfaces/ExpressionOp.h"

#include "messner/Dialect/EKL/Analysis/LocalTypeChecker.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/ExpressionOp.cpp.inc"

//===----------------------------------------------------------------------===//

LogicalResult mlir::ekl::detail::verifyExpressionOp(Operation *op)
{
    auto iface = llvm::cast<ExpressionOp>(op);

    // Verify the IR return types of this operation using a LocalTypeChecker.
    LocalTypeChecker typeChecker(iface);
    return iface.typeCheck(typeChecker);
}
