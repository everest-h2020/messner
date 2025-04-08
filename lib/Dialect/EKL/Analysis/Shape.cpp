/// Implements the shape utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Shape.h"

using namespace mlir;
using namespace mlir::ekl;

auto mlir::ekl::flatten(ShapeRef shape) -> FailureOr<Extent>
{
    Extent result{1U};
    for (auto &extent : shape) {
        if (extent == 0) return Extent(0);
        const auto next = result + extent;
        if (!next || !next->isBounded()) return failure();
        result = *next;
    }

    return result;
}

auto mlir::ekl::broadcast(ShapeBuilder &lhs, ShapeRef rhs) -> LogicalResult
{
    if (lhs.size() != rhs.size()) return failure();
    for (auto &&[l, r] : llvm::zip_equal(lhs, rhs)) {
        const auto lr = broadcast(l, r);
        if (failed(lr)) return failure();
        l = *lr;
    }
    return success();
}
