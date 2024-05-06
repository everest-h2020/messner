/// Implements types and utilities for working with aggregate value extents.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Extents.h"

using namespace mlir;
using namespace mlir::ekl;

FailureOr<extent_t> mlir::ekl::flatten(ExtentRange extents)
{
    extent_t result = 1UL;
    for (auto extent : extents)
        if (__builtin_mul_overflow(result, extent, &result)) [[unlikely]]
            return failure();

    return FailureOr<extent_t>(result);
}

LogicalResult
mlir::ekl::broadcast(SmallVectorImpl<extent_t> &lhs, ExtentRange rhs)
{
    // Handle scalar broadcasting.
    if (lhs.empty()) {
        lhs.assign(rhs.begin(), rhs.end());
        return success();
    }
    if (rhs.empty()) return success();

    // Handle array broadcasting.
    if (lhs.size() != rhs.size()) return failure();
    for (auto &&[l, r] : llvm::zip_equal(lhs, rhs)) {
        const auto lr = broadcast(l, r);
        if (failed(lr)) return failure();
        l = *lr;
    }
    return success();
}
