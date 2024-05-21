/// Implements the EKL dialect traits.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Traits.h"

#include "messner/Dialect/EKL/IR/EKL.h"

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// HasFunctors implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::ekl::impl::verifyHasFunctors(Operation *op)
{
    // Check that all regions are functors.
    for (auto &region : op->getRegions()) {
        if (region.empty()) {
            // An empty region is considered a functor.
            continue;
        }

        const auto fail = [&]() -> InFlightDiagnostic {
            auto diag = op->emitOpError() << "functor region";
            if (op->getNumRegions() > 0)
                diag << " #" << region.getRegionNumber();
            diag << " ";
            return diag;
        };

        // Arguments to functors must be expressions only.
        for (auto arg : region.getArguments()) {
            if (llvm::isa<Expression>(arg)) [[likely]]
                continue;

            auto diag = fail() << "has invalid arguments";
            diag.attachNote(arg.getLoc())
                << "see non-expression argument #" << arg.getArgNumber();
            return diag;
        }

        // Functors may have up to one block.
        assert(region.hasOneBlock());
        auto &block = region.front();
        if (block.empty()) continue;

        // Terminators of a functor must always be YieldOp.
        auto terminator = &block.back();
        if (!terminator->hasTrait<mlir::OpTrait::IsTerminator>()
            && op->hasTrait<mlir::OpTrait::NoTerminator>()) {
            // No terminator is allowed.
            continue;
        }
        if (!llvm::isa<YieldOp>(terminator)) {
            auto diag = fail() << "must be terminated by yield";
            diag.attachNote(terminator->getLoc()) << "see invalid terminator";
            return diag;
        }
    }

    return success();
}
