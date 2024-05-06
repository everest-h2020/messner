/// Implements the LocalTypeChecker.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/LocalTypeChecker.h"

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// LocalTypeChecker implementation
//===----------------------------------------------------------------------===//

LogicalResult LocalTypeChecker::refineBound(Expression expr, Type incoming)
{
    // Determine the legality of the refinement.
    const auto present = getType(expr);
    if (ekl::refineBound(present, incoming) != RefinementResult::Illegal)
        return success();

    // Report this illegal refinement.
    auto [diag, isError] = report(expr);
    diag << present << " is not a subtype of deduced type " << incoming;
    return success(!isError);
}

LogicalResult LocalTypeChecker::meetBound(Expression expr, Type incoming)
{
    // Determine the legality of the meet.
    const auto present = getType(expr);
    if (ekl::meetBound(present, incoming) != RefinementResult::Illegal)
        return success();

    // Report this illegal meet.
    auto [diag, isError] = report(expr);
    diag << present << " is not the same as deduced type " << incoming;
    return success(!isError);
}

std::pair<InFlightDiagnostic, bool>
LocalTypeChecker::report(Expression expr) const
{
    bool isError;
    const auto raise = [&](Operation *at) -> InFlightDiagnostic {
        isError = getParent() == at;
        return isError ? at->emitOpError() : at->emitWarning();
    };

    // NOTE: InFlightDiagnostic deletes the move-assignment operator, so we'll
    //       cheat a bit to achieve the same thing.
    std::optional<InFlightDiagnostic> diag;

    // Raise the diagnostic at the owner of the expression and describe where
    // that is in human readable terms.
    if (const auto argument = llvm::dyn_cast<BlockArgument>(expr)) {
        diag.emplace(raise(argument.getOwner()->getParentOp()));
        diag->attachNote(argument.getLoc())
            << "for argument #" << argument.getArgNumber() << " of region #"
            << argument.getOwner()->getParent()->getRegionNumber();
    } else {
        const auto result = llvm::cast<OpResult>(expr);
        diag.emplace(raise(result.getOwner()));
        diag->attachNote() << "for result #" << result.getResultNumber();
    }

    if (!isError) {
        // Indicate who is responsible for generating this warning.
        diag->attachNote(getParent().getLoc()) << "caused by this op";
    }

    return std::make_pair(std::move(*diag), isError);
}
