/// Implements the EKL dialect SemaOpInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Interfaces/SemaOpInterface.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/EKL/Interfaces/SemaOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//

auto mlir::ekl::verifySemaOpInterface(Operation *op) -> LogicalResult
{
    auto iface = llvm::cast<SemaOpInterface>(op);

    SmallVector<Diagnostic> diags;
    const auto result = iface.checkSemantics(diags);
    auto hasError     = false;

    auto &engine = op->getContext()->getDiagEngine();
    for (auto &diag : diags) {
        hasError |= diag.getSeverity() == DiagnosticSeverity::Error;
        engine.emit(std::move(diag));
    }

    if (failed(result) && !hasError)
        return op->emitError("semantic analysis failed");

    return success(succeeded(result) && !hasError);
}
