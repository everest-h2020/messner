/// Implementation of the EKL dialect semantic analysis.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeSystem.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Typing/Contradiction.h>
#include <mlir/Typing/TypeChecker.h>

using namespace mlir;
using namespace mlir::Typing;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

auto FuncOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    if (!isExternal()) return success();

    // There may be at most one result value.
    if (getResultTypes().size() > 1) {
        diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "FFI functions can have at most one result";
    }

    // All arguments and the result must have an ABIType.
    for (const auto ty : getFunctionType().getInputs()) {
        if (!llvm::isa<ABIType>(ty)) {
            auto &diag =
                diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "FFI function arguments must have ABI types";
            diag.attachNote() << ty << " is not an ABI type";
        }
    }
    if (!getResultTypes().empty() && !llvm::isa<ABIType>(getResultTypes()[0])) {
        auto &diag =
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "FFI function results must have ABI types";
        diag.attachNote() << getResultTypes()[0] << " is not an ABI type";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

auto KernelOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    // All arguments must have an ABIType.
    for (const auto &arg : getArguments()) {
        if (!llvm::isa<ABIType>(arg.getType())) {
            auto &diag =
                diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "kernel arguments must have ABI types";
            diag.attachNote(arg.getLoc())
                << arg.getType() << " is not an ABI type";
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// StaticOp implementation
//===----------------------------------------------------------------------===//

auto StaticOp::checkSemantics(SmallVectorImpl<Diagnostic> &diagnostics)
    -> LogicalResult
{
    // Scalable array references must be imports.
    if (getType().isScalable()) {
        if (!isImported())
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "can't define static scalable arrays";
    }

    // Public symbols must have initializers.
    if (isPublic() && isDeclaration())
        diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
            << "can't export a static declaration";

    // The initializer must be assignable to the declared type.
    if (const auto maybeInit = getInitializer(); maybeInit) {
        if (!getTypeSystem(getContext())
                 .isSubtype(
                     maybeInit->getArrayType(),
                     getType().getCellType())) {
            diagnostics.emplace_back(getLoc(), DiagnosticSeverity::Error)
                << "can't initialize a variable of type "
                << getType().getCellType() << " with a value of type "
                << maybeInit->getArrayType();
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// YieldOp implementation
//===----------------------------------------------------------------------===//

auto YieldOp::typeCheck(AbstractTypeChecker &typeChecker)
    -> std::optional<Contradiction>
{
    // The YieldOp is allowed to invalidate its parent so that it can adjust its
    // result type based on the result of its functor regions. Bounded execution
    // is guaranteed when no parent deduces block argument types based on its
    // functor result types.
    typeChecker.invalidate((*this)->getParentOp());
    return {};
}

//===----------------------------------------------------------------------===//
// PromoteOp implementation
//===----------------------------------------------------------------------===//

auto PromoteOp::typeCheck(AbstractTypeChecker &) -> std::optional<Contradiction>
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

auto BroadcastOp::typeCheck(AbstractTypeChecker &)
    -> std::optional<Contradiction>
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

auto CoerceOp::typeCheck(AbstractTypeChecker &) -> std::optional<Contradiction>
{
    // TODO: Implement.
    return {};
}
