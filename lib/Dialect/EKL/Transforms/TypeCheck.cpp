/// Implements the TypeCheckPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Transforms/TypeCheck.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-type-check"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_TYPECHECK
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {
struct TypeCheckPass : ekl::impl::TypeCheckBase<TypeCheckPass> {
    using TypeCheckBase::TypeCheckBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// TypeChecker implementation
//===----------------------------------------------------------------------===//

LogicalResult TypeChecker::refineBound(Expression expr, Type incoming)
{
    // Attempt to refine the bound.
    const auto present = getType(expr);
    switch (ekl::refineBound(present, incoming)) {
    case RefinementResult::NoChange: return success();
    case RefinementResult::Restricted:
        LLVM_DEBUG(
            llvm::dbgs()
            << "[TypeChecker] " << expr << " => " << incoming << "\n");
        m_context[expr] = incoming;
        invalidateUsers(expr);
        return success();
    default: break;
    }

    return emitError(expr.getLoc())
        << "deduced type " << incoming << " is not a subtype of " << present;
}

LogicalResult TypeChecker::meetBound(Expression expr, Type incoming)
{
    const auto present = getType(expr);
    switch (ekl::meetBound(present, incoming)) {
    case RefinementResult::NoChange: return success();
    case RefinementResult::Restricted:
        LLVM_DEBUG(
            llvm::dbgs()
            << "[TypeChecker] " << expr << " => " << incoming << "\n");
        m_context[expr] = incoming;
        invalidateUsers(expr);
        return success();
    default: break;
    }

    return emitError(expr.getLoc())
        << "deduced type " << incoming << " is different from " << present;
}

void TypeChecker::invalidate(Operation *op)
{
    const auto expr = llvm::dyn_cast<TypeCheckOpInterface>(op);
    if (!expr) return;

    const auto [it, ok] = m_invalid.insert(expr);
    if (ok)
        LLVM_DEBUG(
            llvm::dbgs() << "[TypeChecker] invalidated " << expr << "\n");
}

LogicalResult TypeChecker::check(RewriterBase::Listener *listener)
{
    // Continue refining type bounds until a fixpoint is reached.
    auto succeeded = true;
    while (auto expr = popInvalid()) {
        // Put the expression on the diagnostic stack.
        ScopedDiagnosticHandler handler(
            expr.getContext(),
            [&](Diagnostic &diag) -> LogicalResult {
                diag.attachNote(expr.getLoc())
                    << "while type checking this expression";
                return failure();
            });

        LLVM_DEBUG(llvm::dbgs() << "[TypeChecker] checking " << expr << "\n");
        if (failed(expr.typeCheck(*this))) {
            LLVM_DEBUG(llvm::dbgs() << "[TypeChecker] failed!\n");
            succeeded = false;
        }
    }
    if (!succeeded) return failure();

    // Apply the deductions to the IR.
    applyToIR(listener);
    return success();
}

TypeCheckOpInterface TypeChecker::popInvalid()
{
    const auto it = m_invalid.begin();
    if (it == m_invalid.end()) return nullptr;

    const auto result = *it;
    m_invalid.erase(it);
    return result;
}

[[nodiscard]] Operation *getOwner(Expression expr)
{
    if (const auto result = llvm::dyn_cast<OpResult>(expr))
        return result.getOwner();
    return llvm::cast<BlockArgument>(expr).getParentRegion()->getParentOp();
}

void TypeChecker::applyToIR(RewriterBase::Listener *listener)
{
    LLVM_DEBUG(
        llvm::dbgs()
        << "[TypeChecker] applying " << m_context.size() << " deductions\n");

    for (auto [expr, bound] : m_context) {
        assert(bound);
        if (listener) listener->notifyOperationModified(getOwner(expr));
        expr.setType(ExpressionType::get(expr.getContext(), bound));
    }
    m_context.clear();
}

LogicalResult mlir::ekl::typeCheck(Operation *root)
{
    assert(root);

    TypeChecker typeChecker;

    // Put the kernel on the diagnostic stack.
    ScopedDiagnosticHandler kernelNote(
        root->getContext(),
        [&](Diagnostic &diag) -> LogicalResult {
            diag.attachNote(root->getLoc())
                << "while type checking this operation";
            return failure();
        });

    // Work is initialized by recursively invalidating all expressions.
    typeChecker.recursivelyInvalidate(root);

    return typeChecker.check();
}

//===----------------------------------------------------------------------===//
// TypeCheckPass implementation
//===----------------------------------------------------------------------===//

void TypeCheckPass::runOnOperation()
{
    if (failed(typeCheck(getOperation()))) signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createTypeCheckPass()
{
    return std::make_unique<TypeCheckPass>();
}
