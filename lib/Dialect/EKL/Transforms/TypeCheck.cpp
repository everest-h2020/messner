/// Implements the TypeCheckPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

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

struct TypeChecker : AbstractTypeChecker {
    [[nodiscard]] virtual Type getType(Expression expr) const override
    {
        if (const auto deduced = m_context.lookup(expr)) return deduced;
        return getTypeBound(expr);
    }

    virtual LogicalResult refineBound(Expression expr, Type incoming) override;

    virtual LogicalResult meetBound(Expression expr, Type incoming) override;

    virtual void invalidate(Operation *op) override
    {
        const auto expr = llvm::dyn_cast<ExpressionOp>(op);
        if (!expr) return;

        const auto [it, ok] = m_invalid.insert(expr);
        if (ok)
            LLVM_DEBUG(
                llvm::dbgs() << "[TypeChecker] invalidated " << expr << "\n");
    }

    ExpressionOp popInvalid()
    {
        const auto it = m_invalid.begin();
        if (it == m_invalid.end()) return nullptr;

        const auto result = *it;
        m_invalid.erase(it);
        return result;
    }

    void applyToIR()
    {
        LLVM_DEBUG(
            llvm::dbgs() << "[TypeChecker] applying " << m_context.size()
                         << " deductions\n");

        for (auto [expr, bound] : m_context) {
            assert(bound);
            expr.setType(ExpressionType::get(expr.getContext(), bound));
        }
        m_context.clear();
    }

private:
    llvm::DenseMap<Expression, Type> m_context;
    llvm::DenseSet<ExpressionOp> m_invalid;
};

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

//===----------------------------------------------------------------------===//
// TypeCheckPass implementation
//===----------------------------------------------------------------------===//

void TypeCheckPass::runOnOperation()
{
    TypeChecker typeChecker;

    // Put the kernel on the diagnostic stack.
    ScopedDiagnosticHandler kernelNote(
        &getContext(),
        [&](Diagnostic &diag) -> LogicalResult {
            diag.attachNote(getOperation().getLoc())
                << "while type checking this kernel";
            return failure();
        });

    // Work is initialized by recursively invalidating all expressions.
    typeChecker.recursivelyInvalidate(getOperation());

    // Continue refining type bounds until a fixpoint is reached.
    while (auto expr = typeChecker.popInvalid()) {
        // Put the expression on the diagnostic stack.
        ScopedDiagnosticHandler handler(
            &getContext(),
            [&](Diagnostic &diag) -> LogicalResult {
                diag.attachNote(expr.getLoc())
                    << "while type checking this expression";
                return failure();
            });

        LLVM_DEBUG(llvm::dbgs() << "[TypeChecker] checking " << expr << "\n");
        if (failed(expr.typeCheck(typeChecker))) {
            signalPassFailure();
            return;
        }
    }

    // Apply the deductions to the IR.
    typeChecker.applyToIR();
}

std::unique_ptr<Pass> mlir::ekl::createTypeCheckPass()
{
    return std::make_unique<TypeCheckPass>();
}
