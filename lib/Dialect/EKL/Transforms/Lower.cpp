/// Implements the LowerPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-lower"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_LOWER
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct LowerPass : ekl::impl::LowerBase<LowerPass> {
    using LowerBase::LowerBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateLowerPatterns implementation
//===----------------------------------------------------------------------===//

namespace {

struct Hoist : RewritePattern {
    explicit Hoist(MLIRContext *context)
            : RewritePattern(MatchAnyOpTypeTag{}, PatternBenefit(100), context)
    {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &) const final
    {
        return llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](IfOp op) { return visit(op); })
            .Case([&](AssocOp op) { return visit(op); })
            .Case([&](ZipOp op) { return visit(op); })
            .Case([&](ReduceOp op) { return visit(op); })
            .Default(failure());
    }

private:
    LogicalResult visit(IfOp op) const
    {
        return success(
            succeeded(visit(op.getThenBranch()))
            || succeeded(visit(op.getElseBranch())));
    }
    LogicalResult visit(AssocOp op) const { return visit(op.getMap()); }
    LogicalResult visit(ZipOp op) const { return visit(op.getCombinator()); }
    LogicalResult visit(ReduceOp op) const
    {
        return success(
            succeeded(visit(op.getMap()))
            || succeeded(visit(op.getReduction())));
    }

    LogicalResult visit(Region &functor) const
    {
        const auto moved = moveLoopInvariantCode(
            {&functor},
            [&](Value value, Region *region) -> bool {
                return value.getParentRegion()->isProperAncestor(region);
            },
            [&](Operation *op, Region *) -> bool {
                return isMemoryEffectFree(op) && isSpeculatable(op);
            },
            [&](Operation *op, Region *region) {
                op->moveBefore(region->getParentOp());
            });
        return success(moved > 0UL);
    }
};

struct EliminateIf : OpRewritePattern<IfOp> {
    using OpRewritePattern<IfOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        // Only applies to expression-style ifs.
        if (!op.getResult())
            return rewriter.notifyMatchFailure(op, "only applies to if-expr");

        // Only applies if both branches are known to yield the same thing.
        const auto value = op.getThenBranch().front().back().getOperand(0);
        if (value != op.getElseBranch().front().back().getOperand(0))
            return rewriter.notifyMatchFailure(
                op,
                "only applies to equal expressions");

        // Short-circuit the result of the if.
        rewriter.replaceAllUsesWith(op.getResult(), value);
        return success();
    }
};

struct DissolveIf : OpRewritePattern<IfOp> {
    using OpRewritePattern<IfOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        // Determine if there is a statically-known code path.
        BoolAttr condition;
        if (!matchPattern(op.getCondition(), m_Constant(&condition)))
            return rewriter.notifyMatchFailure(
                op,
                "only applies to known conditionals");

        // Inline the statically-known code path before the IfOp.
        auto *branch = condition.getValue() ? &op.getThenBranch().front()
                                            : &op.getElseBranch().front();
        auto yieldOp = llvm::cast<YieldOp>(&branch->back());
        rewriter.inlineBlockBefore(branch, op);

        // Since the yield type may be a subtype of the result type, unify.
        rewriter.setInsertionPoint(yieldOp);
        const auto result = rewriter
                                .create<UnifyOp>(
                                    yieldOp->getLoc(),
                                    yieldOp.getValue(),
                                    op.getType(0))
                                .getResult();

        // Erase the extraneous terminator and throw away the rest of the IfOp.
        yieldOp.erase();
        rewriter.replaceOp(op, result);
        return success();
    }
};

} // namespace

void mlir::ekl::populateLowerPatterns(RewritePatternSet &patterns)
{
    // Hoisting applies to all functors and is essential for other rewrites.
    patterns.add<Hoist>(patterns.getContext());

    patterns.add<EliminateIf, DissolveIf>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerPass implementation
//===----------------------------------------------------------------------===//

void LowerPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateLowerPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createLowerPass()
{
    return std::make_unique<LowerPass>();
}
