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
            succeeded(visit(op.getThenRegion()))
            || succeeded(visit(op.getElseRegion())));
    }
    LogicalResult visit(AssocOp op) const { return visit(op.getMapRegion()); }
    LogicalResult visit(ZipOp op) const
    {
        return visit(op.getCombinatorRegion());
    }
    LogicalResult visit(ReduceOp op) const
    {
        return visit(op.getReductionRegion());
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
        const auto value = op.getThenBranch()->back().getOperand(0);
        if (value != op.getElseBranch()->back().getOperand(0))
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
        auto branch =
            condition.getValue() ? op.getThenBranch() : op.getElseBranch();
        auto yieldOp = llvm::cast<YieldOp>(branch->getTerminator());
        rewriter.inlineBlockBefore(branch, op);

        // Since the yield type may be a subtype of the result type, unify.
        rewriter.setInsertionPoint(yieldOp);
        const auto result = rewriter
                                .create<UnifyOp>(
                                    yieldOp->getLoc(),
                                    yieldOp.getExpression(),
                                    op.getType(0))
                                .getResult();

        // Erase the extraneous terminator and throw away the rest of the IfOp.
        yieldOp.erase();
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct RewriteIfToStatement : OpRewritePattern<IfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isExpression()) return failure();
        if (!op.getResult().use_empty()) return failure();

        auto result = rewriter.create<IfOp>(op.getLoc(), op.getCondition());
        rewriter.inlineBlockBefore(
            op.getThenBranch(),
            result.getThenBranch(),
            result.getThenBranch()->end());
        rewriter.eraseOp(result.getThenBranch()->getTerminator());
        rewriter.inlineBlockBefore(
            op.getElseBranch(),
            result.getElseBranch(),
            result.getElseBranch()->end());
        rewriter.eraseOp(result.getElseBranch()->getTerminator());
        rewriter.eraseOp(op);
        return success();
    }
};

struct RewriteIfToChoice : OpRewritePattern<IfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isExpression()) return failure();

        const auto trueValue   = op.getThenExpression();
        const auto falseValue  = op.getElseExpression();
        const auto definedInIf = [&](Value value) {
            if (const auto result = llvm::cast<OpResult>(value))
                return op->isAncestor(result.getOwner());
            return op->isAncestor(
                llvm::cast<BlockArgument>(value).getOwner()->getParentOp());
        };
        if (definedInIf(trueValue) || definedInIf(falseValue)) return failure();

        const auto result = rewriter
                                .create<ChoiceOp>(
                                    op.getLoc(),
                                    op.getCondition(),
                                    ValueRange{falseValue, trueValue})
                                .getResult();
        rewriter.replaceAllUsesWith(op.getResult(), result);
        return success();
    }
};

struct ExpandEllipsisSubscript : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "not fully typed");

        // Find the ellipsis operand, if any.
        const auto it = llvm::find_if(op.getSubscripts(), [](Value value) {
            return llvm::isa<EllipsisType>(
                llvm::cast<Expression>(value).getType().getTypeBound());
        });
        if (it == op.getSubscripts().end()) return failure();

        // Calculate the number of identities it should expand to.
        const auto extentExprTy = ExpressionType::get(
            rewriter.getContext(),
            ExtentType::get(rewriter.getContext()));
        // How many extents do we need to index?
        const auto numExtents =
            llvm::cast<ArrayType>(op.getArray().getType().getTypeBound())
                .getNumExtents();
        // How many extents did we insert?
        const auto numExpanded = std::size_t(
            llvm::count(op.getSubscripts().getTypes(), extentExprTy));
        const auto expand =
            numExtents - op.getSubscripts().size() + 1UL + numExpanded;

        // Create an instance of the identity literal. We use the location of
        // the ellipsis literal to track where they came from.
        const auto identity = rewriter
                                  .create<LiteralOp>(
                                      (*it).getLoc(),
                                      IdentityAttr::get(rewriter.getContext()))
                                  .getResult();
        SmallVector<Value> identities(expand, identity);

        // Replace the single ellipsis operand with that many identity literals.
        rewriter.updateRootInPlace(op, [&]() {
            op->setOperands(
                op.getSubscripts().getBeginOperandIndex(),
                1U,
                identities);
        });
        return success();
    }
};

struct RewriteSubscriptToAssoc : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        const auto resultTy =
            llvm::cast_if_present<BroadcastType>(op.getType().getTypeBound());
        if (!resultTy) return failure();
        const auto extents = resultTy.getExtents();
        if (extents.empty()) return failure();

        const auto ellExprTy =
            ExpressionType::get(getContext(), EllipsisType::get(getContext()));
        if (llvm::count(op.getSubscripts().getTypes(), ellExprTy) > 0)
            return failure();

        rewriter.replaceOpWithNewOp<AssocOp>(
            op,
            llvm::cast<ArrayType>(resultTy),
            [&](OpBuilder &builder, Location loc, ValueRange indices) {
                SmallVector<Value> newSubscripts;
                for (auto sub : op.getSubscripts()) {
                    const auto subTy = llvm::cast<ExpressionType>(sub.getType())
                                           .getTypeBound();
                    if (llvm::isa<ExtentType>(subTy)) {
                        indices = indices.drop_front();
                        continue;
                    }
                    if (llvm::isa<IdentityType>(subTy)) {
                        newSubscripts.push_back(indices.front());
                        indices = indices.drop_front();
                        continue;
                    }
                    if (const auto arrayTy = llvm::dyn_cast<ArrayType>(subTy)) {
                        newSubscripts.push_back(
                            builder
                                .create<SubscriptOp>(
                                    loc,
                                    sub,
                                    indices.take_front(arrayTy.getNumExtents()),
                                    arrayTy.getScalarType())
                                .getResult());
                        indices = indices.drop_front(arrayTy.getNumExtents());
                        continue;
                    }
                    newSubscripts.push_back(sub);
                }
                newSubscripts.append(indices.begin(), indices.end());

                const auto element = builder
                                         .create<SubscriptOp>(
                                             loc,
                                             op.getArray(),
                                             newSubscripts,
                                             resultTy.getScalarType())
                                         .getResult();
                builder.create<YieldOp>(loc, element);
            });
        return success();
    }
};

struct DissolveZip : OpRewritePattern<ZipOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        const auto resultTy =
            llvm::dyn_cast<ScalarType>(op.getType().getTypeBound());
        if (!resultTy) return failure();

        auto yieldOp = llvm::cast<YieldOp>(op.getCombinator()->getTerminator());
        rewriter.inlineBlockBefore(op.getCombinator(), op, op.getOperands());
        rewriter.replaceOp(op, yieldOp.getOperand());
        rewriter.eraseOp(yieldOp);
        return success();
    }
};

struct RewriteZipToAssoc : OpRewritePattern<ZipOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        const auto resultTy =
            llvm::dyn_cast<ArrayType>(op.getType().getTypeBound());
        if (!resultTy) return failure();

        SmallVector<Value> newOperands(op.getOperands());
        for (auto &op : newOperands) {
            op =
                rewriter
                    .create<BroadcastOp>(op.getLoc(), op, resultTy.getExtents())
                    .getResult();
        }

        rewriter.replaceOpWithNewOp<AssocOp>(
            op,
            resultTy,
            [&](OpBuilder &builder, Location loc, ValueRange indices) {
                SmallVector<Value> elements;
                for (auto [idx, opd] : llvm::enumerate(newOperands)) {
                    const auto elementTy =
                        llvm::cast<ExpressionType>(
                            op.getCombinator()->getArgument(idx).getType())
                            .getTypeBound();
                    elements.push_back(
                        builder
                            .create<SubscriptOp>(loc, opd, indices, elementTy)
                            .getResult());
                }

                rewriter.inlineBlockBefore(
                    op.getCombinator(),
                    builder.getInsertionBlock(),
                    builder.getInsertionBlock()->end(),
                    elements);
            });
        return success();
    }
};

struct CollapseAssoc : OpRewritePattern<AssocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(AssocOp op, PatternRewriter &rewriter) const final
    {
        auto subexpr = op.getMapExpression().getDefiningOp<AssocOp>();
        if (!subexpr) return failure();

        SmallVector<Value> inlined;
        for (auto arg : subexpr.getMap()->getArguments())
            inlined.push_back(
                op.getMap()->addArgument(arg.getType(), arg.getLoc()));

        rewriter.inlineBlockBefore(
            subexpr.getMap(),
            op.getMap()->getTerminator(),
            inlined);
        rewriter.eraseOp(op.getMap()->getTerminator());
        return success();
    }
};

} // namespace

void mlir::ekl::populateLowerPatterns(RewritePatternSet &patterns)
{
    // Hoisting applies to all functors and is essential for other rewrites.
    patterns.add<Hoist>(patterns.getContext());

    patterns
        .add<EliminateIf, DissolveIf, RewriteIfToStatement, RewriteIfToChoice>(
            patterns.getContext());

    patterns.add<ExpandEllipsisSubscript, RewriteSubscriptToAssoc>(
        patterns.getContext());

    patterns.add<RewriteZipToAssoc>(patterns.getContext());

    patterns.add<CollapseAssoc>(patterns.getContext());
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
