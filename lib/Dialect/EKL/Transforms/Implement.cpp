/// Implements the ImplementPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-implement"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_IMPLEMENT
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct ImplementPass : ekl::impl::ImplementBase<ImplementPass> {
    using ImplementBase::ImplementBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateImplementPatterns implementation
//===----------------------------------------------------------------------===//

namespace {

struct ImplementElementwise {
protected:
    static LogicalResult
    rewriteElementwise(PatternRewriter &rewriter, Operation *op)
    {
        const auto resultTy = llvm::dyn_cast_if_present<ekl::ArrayType>(
            llvm::cast<ExpressionType>(op->getResult(0).getType())
                .getTypeBound());
        if (!resultTy) return failure();

        rewriter.replaceOpWithNewOp<ZipOp>(
            op,
            op->getOperands(),
            [&](OpBuilder &builder, Location loc, ValueRange operands) {
                IRMapping mapping;
                for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                    operands[i].setType(ExpressionType::get(
                        rewriter.getContext(),
                        getScalarType(
                            llvm::cast<ExpressionType>(op->getOperandTypes()[i])
                                .getTypeBound())));
                    mapping.map(op->getOperand(i), operands[i]);
                }

                auto elOp = builder.insert(op->clone(mapping));
                elOp->getResult(0).setType(ExpressionType::get(
                    rewriter.getContext(),
                    getScalarType(
                        llvm::cast<ExpressionType>(op->getResultTypes()[0])
                            .getTypeBound())));
                builder.create<YieldOp>(loc, elOp->getResult(0));
            },
            resultTy);
        return success();
    }
};

struct ImplementBroadcast : OpRewritePattern<BroadcastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(BroadcastOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped()) return failure();

        auto inExtents = *getExtents(op.getOperand().getType().getTypeBound());
        auto elTy = getScalarType(op.getOperand().getType().getTypeBound());

        if (inExtents.empty()) {
            rewriter.replaceOpWithNewOp<AssocOp>(
                op,
                llvm::cast<ArrayType>(op.getType().getTypeBound()),
                [&](OpBuilder &builder, Location loc, ValueRange) {
                    builder.create<YieldOp>(loc, op.getOperand());
                });
            return success();
        }

        rewriter.replaceOpWithNewOp<AssocOp>(
            op,
            llvm::cast<ArrayType>(op.getType().getTypeBound()),
            [&](OpBuilder &builder, Location loc, ValueRange indices) {
                auto zero =
                    builder
                        .create<LiteralOp>(
                            loc,
                            ekl::IndexAttr::get(builder.getContext(), 0UL))
                        .getResult();

                SmallVector<Value> subscripts(indices.size(), zero);
                for (unsigned i = 0; i < indices.size(); ++i)
                    if (inExtents[i] != 1) subscripts[i] = indices[i];

                auto el = builder
                              .create<SubscriptOp>(
                                  loc,
                                  op.getOperand(),
                                  subscripts,
                                  elTy)
                              .getResult();

                builder.create<YieldOp>(loc, el);
            });
        return success();
    }
};

struct ImplementCoerce : OpRewritePattern<CoerceOp>, ImplementElementwise {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(CoerceOp op, PatternRewriter &rewriter) const final
    {
        return rewriteElementwise(rewriter, op);
    }
};

struct ImplementRelational : OpTraitRewritePattern<ekl::OpTrait::IsRelational>,
                             ImplementElementwise {
    using OpTraitRewritePattern<
        ekl::OpTrait::IsRelational>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        return rewriteElementwise(rewriter, op);
    }
};

struct ImplementArithmetic : OpTraitRewritePattern<ekl::OpTrait::IsArithmetic>,
                             ImplementElementwise {
    using OpTraitRewritePattern<
        ekl::OpTrait::IsArithmetic>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        return rewriteElementwise(rewriter, op);
    }
};

struct ImplementChoice : OpRewritePattern<ChoiceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ChoiceOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped()) return failure();
        const auto selectorTy = llvm::dyn_cast<ekl::ArrayType>(
            op.getSelector().getType().getTypeBound());
        if (!selectorTy) return failure();
        if (static_cast<size_t>(llvm::count(
                op.getAlternatives().getTypes(),
                op.getAlternatives().front().getType()))
            != op.getAlternatives().size())
            return failure();
        const auto altTy = llvm::cast<BroadcastType>(
            llvm::cast<ExpressionType>(op.getAlternatives().front().getType())
                .getTypeBound());

        Type yieldTy = altTy.getScalarType();
        if (altTy.getExtents().size() > selectorTy.getNumExtents())
            yieldTy = ArrayType::get(
                yieldTy,
                altTy.getExtents().drop_front(selectorTy.getNumExtents()));

        auto assoc = rewriter.replaceOpWithNewOp<AssocOp>(
            op,
            selectorTy,
            [&](OpBuilder &builder, Location loc, ValueRange indices) {
                auto sel = builder
                               .create<SubscriptOp>(
                                   loc,
                                   op.getSelector(),
                                   indices,
                                   selectorTy.getScalarType())
                               .getResult();
                SmallVector<Value> alts(op.getAlternatives());
                if (!llvm::isa<ScalarType>(altTy)) {
                    indices = indices.take_front(altTy.getExtents().size());
                    for (auto &alt : alts)
                        alt =
                            builder
                                .create<SubscriptOp>(loc, alt, indices, yieldTy)
                                .getResult();
                }
                auto choose = builder.create<ChoiceOp>(loc, sel, alts, yieldTy)
                                  .getResult();
                builder.create<YieldOp>(loc, choose);
            });
        assoc.getResult().setType(
            ExpressionType::get(rewriter.getContext(), altTy));
        return success();
    }
};

struct InlineAssoc : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped()
            || !llvm::isa<ScalarType>(op.getResult().getType().getTypeBound()))
            return failure();
        auto subexpr = op.getArray().getDefiningOp<AssocOp>();
        if (!subexpr || !subexpr.getResult().hasOneUse()) return failure();
        if (subexpr.getMap()->getNumArguments() != op.getSubscripts().size())
            return failure();
        if (subexpr->getAttr("ekl.lifted")) return failure();

        auto res = subexpr.getMap()->getTerminator()->getOperand(0);
        rewriter.eraseOp(subexpr.getMap()->getTerminator());
        rewriter.inlineBlockBefore(subexpr.getMap(), op, op.getSubscripts());
        rewriter.replaceOp(op, ValueRange(res));
        return success();
    }
};

} // namespace

void mlir::ekl::populateImplementPatterns(RewritePatternSet &patterns)
{
    populateLowerPatterns(patterns);

    patterns.add<ImplementBroadcast, ImplementCoerce>(patterns.getContext());
    patterns.add<ImplementRelational, ImplementArithmetic>(
        patterns.getContext());
    patterns.add<ImplementChoice>(patterns.getContext());
    patterns.add<InlineAssoc>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ImplementPass implementation
//===----------------------------------------------------------------------===//

void ImplementPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateImplementPatterns(patterns);

    if (failed(applyPatternsGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createImplementPass()
{
    return std::make_unique<ImplementPass>();
}
