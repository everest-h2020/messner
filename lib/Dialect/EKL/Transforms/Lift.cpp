/// Implements the LiftPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ImplicitCast.h"
#include "messner/Dialect/EKL/Analysis/Extents.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Traits.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "messner/Dialect/EKL/Transforms/Passes.h"
#include "messner/Dialect/EKL/Transforms/TypeCheck.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-lift"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_LIFT
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct LiftPass : ekl::impl::LiftBase<LiftPass> {
    using LiftBase::LiftBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateLiftPatterns implementation
//===----------------------------------------------------------------------===//

namespace {

struct SplitReduction : OpRewritePattern<ReduceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ReduceOp op, PatternRewriter &rewriter) const final
    {
        auto source = op.getArray().getDefiningOp<AssocOp>();
        if (!source || !source->hasOneUse()) return failure();
        auto numDims = source.getMap()->getNumArguments();
        if (numDims < 2) return failure();
        const auto arrayTy =
            llvm::cast<ArrayType>(getTypeBound(source.getType()));

        SmallVector<AssocOp> axes;
        for (unsigned dim = 0; dim < numDims - 1U; ++dim) {
            auto axis = rewriter.create<AssocOp>(
                op.getLoc(),
                ArrayType::get(
                    arrayTy.getScalarType(),
                    arrayTy.getExtent(dim)));
            rewriter.setInsertionPointToStart(axis.getMap());
            axes.push_back(axis);
        }

        rewriter.moveOpBefore(
            source,
            axes.back().getMap(),
            axes.back().getMap()->begin());
        rewriter.modifyOpInPlace(source, [&]() {
            for (unsigned dim = 0; dim < numDims - 1U; ++dim) {
                rewriter.replaceAllUsesWith(
                    source.getMap()->getArgument(dim),
                    axes[dim].getMap()->getArgument(0));
            }
            source.getMap()->eraseArguments(0, numDims - 1U);

            source.getResult().setType(ExpressionType::get(
                getContext(),
                ArrayType::get(
                    arrayTy.getScalarType(),
                    arrayTy.getExtents().back())));
        });
        axes.push_back(source);

        const auto remap = axes.front().getResult();
        op->setOperand(0, remap);

        for (int dim = numDims - 2; dim >= 0; --dim) {
            rewriter.setInsertionPointToEnd(axes[dim].getMap());

            IRMapping mapping;
            mapping.map(remap, axes[dim + 1].getResult());
            auto reduce = llvm::cast<ReduceOp>(rewriter.clone(*op, mapping));

            rewriter.create<YieldOp>(op.getLoc(), reduce.getResult());
        }

        return success();
    }
};

struct Factorizer {
    explicit Factorizer(PatternRewriter &rewriter, AssocOp op)
            : m_rewriter(rewriter),
              m_op(op),
              m_liftable()
    {
        assert(op);
    }

    LogicalResult factorize()
    {
        auto definition = m_op.getMapExpression().getDefiningOp();
        if (!definition) return failure();

        auto maybeFactor = visit(definition);
        if (failed(maybeFactor)) return failure();

        m_op.getMap()->back().setOperand(0, maybeFactor->getLhs());
        m_rewriter.setInsertionPointAfter(m_op);

        auto lifted   = lift(maybeFactor->getRhs());
        auto factored = m_rewriter.create<MultiplyOp>(
            m_op.getLoc(),
            m_op.getResult(),
            lifted,
            getTypeBound(m_op.getType()));
        m_rewriter.replaceAllUsesExcept(m_op.getResult(), factored, factored);
        return success();
    }

private:
    bool isLiftable(Value value)
    {
        assert(value);

        auto [it, compute] = m_liftable.try_emplace(value, false);
        if (!compute) return it->second;

        return m_liftable.insert_or_assign(value, isLiftableImpl(value))
            .first->second;
    }
    bool isLiftable(Operation *op)
    {
        assert(op);

        auto walk = op->walk([&](Operation *op) -> WalkResult {
            if (!isMemoryEffectFree(op) || !isSpeculatable(op))
                return WalkResult::interrupt();

            if (llvm::any_of(op->getOperands(), [&](Value value) {
                    return !isLiftable(value);
                }))
                return WalkResult::interrupt();

            if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
                return WalkResult::skip();
            return WalkResult::advance();
        });

        return !walk.wasInterrupted();
    }
    bool isLiftableImpl(Value value)
    {
        assert(value);

        if (value.getParentRegion()->isProperAncestor(&m_op.getMapRegion()))
            return true;

        auto definition = value.getDefiningOp();
        if (!definition) return false;
        return isLiftable(definition);
    }

    FailureOr<MultiplyOp> visit(Operation *op)
    {
        assert(op);

        return llvm::TypeSwitch<Operation *, FailureOr<MultiplyOp>>(op)
            .Case([&](AddOp add) { return visit(add); })
            .Case([&](MultiplyOp multiply) { return visit(multiply); })
            .Default([](auto) -> FailureOr<MultiplyOp> { return failure(); });
    }
    FailureOr<MultiplyOp> visit(AddOp op)
    {
        assert(op);

        auto lhs = op.getLhs().getDefiningOp<MultiplyOp>();
        auto rhs = op.getRhs().getDefiningOp<MultiplyOp>();
        if (!lhs || !rhs) return failure();

        const auto factorize =
            [&](Value factor, Value lhs, Value rhs) -> FailureOr<MultiplyOp> {
            m_rewriter.setInsertionPoint(op);
            const auto scalarTy = getTypeBound(op.getType());
            auto add =
                m_rewriter.create<AddOp>(op.getLoc(), lhs, rhs, scalarTy);
            auto mul = m_rewriter.create<MultiplyOp>(
                op.getLoc(),
                factor,
                add.getResult(),
                scalarTy);
            return visit(mul);
        };

        if (lhs.getLhs() == rhs.getLhs())
            return factorize(lhs.getLhs(), lhs.getRhs(), rhs.getRhs());
        if (lhs.getLhs() == rhs.getRhs())
            return factorize(lhs.getLhs(), lhs.getRhs(), rhs.getLhs());
        if (lhs.getRhs() == rhs.getLhs())
            return factorize(lhs.getRhs(), lhs.getLhs(), rhs.getRhs());
        if (lhs.getRhs() == rhs.getRhs())
            return factorize(lhs.getRhs(), lhs.getLhs(), rhs.getLhs());
        return failure();
    }
    FailureOr<MultiplyOp> visit(MultiplyOp op)
    {
        assert(op);

        if (isLiftable(op.getRhs())) return op;
        if (auto rhs = op.getRhs().getDefiningOp()) {
            auto maybeFactor = visit(rhs);
            if (succeeded(maybeFactor)) {
                op->setOperand(1, maybeFactor->getRhs());
                maybeFactor->setOperand(1, op.getLhs());
                op.setOperand(0, maybeFactor->getResult());
                return op;
            }
        }
        if (isLiftable(op.getLhs())) {
            Value ops[2] = {op.getRhs(), op.getLhs()};
            op->setOperands(ops);
            return op;
        }
        if (auto lhs = op.getLhs().getDefiningOp()) {
            auto maybeFactor = visit(lhs);
            if (succeeded(maybeFactor)) {
                auto factor = maybeFactor->getRhs();
                maybeFactor->setOperand(1, op.getRhs());
                op.setOperand(1, factor);
                return op;
            }
        }

        return failure();
    }

    Value lift(Value value)
    {
        if (value.getParentRegion()->isProperAncestor(&m_op.getMapRegion()))
            return value;

        auto definition = value.getDefiningOp();
        assert(definition);

        for (auto [i, op] : llvm::enumerate(definition->getOperands()))
            definition->setOperand(i, lift(op));

        m_rewriter.moveOpBefore(definition, m_op);
        return value;
    }

    PatternRewriter &m_rewriter;
    AssocOp m_op;
    DenseMap<Value, bool> m_liftable;
};

struct LiftFactor : OpRewritePattern<AssocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(AssocOp op, PatternRewriter &rewriter) const final
    {
        return Factorizer(rewriter, op).factorize();
    }
};

struct DistributeFactor : OpRewritePattern<ReduceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ReduceOp op, PatternRewriter &rewriter) const final
    {
        auto mul = op.getArray().getDefiningOp<MultiplyOp>();
        if (!mul) return failure();
        if (llvm::isa<ScalarType>(getTypeBound(mul.getLhs()))) {
            Value ops[2] = {mul.getLhs(), mul.getRhs()};
            mul->setOperands(ops);
        } else if (!llvm::isa<ScalarType>(getTypeBound(mul.getRhs())))
            return failure();

        auto add = op.getReductionExpression().getDefiningOp<AddOp>();
        if (!add || add.getOperands() != op.getReduction()->getArguments())
            return failure();

        op.setOperand(0, mul.getLhs());
        rewriter.setInsertionPointAfter(op);
        auto postMul = rewriter.create<MultiplyOp>(
            mul.getLoc(),
            op.getResult(),
            mul.getRhs(),
            getTypeBound(op.getType()));
        rewriter.replaceAllUsesExcept(
            op.getResult(),
            postMul.getResult(),
            postMul);
        return success();
    }
};

static AssocOp findTop(AssocOp bottom, Region *definition)
{
    auto top = bottom;
    for (auto next = top->getParentOfType<AssocOp>();
         next && definition->isAncestor(next->getParentRegion());
         next = next->getParentOfType<AssocOp>())
        top = next;

    return top;
}

static FailureOr<std::pair<AssocOp, bool>>
prepareLiftAssoc(AssocOp assoc, SmallVectorImpl<BlockArgument> &indices)
{
    AssocOp top{};
    for (auto parent = assoc->getParentOfType<AssocOp>(); parent;
         parent      = parent->getParentOfType<AssocOp>()) {
        const auto args = llvm::reverse(parent.getMap()->getArguments());
        indices.insert(indices.end(), args.begin(), args.end());
        top = parent;
    }
    if (!top) return failure();

    DenseSet<BlockArgument> usedIndices;
    assoc->walk([&](Operation *op) {
        for (auto opd : op->getOperands()) {
            if (auto arg = llvm::dyn_cast<BlockArgument>(opd))
                if (llvm::isa<AssocOp>(arg.getOwner()->getParentOp())) {
                    usedIndices.insert(arg);
                    continue;
                }

            const auto definition = opd.getParentRegion();
            if (assoc.getMapRegion().isAncestor(definition)
                || definition->isAncestor(top->getParentRegion()))
                continue;

            top = findTop(assoc, definition);
        }

        if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
            return WalkResult::skip();
        return WalkResult::advance();
    });
    if (top == assoc) return failure();

    const auto numIndices = indices.size();
    llvm::remove_if(indices, [&](BlockArgument arg) {
        return !usedIndices.contains(arg);
    });
    std::reverse(indices.begin(), indices.end());

    return std::make_pair(top, indices.size() < numIndices);
}

struct LiftAssoc : OpRewritePattern<AssocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(AssocOp op, PatternRewriter &rewriter) const final
    {
        SmallVector<BlockArgument> indices;
        const auto maybeResult = prepareLiftAssoc(op, indices);
        if (failed(maybeResult) || !maybeResult->second) return failure();

        const auto arrayTy = llvm::cast<ArrayType>(getTypeBound(op.getType()));
        SmallVector<extent_t> extents;
        for (auto arg : indices) {
            const auto indexTy =
                llvm::dyn_cast<ekl::IndexType>(getTypeBound(arg));
            if (!indexTy) return failure();
            extents.push_back(indexTy.getUpperBound() + 1U);
        }
        concat(extents, arrayTy.getExtents());

        rewriter.setInsertionPoint(maybeResult->first);
        auto lifted = rewriter.create<AssocOp>(
            op.getLoc(),
            ArrayType::get(arrayTy.getScalarType(), extents));

        rewriter.setInsertionPointAfter(op);
        auto subscript = rewriter.create<SubscriptOp>(
            op.getLoc(),
            lifted.getResult(),
            ValueRange(indices),
            getTypeBound(op.getType()));
        rewriter.replaceAllUsesWith(op.getResult(), subscript.getResult());

        rewriter.moveOpBefore(op, lifted.getMap(), lifted.getMap()->begin());
        for (auto [i, idx] : llvm::enumerate(indices))
            rewriter.replaceUsesWithIf(
                idx,
                lifted.getMap()->getArgument(i),
                [&](OpOperand &use) {
                    return lifted->isAncestor(use.getOwner());
                });

        rewriter.setInsertionPointToEnd(lifted.getMap());
        rewriter.create<YieldOp>(op.getLoc(), op.getResult());
        return success();
    }
};

struct LiftReduce : OpRewritePattern<ReduceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ReduceOp reduce, PatternRewriter &rewriter) const final
    {
        auto source = reduce.getArray().getDefiningOp<AssocOp>();
        if (!source || !source->hasOneUse()) return failure();

        SmallVector<BlockArgument> indices;
        const auto maybeResult = prepareLiftAssoc(source, indices);
        if (failed(maybeResult)) return failure();

        auto top = maybeResult->first;
        reduce->walk([&](Operation *op) {
            for (auto opd : op->getOperands()) {
                const auto definition = opd.getParentRegion();
                if (reduce.getReductionRegion().isAncestor(definition)
                    || definition->isAncestor(top->getParentRegion()))
                    continue;

                top = findTop(source, definition);
            }

            if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
                return WalkResult::skip();
            return WalkResult::advance();
        });
        if (top == source) return failure();

        if (llvm::equal(top.getMap()->getArguments(), indices))
            return failure();

        const auto scalarTy =
            llvm::cast<ScalarType>(getTypeBound(reduce.getType()));
        SmallVector<extent_t> extents;
        for (auto arg : indices) {
            const auto indexTy =
                llvm::dyn_cast<ekl::IndexType>(getTypeBound(arg));
            if (!indexTy) return failure();
            extents.push_back(indexTy.getUpperBound() + 1U);
        }

        rewriter.setInsertionPoint(top);
        auto lifted = rewriter.create<AssocOp>(
            reduce.getLoc(),
            ArrayType::get(scalarTy, extents));

        rewriter.setInsertionPointAfter(reduce);
        auto subscript = rewriter.create<SubscriptOp>(
            reduce.getLoc(),
            lifted.getResult(),
            ValueRange(indices),
            scalarTy);
        rewriter.replaceAllUsesWith(reduce.getResult(), subscript.getResult());

        rewriter.moveOpBefore(
            source,
            lifted.getMap(),
            lifted.getMap()->begin());
        for (auto [i, idx] : llvm::enumerate(indices))
            rewriter.replaceUsesWithIf(
                idx,
                lifted.getMap()->getArgument(i),
                [&](OpOperand &use) {
                    return lifted->isAncestor(use.getOwner());
                });

        rewriter.moveOpAfter(reduce, source);
        rewriter.setInsertionPointToEnd(lifted.getMap());
        rewriter.create<YieldOp>(reduce.getLoc(), reduce.getResult());
        return success();
    }
};

} // namespace

void mlir::ekl::populateLiftPatterns(RewritePatternSet &patterns)
{
    populateHoistPatterns(patterns);

    patterns.add<SplitReduction>(patterns.getContext());

    patterns.add<LiftFactor, DistributeFactor>(patterns.getContext());

    patterns.add<LiftAssoc, LiftReduce>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LiftPass implementation
//===----------------------------------------------------------------------===//

void LiftPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateLiftPatterns(patterns);

    if (failed(applyPatternsGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createLiftPass()
{
    return std::make_unique<LiftPass>();
}
