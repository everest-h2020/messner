/// Implements the HomogenizePass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ImplicitCast.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-homogenize"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_HOMOGENIZE
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct HomogenizePass : ekl::impl::HomogenizeBase<HomogenizePass> {
    using HomogenizeBase::HomogenizeBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateHomogenizePatterns implementation
//===----------------------------------------------------------------------===//

namespace {

struct UnifyWriteValue : OpRewritePattern<WriteOp>, ImplicitCast {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(WriteOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "not fully typed");

        // Value operand implicitly unifies to output array type.
        const auto refTy = llvm::cast<ReferenceType>(
            op.getReference().getType().getTypeBound());
        return unifyOperands(
            rewriter,
            op.getValueMutable(),
            refTy.getArrayType());
    }
};

struct BroadcastAndUnifyStackOperands : OpRewritePattern<StackOp>,
                                        ImplicitCast {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(StackOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "not fully typed");

        // The operand type must be the result type without the stacked extent.
        const auto resultTy =
            llvm::cast<ArrayType>(op.getType().getTypeBound());
        const auto argTy =
            resultTy.cloneWith(resultTy.getExtents().drop_front());
        return broadcastAndUnifyOperands(rewriter, op.getOperands(), argTy);
    }
};

struct BroadcastZipOperands : OpRewritePattern<ZipOp>, ImplicitCast {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped()
            || !op.getCombinatorExpression().getType().getTypeBound())
            return rewriter.notifyMatchFailure(op, "not fully typed");

        const auto resultTy =
            llvm::dyn_cast<ArrayType>(op.getType().getTypeBound());
        if (!resultTy) return failure();

        // Broadcast the arguments to the prefix of the result extents.
        const auto numYieldExtents =
            getExtents(op.getCombinatorExpression().getType().getTypeBound())
                ->size();
        const auto argExtents =
            resultTy.getExtents().drop_back(numYieldExtents);
        return broadcastOperands(rewriter, op.getOperands(), argExtents);
    }
};

struct BroadcastAndUnifyChoiceOperands : OpRewritePattern<ChoiceOp>,
                                         ImplicitCast {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ChoiceOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "not fully typed");

        const auto altTy =
            llvm::dyn_cast<ArrayType>(op.getType().getTypeBound());
        if (!altTy) return failure();

        // Broadcast and unify the alternatives to the result type.
        auto updated = succeeded(broadcastAndUnifyOperands(
            rewriter,
            op.getAlternativesMutable(),
            altTy));

        // Broadcast the selector to the prefix type.
        const auto selTy = op.getSelector().getType().getTypeBound();
        const auto selExtents =
            altTy.getExtents().take_front(getExtents(selTy)->size());
        updated |= succeeded(
            broadcastOperands(rewriter, op.getSelectorMutable(), selExtents));

        return success(updated);
    }
};

struct BroadcastAndUnifyRelational
        : OpTraitRewritePattern<ekl::OpTrait::IsRelational>,
          ImplicitCast {
    using OpTraitRewritePattern<
        ekl::OpTrait::IsRelational>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        if (!ekl::impl::isFullyTyped(op))
            return rewriter.notifyMatchFailure(op, "not fully typed");

        // Broadcast and unify the arguments to the common array type.
        const auto resultTy = llvm::dyn_cast<ArrayType>(
            llvm::cast<Expression>(op->getResult(0)).getType().getTypeBound());
        if (!resultTy) return failure();
        auto argScalarTys = llvm::to_vector(
            llvm::map_range(op->getOperandTypes(), [](Type type) -> Type {
                return getScalarType(
                    llvm::cast<ExpressionType>(type).getTypeBound());
            }));
        const auto scalarTy = ekl::unify(argScalarTys);
        return broadcastAndUnifyOperands(
            rewriter,
            op->getOperands(),
            resultTy.cloneWith(*scalarTy));
    }
};

template<template<class> class Trait>
struct BroadcastAndUnifyTrait : OpTraitRewritePattern<Trait>, ImplicitCast {
    using OpTraitRewritePattern<Trait>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        if (!ekl::impl::isFullyTyped(op))
            return rewriter.notifyMatchFailure(op, "not fully typed");

        if (op->getNumResults() != 1)
            return rewriter.notifyMatchFailure(op, "ambiguous result type");
        const auto resultTy = llvm::dyn_cast<ArrayType>(
            llvm::cast<Expression>(op->getResult(0)).getType().getTypeBound());
        if (!resultTy) return failure();

        return broadcastAndUnifyOperands(rewriter, op->getOperands(), resultTy);
    }
};

using BroadcastAndUnifyLogical =
    BroadcastAndUnifyTrait<ekl::OpTrait::IsLogical>;
using BroadcastAndUnifyArithmetic =
    BroadcastAndUnifyTrait<ekl::OpTrait::IsArithmetic>;

} // namespace

void mlir::ekl::populateHomogenizePatterns(RewritePatternSet &patterns)
{
    // WriteOp
    patterns.add<UnifyWriteValue>(patterns.getContext());

    // StackOp
    patterns.add<BroadcastAndUnifyStackOperands>(patterns.getContext());

    // ZipOp
    patterns.add<BroadcastZipOperands>(patterns.getContext());

    // ChoiceOp
    patterns.add<BroadcastAndUnifyChoiceOperands>(patterns.getContext());

    // CompareOp, MinOp, MaxOp
    patterns.add<BroadcastAndUnifyRelational>(patterns.getContext());

    // LogicalNotOp, LogicalAndOp, LogicalOrOp
    patterns.add<BroadcastAndUnifyLogical>(patterns.getContext());

    // NegateOp, AddOp, SubtractOp, MultiplyOp, DivideOp, RemainderOp, PowerOp
    patterns.add<BroadcastAndUnifyArithmetic>(patterns.getContext());

    // TODO: TensorProductOp: broadcast and unify operands.
}

//===----------------------------------------------------------------------===//
// HomogenizePass implementation
//===----------------------------------------------------------------------===//

void HomogenizePass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateHomogenizePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createHomogenizePass()
{
    return std::make_unique<HomogenizePass>();
}
