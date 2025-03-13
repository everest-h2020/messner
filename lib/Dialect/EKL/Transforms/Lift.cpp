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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/Support/LogicalResult.h>

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
            axes.back().getMap()->end());
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

// struct FactorizeReduction : OpRewritePattern<ReduceOp> {
//     using OpRewritePattern::OpRewritePattern;

//     LogicalResult
//     matchAndRewrite(ReduceOp op, PatternRewriter &rewriter) const final
//     {
//         // 1. Find a reduction (op) "add l, r" of an assoc (source) with >1
//         //    dims.
//         auto add = op.getReductionExpression().getDefiningOp<AddOp>();
//         if (!add
//             || !llvm::equal(
//                 add->getOperands(),
//                 op.getReduction()->getArguments()))
//             return failure();
//         auto source = op.getArray().getDefiningOp<AssocOp>();
//         if (!source) return failure();

//         auto dims = source.getMap()->getNumArguments();

//         while (dims > 1)
//             if (succeeded(factorizeDim(source, --dims, rewriter)))
//                 return success();

//         return failure();
//     }

// private:
//     LogicalResult
//     factorizeDim(AssocOp source, unsigned dim, PatternRewriter &rewriter)
//     const
//     {
//         const auto index = source.getMap()->getArgument(dim);

//         // 2. Find a dim that may be factorized:
//         //  a) There must be subscript expressions that the dim is not
//         involved
//         //     in.
//         SmallVector<SubscriptOp> involved;
//         SmallVector<SubscriptOp> uninvolved;
//         for (auto subscript : source.getOps<SubscriptOp>()) {
//             if (llvm::count(subscript.getSubscripts(), index)) {
//                 involved.push_back(subscript);
//                 continue;
//             }

//             uninvolved.push_back(subscript);
//         }
//         if (uninvolved.empty()) return failure();

//         //  b) The yield expression must be rearranged to "mul %involved,
//         //     %uninvolved"
//         // FIXME: For now, it must just be a mul tree.
//         if (failed(matchMulTree(source.getMapExpression()))) return
//         failure();

//         // 3. Factorize that dim:
//         //  a) Create an assoc (partial) before source with %involved
//         subscript
//         //     dims.

//         rewriter.setInsertionPoint(source);

//         auto partial =
//             rewriter.create<AssocOp>(source.getLoc(), ArrayType::get());

//         //  b) Move the %involved (op) map expressions to it (partial).
//         //  c) Create a reduction (lifted) on the assoc (partial), cloning
//         from
//         //     (op).
//         //  d) Replace %involved with a subscript to the reduction
//         //     (lifted).
//         //  e) Remove the dim from the old assoc (source).

//         return failure();
//     }

//     LogicalResult matchMulTree(Value yieldExpr) const
//     {
//         DenseSet<Value> seen{};
//         const auto match = [&](auto &self, Value expr) -> LogicalResult {
//             if (!seen.insert(expr).second) return failure();
//             if (auto subscript = expr.getDefiningOp<SubscriptOp>())
//                 return success();

//             auto mul = expr.getDefiningOp<MultiplyOp>();
//             if (!mul) return failure();
//             return success(
//                 succeeded(self(self, mul.getLhs()))
//                 && succeeded(self(self, mul.getRhs())));
//         };
//         return match(match, yieldExpr);
//     }
// };

} // namespace

void mlir::ekl::populateLiftPatterns(RewritePatternSet &patterns)
{
    populateHoistPatterns(patterns);

    patterns.add<SplitReduction>(patterns.getContext());

    // patterns.add<FactorizeReduction>(patterns.getContext());
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
