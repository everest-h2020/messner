/// Implements the DecayNumberPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ImplicitCast.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/Transforms/TypeCheck.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-decay-number"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_DECAYNUMBER
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct DecayNumberPass : ekl::impl::DecayNumberBase<DecayNumberPass> {
    using DecayNumberBase::DecayNumberBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateDecayNumberPatterns implementation
//===----------------------------------------------------------------------===//

namespace {

struct ScalarDecay : ImplicitCast {
protected:
    static LogicalResult decayOperands(
        PatternRewriter &rewriter,
        OperandRange operands,
        ScalarType scalarTy)
    {
        return rewriteOperands(
            rewriter,
            operands,
            [&](OpBuilder &builder, Location loc, Value &operand) {
                const auto opTy =
                    llvm::cast<ArithmeticType>(getTypeBound(operand));
                if (opTy.getScalarType() == scalarTy) return failure();
                if (llvm::isa<NumberType>(opTy.getScalarType()))
                    operand = builder
                                  .create<CoerceOp>(
                                      loc,
                                      operand,
                                      opTy.cloneWith(scalarTy))
                                  .getResult();
                else
                    operand = builder
                                  .create<UnifyOp>(
                                      loc,
                                      operand,
                                      opTy.cloneWith(scalarTy))
                                  .getResult();
                return success();
            });
    }
};

struct DecayLiteralArithmetic
        : OpTraitRewritePattern<ekl::OpTrait::IsArithmetic>,
          ScalarDecay {
    using OpTraitRewritePattern<
        ekl::OpTrait::IsArithmetic>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        // Find a scalar type to decay to.
        const auto decayedTy = getDecayedScalarType(op);
        if (failed(decayedTy)) return failure();

        // Coerce the decayed operands.
        if (failed(decayOperands(rewriter, op->getOperands(), *decayedTy)))
            return failure();

        // Propagate the type update to users.
        TypeChecker typeChecker;
        auto ok = typeChecker.refineBound(
            llvm::cast<Expression>(op->getResult(0)),
            llvm::cast<ArithmeticType>(getTypeBound(op->getResult(0)))
                .cloneWith(*decayedTy));
        assert(succeeded(ok));
        ok = typeChecker.check();
        assert(succeeded(ok));
        return success();
    }

private:
    [[nodiscard]] static FailureOr<ScalarType>
    getDecayedScalarType(Operation *op)
    {
        auto result = ScalarType{};
        for (auto operand : op->getOperands()) {
            const auto bound = getTypeBound(operand);
            if (!bound) return failure();
            const auto scalarTy =
                llvm::cast<ArithmeticType>(bound).getScalarType();
            if (llvm::isa<NumberType>(scalarTy)) {
                if (operand.getDefiningOp<LiteralOp>()) continue;
                return failure();
            }
            if (!result || isSubtype(result, scalarTy)) result = scalarTy;
        }
        return result;
    }
};

struct DecayCoercedArithmetic
        : OpTraitRewritePattern<ekl::OpTrait::IsArithmetic>,
          ScalarDecay {
    using OpTraitRewritePattern<
        ekl::OpTrait::IsArithmetic>::OpTraitRewritePattern;

    LogicalResult
    matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final
    {
        const auto bound = getTypeBound(op->getResult(0));
        if (!bound) return failure();
        // Check if the only use is a coersion.
        if (!llvm::isa<NumberType>(
                llvm::cast<ArithmeticType>(bound).getScalarType()))
            return failure();
        if (!op->getResult(0).hasOneUse()) return failure();
        auto user =
            llvm::dyn_cast<CoerceOp>(*op->getResult(0).getUsers().begin());
        if (!user) return failure();
        auto resultTy =
            llvm::cast<ArithmeticType>(user.getType().getTypeBound());

        // Decay the operands.
        if (failed(decayOperands(
                rewriter,
                op->getOperands(),
                resultTy.getScalarType())))
            return failure();

        // Replace the coersion.
        rewriter.updateRootInPlace(op, [&]() {
            op->getResult(0).setType(
                ExpressionType::get(getContext(), resultTy));
        });
        rewriter.replaceOp(user, op);
        return success();
    }
};

} // namespace

void mlir::ekl::populateDecayNumberPatterns(RewritePatternSet &patterns)
{
    patterns.add<DecayLiteralArithmetic>(patterns.getContext());

    patterns.add<DecayCoercedArithmetic>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// DecayNumberPass implementation
//===----------------------------------------------------------------------===//

void DecayNumberPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateDecayNumberPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createDecayNumberPass()
{
    return std::make_unique<DecayNumberPass>();
}
