/// Implements the ConvertEKLToIndexPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToIndex/EKLToIndex.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOINDEX
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

//===----------------------------------------------------------------------===//

namespace {

struct ConvertEKLToIndexPass
        : messner::impl::ConvertEKLToIndexBase<ConvertEKLToIndexPass> {
    using ConvertEKLToIndexBase::ConvertEKLToIndexBase;

    void runOnOperation() override;
};

} // namespace

void ConvertEKLToIndexPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    const auto unrealizedCast = [](OpBuilder &builder,
                                   Type resultTy,
                                   ValueRange inputs,
                                   Location loc) -> Value {
        if (inputs.size() != 1) return {};
        return builder.create<UnrealizedConversionCastOp>(loc, resultTy, inputs)
            .getResult(0);
    };

    TypeConverter eklConverter;
    eklConverter.addConversion([&](ekl::ExpressionType exprTy) -> Type {
        if (const auto indexTy =
                llvm::dyn_cast<ekl::IndexType>(exprTy.getTypeBound()))
            return mlir::IndexType::get(exprTy.getContext());
        return exprTy;
    });

    eklConverter.addTargetMaterialization(
        [&](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Value {
            // Unwrap expressions using the EvalOp.
            if (inputs.size() != 1) return {};
            const auto exprTy =
                llvm::dyn_cast<ekl::ExpressionType>(inputs.front().getType());
            if (!exprTy) return {};

            const auto eval = builder
                                  .create<ekl::EvalOp>(
                                      loc,
                                      inputs.front(),
                                      exprTy.getTypeBound())
                                  .getResult();
            return unrealizedCast(builder, resultTy, {eval}, loc);
        });
    eklConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Value {
            // Wrap expressions using the IntroOp.
            if (inputs.size() != 1) return {};
            const auto exprTy = llvm::dyn_cast<ExpressionType>(resultTy);
            if (!exprTy) return {};

            const auto intro =
                unrealizedCast(builder, exprTy.getTypeBound(), inputs, loc);
            return builder.create<ekl::IntroOp>(loc, intro);
        });

    const auto isCastIndexIllegal = [&](Operation *op) -> bool {
        return !eklConverter.convertType(op->getOperand(0).getType())
            || !eklConverter.convertType(op->getResult(0).getType());
    };
    const auto isIndexIllegal = [&](Operation *op) -> bool {
        const auto indexTy =
            eklConverter.convertType(op->getResult(0).getType());
        return !indexTy
            || op->getNumOperands()
                   != llvm::count(
                       op->getOperandTypes(),
                       op->getResult(0).getType());
    };

    messner::populateConvertEKLToIndexPatterns(eklConverter, patterns);

    target.addDynamicallyLegalOp<ekl::UnifyOp, ekl::CoerceOp>(
        isCastIndexIllegal);

    target.addDynamicallyLegalOp<ekl::LiteralOp>(isIndexIllegal);
    target.addDynamicallyLegalOp<ekl::CompareOp>([&](ekl::CompareOp op) {
        if (op.getLhs().getType() != op.getRhs().getType()) return true;
        return !eklConverter.convertType(op.getLhs().getType());
    });
    target.addDynamicallyLegalOp<ekl::MinOp, ekl::MaxOp>(isIndexIllegal);
    target
        .addIllegalOp<ekl::LogicalNotOp, ekl::LogicalAndOp, ekl::LogicalOrOp>();
    target.addDynamicallyLegalOp<
        ekl::NegateOp,
        ekl::AddOp,
        ekl::SubtractOp,
        ekl::MultiplyOp,
        ekl::DivideOp,
        ekl::RemainderOp>(isIndexIllegal);
    target.addLegalDialect<index::IndexDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void messner::populateConvertEKLToIndexPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    // patterns.add<LowerLiteral>(typeConverter, patterns.getContext());
    // patterns.add<LowerUnify, LowerCoerce>(typeConverter,
    // patterns.getContext()); patterns.add<LowerNot, LowerAnd, LowerOr>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<LowerCompare, LowerMin, LowerMax>(
    //     typeConverter,
    //     patterns.getContext());
    // patterns.add<
    //     LowerNegate,
    //     LowerAdd,
    //     LowerSubtract,
    //     LowerMultiply,
    //     LowerDivide,
    //     LowerRemainder>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> messner::createConvertEKLToIndexPass()
{
    return std::make_unique<ConvertEKLToIndexPass>();
}
