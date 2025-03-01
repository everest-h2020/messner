/// Implements the ConvertEKLToArithPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToArith/EKLToArith.h"

#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOARITH
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

struct LowerLiteral : OpConversionPattern<ekl::LiteralOp> {
    using OpConversionPattern<ekl::LiteralOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LiteralOp op,
        ekl::LiteralOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        auto attr = llvm::dyn_cast<TypedAttr>(adaptor.getValueAttr());
        if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            // Convert to signless integer attribute.
            attr = mlir::IntegerAttr::get(
                mlir::IntegerType::get(
                    attr.getContext(),
                    intAttr.getType().getIntOrFloatBitWidth()),
                intAttr.getValue());
        }

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
        return success();
    }
};

template<class Source, class TargetI, class TargetF>
struct LowerClosedBinaryOp : OpConversionPattern<Source> {
    using OpConversionPattern<Source>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        Source op,
        Source::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (!getTypeBound(op.getResult()).isInteger()) {
            rewriter.replaceOpWithNewOp<TargetF>(
                op,
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        rewriter.replaceOpWithNewOp<TargetI>(
            op,
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

using LowerAdd = LowerClosedBinaryOp<ekl::AddOp, arith::AddIOp, arith::AddFOp>;
using LowerSubtract =
    LowerClosedBinaryOp<ekl::SubtractOp, arith::SubIOp, arith::SubFOp>;
using LowerMultiply =
    LowerClosedBinaryOp<ekl::MultiplyOp, arith::MulIOp, arith::MulFOp>;

struct LowerDivide : OpConversionPattern<ekl::DivideOp> {
    using OpConversionPattern<ekl::DivideOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::DivideOp op,
        ekl::DivideOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (!getTypeBound(op.getResult()).isInteger()) {
            rewriter.replaceOpWithNewOp<arith::DivFOp>(
                op,
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        if (getTypeBound(op.getResult()).isSignedInteger()) {
            rewriter.replaceOpWithNewOp<arith::DivSIOp>(
                op,
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        rewriter.replaceOpWithNewOp<arith::DivUIOp>(
            op,
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

//===----------------------------------------------------------------------===//

namespace {

struct ConvertEKLToArithPass
        : messner::impl::ConvertEKLToArithBase<ConvertEKLToArithPass> {
    using ConvertEKLToArithBase::ConvertEKLToArithBase;

    void runOnOperation() override;
};

} // namespace

void ConvertEKLToArithPass::runOnOperation()
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

    TypeConverter arithConverter;
    arithConverter.addConversion([](mlir::IntegerType intTy) -> Type {
        if (intTy.isSignless()) return intTy;
        return mlir::IntegerType::get(intTy.getContext(), intTy.getWidth());
    });
    arithConverter.addConversion(
        [](mlir::FloatType floatTy) -> Type { return floatTy; });

    arithConverter.addTargetMaterialization(unrealizedCast);
    arithConverter.addSourceMaterialization(unrealizedCast);

    TypeConverter eklConverter;
    eklConverter.addConversion([&](ekl::ExpressionType exprTy) -> Type {
        return arithConverter.convertType(exprTy.getTypeBound());
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

            if (eval.getType() == resultTy) return eval;
            return arithConverter
                .materializeTargetConversion(builder, loc, resultTy, {eval});
        });
    eklConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Value {
            // Wrap expressions using the IntroOp.
            if (inputs.size() != 1) return {};
            const auto exprTy = llvm::dyn_cast<ExpressionType>(resultTy);
            if (!exprTy) return {};

            auto intro = inputs.front();
            if (intro.getType() != exprTy.getTypeBound())
                intro = arithConverter.materializeSourceConversion(
                    builder,
                    loc,
                    exprTy.getTypeBound(),
                    inputs);

            return builder.create<ekl::IntroOp>(loc, intro);
        });

    const auto isArithIllegal = [&](Operation *op) -> bool {
        const auto arithTy =
            eklConverter.convertType(op->getResult(0).getType());
        return !arithTy
            || op->getNumOperands()
                   != llvm::count(
                       op->getOperandTypes(),
                       op->getResult(0).getType());
    };

    messner::populateConvertEKLToArithPatterns(eklConverter, patterns);

    target.addDynamicallyLegalOp<ekl::LiteralOp>(isArithIllegal);
    target.addDynamicallyLegalOp<
        ekl::AddOp,
        ekl::SubtractOp,
        ekl::MultiplyOp,
        ekl::DivideOp>(isArithIllegal);
    target.addLegalDialect<arith::ArithDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void messner::populateConvertEKLToArithPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<LowerLiteral>(typeConverter, patterns.getContext());
    patterns.add<LowerAdd, LowerSubtract, LowerMultiply, LowerDivide>(
        typeConverter,
        patterns.getContext());
}

std::unique_ptr<Pass> messner::createConvertEKLToArithPass()
{
    return std::make_unique<ConvertEKLToArithPass>();
}
