/// Implements the ConvertEKLToArithPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToArith/EKLToArith.h"

#include "messner/Dialect/EKL/Analysis/Casting.h"
#include "messner/Dialect/EKL/Enums.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
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

struct LowerUnify : OpConversionPattern<ekl::UnifyOp> {
    using OpConversionPattern<ekl::UnifyOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::UnifyOp op,
        ekl::UnifyOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy = getTypeBound(op.getOperand());
        const auto resultTy =
            getTypeConverter()->convertType(op.getResult().getType());

        if (adaptor.getOperand().getType() == resultTy) {
            // Eliminate no-op casts.
            rewriter.replaceOp(op, {adaptor.getOperand()});
            return success();
        }

        if (resultTy.isInteger()) {
            // Can only be integer extension.
            if (inTy.isSignedInteger()) {
                rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            } else {
                rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            }
            return success();
        }

        if (inTy.isInteger()) {
            // Can only be int-to-float cast.
            if (inTy.isSignedInteger()) {
                rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            } else {
                rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            }
            return success();
        }

        // Can only be float extension.
        rewriter.replaceOpWithNewOp<arith::ExtFOp>(
            op,
            resultTy,
            adaptor.getOperand());
        return success();
    }
};

struct LowerCoerce : OpConversionPattern<ekl::CoerceOp> {
    using OpConversionPattern<ekl::CoerceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::CoerceOp op,
        ekl::CoerceOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy = getTypeBound(op.getOperand());
        const auto resultTy =
            getTypeConverter()->convertType(op.getResult().getType());

        if (adaptor.getOperand().getType() == resultTy) {
            // Eliminate no-op casts.
            rewriter.replaceOp(op, {adaptor.getOperand()});
            return success();
        }

        if (!resultTy.isInteger() && !inTy.isInteger()) {
            // Can only be float truncation.
            rewriter.replaceOpWithNewOp<arith::TruncFOp>(
                op,
                resultTy,
                adaptor.getOperand());
            return success();
        }

        if (resultTy.isInteger() && inTy.isInteger()) {
            // Can only be integer truncation.
            rewriter.replaceOpWithNewOp<arith::TruncIOp>(
                op,
                resultTy,
                adaptor.getOperand());
            return success();
        }

        if (resultTy.isInteger() && !inTy.isInteger()) {
            // Can only be float-to-int cast.
            if (getTypeBound(op.getResult()).isSignedInteger()) {
                rewriter.replaceOpWithNewOp<arith::FPToSIOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            } else {
                rewriter.replaceOpWithNewOp<arith::FPToUIOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
            }
            return success();
        }

        // Can only be int-to-float cast.
        if (inTy.isSignedInteger()) {
            rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
                op,
                resultTy,
                adaptor.getOperand());
        } else {
            rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
                op,
                resultTy,
                adaptor.getOperand());
        }
        return success();
    }
};

struct LowerCompare : OpConversionPattern<ekl::CompareOp> {
    using OpConversionPattern<ekl::CompareOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::CompareOp op,
        ekl::CompareOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (!getTypeBound(op.getLhs()).isInteger()) {
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(
                op,
                convertFloatKind(op.getKind()),
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        if (getTypeBound(op.getLhs()).isSignedInteger()) {
            rewriter.replaceOpWithNewOp<arith::CmpIOp>(
                op,
                convertSIKind(op.getKind()),
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            op,
            convertUIKind(op.getKind()),
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }

private:
    [[nodiscard]]
    static arith::CmpFPredicate convertFloatKind(RelationKind kind)
    {
        switch (kind) {
        case RelationKind::Equivalent:     return arith::CmpFPredicate::OEQ;
        case RelationKind::Antivalent:     return arith::CmpFPredicate::ONE;
        case RelationKind::LessThan:       return arith::CmpFPredicate::OLT;
        case RelationKind::LessOrEqual:    return arith::CmpFPredicate::OLE;
        case RelationKind::GreaterOrEqual: return arith::CmpFPredicate::OGE;
        case RelationKind::GreaterThan:    return arith::CmpFPredicate::OGT;
        }
    }

    [[nodiscard]]
    static arith::CmpIPredicate convertUIKind(RelationKind kind)
    {
        switch (kind) {
        case RelationKind::Equivalent:     return arith::CmpIPredicate::eq;
        case RelationKind::Antivalent:     return arith::CmpIPredicate::ne;
        case RelationKind::LessThan:       return arith::CmpIPredicate::ult;
        case RelationKind::LessOrEqual:    return arith::CmpIPredicate::ule;
        case RelationKind::GreaterOrEqual: return arith::CmpIPredicate::uge;
        case RelationKind::GreaterThan:    return arith::CmpIPredicate::ugt;
        }
    }

    [[nodiscard]]
    static arith::CmpIPredicate convertSIKind(RelationKind kind)
    {
        switch (kind) {
        case RelationKind::Equivalent:     return arith::CmpIPredicate::eq;
        case RelationKind::Antivalent:     return arith::CmpIPredicate::ne;
        case RelationKind::LessThan:       return arith::CmpIPredicate::slt;
        case RelationKind::LessOrEqual:    return arith::CmpIPredicate::sle;
        case RelationKind::GreaterOrEqual: return arith::CmpIPredicate::sge;
        case RelationKind::GreaterThan:    return arith::CmpIPredicate::sgt;
        }
    }
};

struct LowerNot : OpConversionPattern<ekl::LogicalNotOp> {
    using OpConversionPattern<ekl::LogicalNotOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LogicalNotOp op,
        ekl::LogicalNotOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto one = rewriter
                             .create<arith::ConstantOp>(
                                 op.getLoc(),
                                 rewriter.getBoolAttr(true))
                             .getResult();

        rewriter.replaceOpWithNewOp<arith::XOrIOp>(
            op,
            adaptor.getOperand(),
            one);
        return success();
    }
};

struct LowerAnd : OpConversionPattern<ekl::LogicalAndOp> {
    using OpConversionPattern<ekl::LogicalAndOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LogicalAndOp op,
        ekl::LogicalAndOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<arith::AndIOp>(
            op,
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

struct LowerOr : OpConversionPattern<ekl::LogicalOrOp> {
    using OpConversionPattern<ekl::LogicalOrOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LogicalOrOp op,
        ekl::LogicalOrOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<arith::OrIOp>(
            op,
            adaptor.getLhs(),
            adaptor.getRhs());
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

template<class Source, class TargetUI, class TargetSI, class TargetF>
struct LowerClosedBinarySignedOp : OpConversionPattern<Source> {
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

        if (getTypeBound(op.getResult()).isSignedInteger()) {
            rewriter.replaceOpWithNewOp<TargetSI>(
                op,
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        rewriter.replaceOpWithNewOp<TargetUI>(
            op,
            adaptor.getLhs(),
            adaptor.getRhs());
        return success();
    }
};

using LowerMin = LowerClosedBinarySignedOp<
    ekl::MinOp,
    arith::MinUIOp,
    arith::MinSIOp,
    arith::MinNumFOp>;
using LowerMax = LowerClosedBinarySignedOp<
    ekl::MaxOp,
    arith::MaxUIOp,
    arith::MaxSIOp,
    arith::MaxNumFOp>;

struct LowerNegate : OpConversionPattern<ekl::NegateOp> {
    using OpConversionPattern<ekl::NegateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::NegateOp op,
        ekl::NegateOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (!getTypeBound(op.getResult()).isInteger()) {
            rewriter.replaceOpWithNewOp<arith::NegFOp>(
                op,
                adaptor.getOperand());
            return success();
        }

        const auto zero =
            rewriter
                .create<arith::ConstantOp>(
                    op.getLoc(),
                    rewriter.getIntegerAttr(adaptor.getOperand().getType(), 0))
                .getResult();

        rewriter.replaceOpWithNewOp<arith::SubIOp>(
            op,
            zero,
            adaptor.getOperand());
        return success();
    }
};

using LowerAdd = LowerClosedBinaryOp<ekl::AddOp, arith::AddIOp, arith::AddFOp>;
using LowerSubtract =
    LowerClosedBinaryOp<ekl::SubtractOp, arith::SubIOp, arith::SubFOp>;
using LowerMultiply =
    LowerClosedBinaryOp<ekl::MultiplyOp, arith::MulIOp, arith::MulFOp>;
using LowerDivide = LowerClosedBinarySignedOp<
    ekl::DivideOp,
    arith::DivUIOp,
    arith::DivSIOp,
    arith::DivFOp>;
using LowerRemainder = LowerClosedBinarySignedOp<
    ekl::RemainderOp,
    arith::RemUIOp,
    arith::RemSIOp,
    arith::RemFOp>;

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

    const auto isCastArithIllegal = [&](Operation *op) -> bool {
        return !eklConverter.convertType(op->getOperand(0).getType())
            || !eklConverter.convertType(op->getResult(0).getType());
    };
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

    target.addDynamicallyLegalOp<ekl::UnifyOp, ekl::CoerceOp>(
        isCastArithIllegal);

    target.addDynamicallyLegalOp<ekl::LiteralOp>(isArithIllegal);
    target.addDynamicallyLegalOp<ekl::CompareOp>([&](ekl::CompareOp op) {
        if (op.getLhs().getType() != op.getRhs().getType()) return true;
        return !eklConverter.convertType(op.getLhs().getType());
    });
    target.addDynamicallyLegalOp<ekl::MinOp, ekl::MaxOp>(isArithIllegal);
    target
        .addIllegalOp<ekl::LogicalNotOp, ekl::LogicalAndOp, ekl::LogicalOrOp>();
    target.addDynamicallyLegalOp<
        ekl::NegateOp,
        ekl::AddOp,
        ekl::SubtractOp,
        ekl::MultiplyOp,
        ekl::DivideOp,
        ekl::RemainderOp>(isArithIllegal);
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
    patterns.add<LowerUnify, LowerCoerce>(typeConverter, patterns.getContext());
    patterns.add<LowerNot, LowerAnd, LowerOr>(
        typeConverter,
        patterns.getContext());
    patterns.add<LowerCompare, LowerMin, LowerMax>(
        typeConverter,
        patterns.getContext());
    patterns.add<
        LowerNegate,
        LowerAdd,
        LowerSubtract,
        LowerMultiply,
        LowerDivide,
        LowerRemainder>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> messner::createConvertEKLToArithPass()
{
    return std::make_unique<ConvertEKLToArithPass>();
}
