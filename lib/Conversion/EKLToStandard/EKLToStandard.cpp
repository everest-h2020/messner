/// Implements the ConvertEKLToStandardPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToStandard/EKLToStandard.h"

#include "messner/Dialect/EKL/Analysis/Casting.h"
#include "messner/Dialect/EKL/Enums.h"
#include "messner/Dialect/EKL/IR/Base.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Traits.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Index/IR/IndexAttrs.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOSTANDARD
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

//===----------------------------------------------------------------------===//

static Value createUnrealizedCast(
    OpBuilder &builder,
    Type resultTy,
    ValueRange inputs,
    Location loc)
{
    if (inputs.size() != 1) return {};
    return builder.create<UnrealizedConversionCastOp>(loc, resultTy, inputs)
        .getResult(0);
};

namespace {

struct ConvertEKLToStandardPass
        : messner::impl::ConvertEKLToStandardBase<ConvertEKLToStandardPass> {
    using ConvertEKLToStandardBase::ConvertEKLToStandardBase;

    void runOnOperation() override;
};

struct ConvertLiteral : OpConversionPattern<ekl::LiteralOp> {
    using OpConversionPattern<ekl::LiteralOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LiteralOp op,
        ekl::LiteralOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (const auto indexAttr =
                llvm::dyn_cast<ekl::IndexAttr>(adaptor.getValueAttr())) {
            rewriter.replaceOpWithNewOp<index::ConstantOp>(
                op,
                std::bit_cast<int64_t>(indexAttr.getValue()));
            return success();
        }

        auto attr = llvm::cast<TypedAttr>(adaptor.getValueAttr());
        if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            // Convert to signless integer attribute.
            attr = mlir::IntegerAttr::get(
                mlir::IntegerType::get(
                    attr.getContext(),
                    intAttr.getType().getIntOrFloatBitWidth()),
                intAttr.getValue());
        }

        assert(llvm::isa<mlir::FloatAttr>(attr));
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
        return success();
    }
};

struct ConvertUnify : OpConversionPattern<ekl::UnifyOp> {
    using OpConversionPattern<ekl::UnifyOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::UnifyOp op,
        ekl::UnifyOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy     = adaptor.getOperand().getType();
        const auto resultTy = getTypeConverter()->convertType(op.getType());

        if (inTy == resultTy) {
            // index -> index and other no-op conversions.
            rewriter.replaceOp(op, adaptor.getOperand());
            return success();
        } else if (inTy.isIndex()) {
            if (llvm::isa<FloatType>(resultTy)) {
                // index -> float
                // Solve this using another legalization step that casts via the
                // direct integer type to the index type.
                const auto asInt =
                    rewriter
                        .create<ekl::UnifyOp>(
                            op.getLoc(),
                            op.getOperand(),
                            llvm::cast<ekl::IndexType>(
                                getTypeBound(op.getOperand().getType()))
                                .getIntegerType())
                        .getResult();
                rewriter.replaceOpWithNewOp<ekl::UnifyOp>(
                    op,
                    asInt,
                    getTypeBound(op.getType()));
                return success();
            } else if (resultTy.isInteger()) {
                // index -> int
                rewriter.replaceOpWithNewOp<index::CastUOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
                return success();
            }
        } else if (inTy.isInteger()) {
            const auto inSigned =
                getTypeBound(op.getOperand().getType()).isSignedInteger();
            if (resultTy.isInteger()) {
                // int -> int
                if (inSigned) {
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
            } else if (llvm::isa<FloatType>(inTy)) {
                // int -> float
                if (inSigned) {
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
        } else if (llvm::isa<FloatType>(inTy)) {
            // float -> float
            rewriter.replaceOpWithNewOp<arith::ExtFOp>(
                op,
                resultTy,
                adaptor.getOperand());
            return success();
        }

        return failure();
    }
};

struct ConvertCoerce : OpConversionPattern<ekl::CoerceOp> {
    using OpConversionPattern<ekl::CoerceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::CoerceOp op,
        ekl::CoerceOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy     = adaptor.getOperand().getType();
        const auto resultTy = getTypeConverter()->convertType(op.getType());

        if (inTy == resultTy) {
            // index -> index and other no-op conversions.
            rewriter.replaceOp(op, adaptor.getOperand());
            return success();
        } else if (inTy.isIndex()) {
            // index -> ?
            // Solve this using another legalization step that casts via the
            // direct integer type.
            const auto asInt =
                rewriter
                    .create<ekl::UnifyOp>(
                        op.getLoc(),
                        op.getOperand(),
                        llvm::cast<ekl::IndexType>(
                            getTypeBound(op.getOperand().getType()))
                            .getIntegerType())
                    .getResult();
            rewriter.replaceOpWithNewOp<ekl::CoerceOp>(
                op,
                asInt,
                getTypeBound(op.getType()));
            return success();
        } else if (resultTy.isIndex()) {
            // ? -> index
            // Solve this using another legalization step that casts via the
            // direct integer type.
            const auto toInt =
                rewriter
                    .create<ekl::CoerceOp>(
                        op.getLoc(),
                        op.getOperand(),
                        llvm::cast<ekl::IndexType>(getTypeBound(op.getType()))
                            .getIntegerType())
                    .getResult();
            rewriter.replaceOpWithNewOp<ekl::UnifyOp>(
                op,
                toInt,
                getTypeBound(op.getType()));
            return success();
        } else if (inTy.isInteger()) {
            if (resultTy.isInteger()) {
                // int -> int
                rewriter.replaceOpWithNewOp<arith::TruncIOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
                return success();
            } else if (llvm::isa<FloatType>(resultTy)) {
                // int -> float
                const auto inSigned =
                    getTypeBound(op.getOperand().getType()).isSignedInteger();
                if (inSigned) {
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
        } else if (llvm::isa<FloatType>(inTy)) {
            if (resultTy.isInteger()) {
                // float -> int
                const auto outSigned =
                    getTypeBound(op.getType()).isSignedInteger();
                if (outSigned) {
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
            } else if (llvm::isa<FloatType>(resultTy)) {
                // float -> float
                rewriter.replaceOpWithNewOp<arith::TruncFOp>(
                    op,
                    resultTy,
                    adaptor.getOperand());
                return success();
            }
        }

        return failure();
    }
};

struct ConvertNot : OpConversionPattern<ekl::LogicalNotOp> {
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

template<class Source, class Target>
struct ConvertLogical : OpConversionPattern<Source> {
    using OpConversionPattern<Source>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        Source op,
        Source::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<Target>(op, adaptor.getOperands());
        return success();
    }
};

using ConvertAnd = ConvertLogical<ekl::LogicalAndOp, arith::AndIOp>;
using ConvertOr  = ConvertLogical<ekl::LogicalOrOp, arith::OrIOp>;

struct ConvertCompare : OpConversionPattern<ekl::CompareOp> {
    using OpConversionPattern<ekl::CompareOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::CompareOp op,
        ekl::CompareOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto lhsTy = adaptor.getLhs().getType();
        if (adaptor.getRhs().getType() != lhsTy) return failure();

        if (lhsTy.isIndex()) {
            rewriter.replaceOpWithNewOp<index::CmpOp>(
                op,
                convertIndexKind(op.getKind()),
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        } else if (lhsTy.isInteger()) {
            if (lhsTy.isSignedInteger()) {
                rewriter.replaceOpWithNewOp<arith::CmpIOp>(
                    op,
                    convertSIKind(op.getKind()),
                    adaptor.getLhs(),
                    adaptor.getRhs());
            } else {
                rewriter.replaceOpWithNewOp<arith::CmpIOp>(
                    op,
                    convertUIKind(op.getKind()),
                    adaptor.getLhs(),
                    adaptor.getRhs());
            }
            return success();
        } else if (llvm::isa<FloatType>(lhsTy)) {
            rewriter.replaceOpWithNewOp<arith::CmpFOp>(
                op,
                convertFloatKind(op.getKind()),
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        return failure();
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
    static index::IndexCmpPredicate convertIndexKind(RelationKind kind)
    {
        switch (kind) {
        case RelationKind::Equivalent:     return index::IndexCmpPredicate::EQ;
        case RelationKind::Antivalent:     return index::IndexCmpPredicate::NE;
        case RelationKind::LessThan:       return index::IndexCmpPredicate::ULT;
        case RelationKind::LessOrEqual:    return index::IndexCmpPredicate::ULE;
        case RelationKind::GreaterOrEqual: return index::IndexCmpPredicate::UGE;
        case RelationKind::GreaterThan:    return index::IndexCmpPredicate::UGT;
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

template<
    class Source,
    class TargetUI,
    class TargetSI,
    class TargetF,
    class TargetIdx = void>
struct ConvertBinary : OpConversionPattern<Source> {
    using OpConversionPattern<Source>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        Source op,
        Source::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto lhsTy = adaptor.getLhs().getType();
        if (adaptor.getRhs().getType() != lhsTy) return failure();

        if constexpr (!std::is_void_v<TargetIdx>) {
            if (lhsTy.isIndex()) {

                rewriter.replaceOpWithNewOp<TargetIdx>(
                    op,
                    adaptor.getLhs(),
                    adaptor.getRhs());
                return success();
            }
        }

        if (lhsTy.isInteger()) {
            if (lhsTy.isSignedInteger()) {
                rewriter.replaceOpWithNewOp<TargetSI>(
                    op,
                    adaptor.getLhs(),
                    adaptor.getRhs());
            } else {
                rewriter.replaceOpWithNewOp<TargetUI>(
                    op,
                    adaptor.getLhs(),
                    adaptor.getRhs());
            }
            return success();
        } else if (llvm::isa<FloatType>(lhsTy)) {
            rewriter.replaceOpWithNewOp<TargetF>(
                op,
                adaptor.getLhs(),
                adaptor.getRhs());
            return success();
        }

        return failure();
    }
};

using ConvertMin = ConvertBinary<
    ekl::MinOp,
    arith::MinUIOp,
    arith::MinSIOp,
    arith::MinNumFOp,
    index::MinUOp>;
using ConvertMax = ConvertBinary<
    ekl::MaxOp,
    arith::MaxUIOp,
    arith::MaxSIOp,
    arith::MaxNumFOp,
    index::MaxUOp>;

struct ConvertNegate : OpConversionPattern<ekl::NegateOp> {
    using OpConversionPattern<ekl::NegateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::NegateOp op,
        ekl::NegateOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy = adaptor.getOperand().getType();
        if (inTy.isInteger()) {
            const auto zero = rewriter
                                  .create<arith::ConstantOp>(
                                      op.getLoc(),
                                      rewriter.getIntegerAttr(inTy, 0))
                                  .getResult();

            rewriter.replaceOpWithNewOp<arith::SubIOp>(
                op,
                zero,
                adaptor.getOperand());
            return success();
        } else if (llvm::isa<FloatType>(inTy)) {
            rewriter.replaceOpWithNewOp<math::PowFOp>(op, adaptor.getOperand());
            return success();
        }

        return failure();
    }
};

using ConvertAdd = ConvertBinary<
    ekl::AddOp,
    arith::AddIOp,
    arith::AddIOp,
    arith::AddFOp,
    index::AddOp>;
using ConvertSubtract = ConvertBinary<
    ekl::SubtractOp,
    arith::SubIOp,
    arith::SubIOp,
    arith::SubFOp,
    index::SubOp>;
using ConvertMultiply = ConvertBinary<
    ekl::MultiplyOp,
    arith::MulIOp,
    arith::MulIOp,
    arith::MulFOp,
    index::MulOp>;
using ConvertDivide = ConvertBinary<
    ekl::DivideOp,
    arith::DivUIOp,
    arith::DivSIOp,
    arith::DivFOp,
    index::DivUOp>;
using ConvertRemainder = ConvertBinary<
    ekl::RemainderOp,
    arith::RemUIOp,
    arith::RemSIOp,
    arith::RemFOp,
    index::RemUOp>;
using ConvertPower =
    ConvertBinary<ekl::PowerOp, math::IPowIOp, math::IPowIOp, math::PowFOp>;

} // namespace

void ConvertEKLToStandardPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Create a helper TypeConverter to provide arith & index op types.
    TypeConverter stdConverter;
    {
        // Float types pass through unchanged.
        stdConverter.addConversion([](mlir::FloatType type) { return type; });
        // Integer types must be converted to signless integers.
        stdConverter.addConversion([](mlir::IntegerType type) {
            if (type.isSignless()) return type;
            return mlir::IntegerType::get(type.getContext(), type.getWidth());
        });
        // The index type is converted to the mlir index type.
        stdConverter.addConversion([](ekl::IndexType type) {
            return mlir::IndexType::get(type.getContext());
        });

        // In any case, none of these conversions are actually performed, we
        // always use an unrealized cast rely on them going away once everything
        // is converted to standard.
        stdConverter.addTargetMaterialization(createUnrealizedCast);
        stdConverter.addSourceMaterialization(createUnrealizedCast);
    }

    TypeConverter converter;
    {
        // Unpack the contained expression type and convert it with the helper.
        converter.addConversion([&](ekl::ExpressionType exprTy) -> Type {
            if (const auto boundTy = exprTy.getTypeBound())
                return stdConverter.convertType(boundTy);
            return {};
        });

        // Unwrap expressions as values using the ekl.eval op.
        converter.addTargetMaterialization(
            [&](OpBuilder &builder,
                Type resultTy,
                ValueRange inputs,
                Location loc) -> Value {
                if (inputs.size() != 1) return {};
                const auto exprTy = llvm::dyn_cast<ekl::ExpressionType>(
                    inputs.front().getType());
                if (!exprTy) return {};

                const auto eval = builder
                                      .create<ekl::EvalOp>(
                                          loc,
                                          inputs.front(),
                                          exprTy.getTypeBound())
                                      .getResult();

                if (eval.getType() == resultTy) return eval;

                // Use the helper to perform the standard conversion.
                return stdConverter.materializeTargetConversion(
                    builder,
                    loc,
                    resultTy,
                    {eval});
            });

        // Wrap values as expressions using the ekl.intro op.
        converter.addSourceMaterialization(
            [&](OpBuilder &builder,
                ekl::ExpressionType resultTy,
                ValueRange inputs,
                Location loc) -> Value {
                if (inputs.size() != 1) return {};

                auto intro = inputs.front();
                if (intro.getType() != resultTy.getTypeBound()) {
                    // Use the helper to perform the standard conversion.
                    intro = stdConverter.materializeSourceConversion(
                        builder,
                        loc,
                        resultTy.getTypeBound(),
                        inputs);
                }

                return builder.create<ekl::IntroOp>(loc, intro);
            });
    }

    messner::populateConvertEKLToStandardPatterns(converter, patterns);

    // A type is illegal if we don't know how to convert it.
    const auto isIllegalType = [&](Type type) {
        return !converter.convertType(type);
    };
    // An op is illegal if its operand types don't match, or any of its operand
    // or result types is illegal.
    const auto isIllegalOp = [&](Operation *op) {
        if (const auto numOps = op->getNumOperands()) {
            if (llvm::count(op->getOperandTypes(), op->getOperand(0).getType())
                != numOps)
                return true;
        }

        return llvm::any_of(op->getOperandTypes(), isIllegalType)
            || llvm::any_of(op->getResultTypes(), isIllegalType);
    };

    target.addDynamicallyLegalOp<ekl::LiteralOp>(isIllegalOp);
    target.addDynamicallyLegalOp<ekl::UnifyOp>(isIllegalOp);
    target.addDynamicallyLegalOp<ekl::CoerceOp>(isIllegalOp);
    target.addDynamicallyLegalDialect<ekl::EKLDialect>([&](Operation *op) {
        // All logical ops must rewrite to arith.
        if (op->hasTrait<ekl::OpTrait::IsLogical>()) return false;
        // All legalizable relational ops rewrite to arith or index.
        if (op->hasTrait<ekl::OpTrait::IsRelational>()) return isIllegalOp(op);
        // All legalizable arithmetic ops rewrite to arith, index or math.
        if (op->hasTrait<ekl::OpTrait::IsArithmetic>()) return isIllegalOp(op);

        // All other ops are assumed legal.
        return true;
    });

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<index::IndexDialect>();
    target.addLegalDialect<math::MathDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void messner::populateConvertEKLToStandardPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertLiteral,
        ConvertUnify,
        ConvertCoerce,
        ConvertNot,
        ConvertAnd,
        ConvertOr,
        ConvertCompare,
        ConvertMin,
        ConvertMax,
        ConvertNegate,
        ConvertAdd,
        ConvertSubtract,
        ConvertMultiply,
        ConvertDivide,
        ConvertRemainder,
        ConvertPower>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> messner::createConvertEKLToStandardPass()
{
    return std::make_unique<ConvertEKLToStandardPass>();
}
