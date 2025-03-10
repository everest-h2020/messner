/// Implements the ConvertEKLToLinalgPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToLinalg/EKLToLinalg.h"

#include "../EKLConverter.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/Base.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/ReshapeOpsUtils.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

using namespace mlir;
using namespace mlir::ekl;

[[nodiscard]]
static ArrayRef<int64_t> toShape(ExtentRange extents)
{
    return {reinterpret_cast<const int64_t *>(extents.data()), extents.size()};
}

[[nodiscard]]
AffineMap getMajorId(unsigned dims, unsigned results, MLIRContext *context)
{
    auto id = AffineMap::getMultiDimIdentityMap(dims, context);
    return AffineMap::get(
        dims,
        0,
        id.getResults().take_front(results),
        context);
}

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOLINALG
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

//===----------------------------------------------------------------------===//

namespace {

struct ConvertEKLToLinalgPass
        : messner::impl::ConvertEKLToLinalgBase<ConvertEKLToLinalgPass> {
    using ConvertEKLToLinalgBase::ConvertEKLToLinalgBase;

    void runOnOperation() override;
};

struct ConvertLiteral : OpConversionPattern<ekl::LiteralOp> {
    using OpConversionPattern<ekl::LiteralOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::LiteralOp op,
        ekl::LiteralOp::Adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto outTy = llvm::dyn_cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getType()));
        if (!outTy) return failure();

        DenseElementsAttr dense;
        const auto value = llvm::cast<ekl::ArrayAttr>(op.getValue());
        if (const auto splat = value.getSplatValue()) {
            dense = DenseElementsAttr::get(outTy, splat);
        } else {
            // TODO: Implement.
            return failure();
        }

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, dense);
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
        const auto outTy = llvm::dyn_cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getType()));
        if (!outTy || adaptor.getOperand().getType() != outTy.getElementType())
            return failure();

        rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
            op,
            outTy,
            adaptor.getOperand());
        return success();
    }
};

struct ConvertZip : OpConversionPattern<ekl::ZipOp> {
    using OpConversionPattern<ekl::ZipOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::ZipOp op,
        ekl::ZipOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::dyn_cast<ekl::ArrayType>(getTypeBound(op.getType()));
        if (!arrayTy) return failure();
        const auto tensorTy = llvm::cast<RankedTensorType>(
            getTypeConverter()->convertType(arrayTy));

        const auto init = rewriter
                              .create<tensor::EmptyOp>(
                                  op.getLoc(),
                                  tensorTy.getShape(),
                                  tensorTy.getElementType())
                              .getResult();

        SmallVector<AffineMap> indexMaps(
            adaptor.getOperands().size(),
            AffineMap::getMultiDimIdentityMap(
                arrayTy.getNumExtents(),
                rewriter.getContext()));
        SmallVector<utils::IteratorType> iterTypes(
            adaptor.getOperands().size(),
            utils::IteratorType::parallel);

        const auto generic = rewriter.create<linalg::GenericOp>(
            op.getLoc(),
            TypeRange{tensorTy},
            adaptor.getOperands(),
            ValueRange{init},
            indexMaps,
            iterTypes);

        // TODO: Inline block.

        rewriter.replaceOp(op, generic);
        return success();
    }
};

struct ConvertChoice : OpConversionPattern<ekl::ChoiceOp> {
    using OpConversionPattern<ekl::ChoiceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::ChoiceOp op,
        ekl::ChoiceOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        if (!adaptor.getSelector().getType().isSignlessInteger(1))
            return failure();
        if (!llvm::isa<ArrayType>(getTypeBound(op.getType()))) return failure();

        const auto makeMemref = [&](Value operand) {
            const auto tensorTy =
                llvm::cast<RankedTensorType>(operand.getType());
            const auto memrefTy = mlir::MemRefType::get(
                tensorTy.getShape(),
                tensorTy.getElementType());
            return rewriter
                .create<bufferization::ToMemrefOp>(
                    op.getLoc(),
                    memrefTy,
                    operand,
                    true)
                .getResult();
        };

        const auto selected = rewriter
                                  .create<arith::SelectOp>(
                                      op.getLoc(),
                                      adaptor.getSelector(),
                                      makeMemref(adaptor.getAlternatives()[0]),
                                      makeMemref(adaptor.getAlternatives()[1]))
                                  .getResult();
        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
            op,
            selected,
            true);
        return success();
    }
};

struct ConvertStack : OpConversionPattern<ekl::StackOp> {
    using OpConversionPattern<ekl::StackOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::StackOp op,
        ekl::StackOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto tensorTy = llvm::cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getType()));

        SmallVector<int64_t> inShape(tensorTy.getShape());
        inShape.front() = 1;
        const auto inTy =
            RankedTensorType::get(inShape, tensorTy.getElementType());
        SmallVector<ReassociationIndices> inReassoc;

        if (inShape.size() > 1) {
            for (auto i : llvm::iota_range<int64_t>(1, inShape.size(), false))
                inReassoc.emplace_back().push_back(i);
            inReassoc.front().insert(inReassoc.front().begin(), 0);
        }

        SmallVector<Value> ins;
        for (auto in : adaptor.getOperands())
            ins.push_back(rewriter
                              .create<tensor::ExpandShapeOp>(
                                  op.getLoc(),
                                  inTy,
                                  in,
                                  inReassoc)
                              .getResult());

        rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, 0, ins);
        return success();
    }
};

// struct AffineExprBuilder {
//     AffineExprBuilder(MLIRContext *context) : context(context), cache() {}

//     [[nodiscard]] AffineExpr build(Value value)
//     {
//         const auto it = cache.find(value);
//         if (it != cache.end()) return it->second;

//         if (!llvm::isa<mlir::IndexType, ekl::IndexType>(
//                 getTypeBound(value.getType())))
//             return {};

//         const auto source = value.getDefiningOp();
//         if (!source) return {};
//         auto result =
//             llvm::TypeSwitch<Operation *, AffineExpr>(source)
//                 .Case(
//                     [&](ekl::IntroOp intro) { return build(intro.getValue());
//                     })
//                 .Case([&](ekl::EvalOp eval) { return build(eval.getValue());
//                 }) .Case([&](ekl::AddOp add) -> AffineExpr {
//                     const auto lhs = build(add.getLhs());
//                     const auto rhs = build(add.getRhs());
//                     if (!lhs || !rhs) return {};
//                     return lhs + rhs;
//                 })
//                 .Case([&](ekl::SubtractOp add) -> AffineExpr {
//                     const auto lhs = build(add.getLhs());
//                     const auto rhs = build(add.getRhs());
//                     if (!lhs || !rhs) return {};
//                     return lhs - rhs;
//                 })
//                 .Case([&](ekl::MultiplyOp add) -> AffineExpr {
//                     const auto lhs = build(add.getLhs());
//                     const auto rhs = build(add.getRhs());
//                     if (!lhs || !rhs) return {};
//                     return lhs * rhs;
//                 })
//                 .Case([&](ekl::LiteralOp literal) {
//                     return getAffineConstantExpr(
//                         llvm::cast<ekl::IndexAttr>(literal.getValue())
//                             .getValue(),
//                         context);
//                 })
//                 .Case([&](index::ConstantOp index) {
//                     return getAffineConstantExpr(
//                         index.getValue().getZExtValue(),
//                         context);
//                 })
//                 .Default([](auto) -> AffineExpr { return {}; });

//         cache.insert({value, result});
//         return result;
//     }

//     [[nodiscard]] AffineMap
//     build(unsigned dims, unsigned syms, ValueRange values)
//     {
//         SmallVector<AffineExpr> exprs;
//         for (auto value : values) {
//             exprs.push_back(build(value));
//             if (!exprs.back()) return {};
//         }

//         return AffineMap::get(dims, syms, exprs, context);
//     }

//     MLIRContext *context;
//     DenseMap<Value, AffineExpr> cache;
// };

struct ConvertAssoc : OpConversionPattern<ekl::AssocOp> {
    using OpConversionPattern<ekl::AssocOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::AssocOp outer,
        ekl::AssocOp::Adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        return generate(outer, rewriter);

        // auto reduce =
        // outer.getMapExpression().getDefiningOp<ekl::ReduceOp>(); ekl::AssocOp
        // inner; if (reduce) inner =
        // reduce.getArray().getDefiningOp<ekl::AssocOp>();

        // AffineExprBuilder exprs(getContext());
        // unsigned numDims = 0;
        // for (auto arg : outer.getMap()->getArguments())
        //     exprs.cache.insert(
        //         {arg, getAffineDimExpr(numDims++, exprs.context)});
        // const unsigned numOuter = numDims;
        // if (inner)
        //     for (auto arg : inner.getMap()->getArguments())
        //         exprs.cache.insert(
        //             {arg, getAffineDimExpr(numDims++, exprs.context)});

        // SmallVector<Value> inputs;
        // DenseMap<ekl::SubscriptOp, unsigned> inputMap;
        // const auto collectSubscripts = [&](ekl::AssocOp root) {
        //     for (auto subscript : root.getOps<ekl::SubscriptOp>()) {
        //         const auto inTy = getTypeConverter()->convertType(
        //             subscript.getArray().getType());
        //         const auto input =
        //             getTypeConverter()->materializeTargetConversion(
        //                 rewriter,
        //                 subscript.getLoc(),
        //                 inTy,
        //                 subscript.getArray());
        //         inputs.push_back(input);
        //         inputMap.insert({subscript, inputMap.size()});
        //     }
        // };
        // collectSubscripts(outer);
        // if (inner) collectSubscripts(inner);

        // SmallVector<utils::IteratorType> iterTypes(
        //     numOuter,
        //     utils::IteratorType::parallel);
        // iterTypes.resize(numDims, utils::IteratorType::reduction);

        // SmallVector<AffineMap> iterMaps(inputMap.size());
        // for (auto [subscript, i] : inputMap) {
        //     const auto map = exprs.build(numDims, 0,
        //     subscript.getSubscripts()); if (!map) return generate(outer,
        //     rewriter); iterMaps[i] = map;
        // }
        // iterMaps.push_back(getMajorId(numDims, numOuter, getContext()));

        // const auto tensorTy = llvm::cast<RankedTensorType>(
        //     getTypeConverter()->convertType(outer.getType()));
        // const auto init = rewriter
        //                       .create<tensor::EmptyOp>(
        //                           outer.getLoc(),
        //                           tensorTy.getShape(),
        //                           tensorTy.getElementType())
        //                       .getResult();

        // auto generic = rewriter.create<linalg::GenericOp>(
        //     outer.getLoc(),
        //     TypeRange{tensorTy},
        //     inputs,
        //     ValueRange{init},
        //     iterMaps,
        //     iterTypes);

        // auto &body = generic.getBodyRegion().emplaceBlock();
        // for (auto in : inputs) {
        //     const auto scalarTy =
        //         llvm::cast<RankedTensorType>(in.getType()).getElementType();
        //     body.addArgument(scalarTy, rewriter.getUnknownLoc());
        // }
        // body.addArgument(tensorTy.getElementType(),
        // rewriter.getUnknownLoc()); rewriter.setInsertionPointToStart(&body);
        // for (auto [subscript, i] : inputMap)
        //     rewriter.replaceOp(subscript, body.getArgument(i));

        // const auto zero = rewriter
        //                       .create<ekl::LiteralOp>(
        //                           outer.getLoc(),
        //                           ekl::IndexAttr::get(getContext(), 0))
        //                       .getResult();
        // SmallVector<Value> args(outer.getMap()->getNumArguments(), zero);
        // rewriter.inlineBlockBefore(outer.getMap(), &body, body.end(), args);

        // rewriter.setInsertionPoint(&body.back());
        // const auto eval =
        //     rewriter
        //         .create<ekl::EvalOp>(
        //             body.back().getLoc(),
        //             body.back().getOperand(0),
        //             getTypeBound(body.back().getOperand(0).getType()))
        //         .getResult();
        // rewriter.replaceOpWithNewOp<linalg::YieldOp>(
        //     &body.back(),
        //     createUnrealizedCast(
        //         rewriter,
        //         tensorTy.getElementType(),
        //         {eval},
        //         body.back().getLoc()));
        // rewriter.replaceOp(outer, generic);

        // if (inner) {
        //     args.assign(inner.getMap()->getNumArguments(), zero);
        //     rewriter.inlineBlockBefore(inner.getMap(), inner, args);

        //     const auto rhs = inner->getPrevNode()->getOperand(0);
        //     rewriter.eraseOp(inner->getPrevNode());
        //     rewriter.eraseOp(inner);

        //     rewriter.setInsertionPoint(reduce);
        //     const auto lhs = rewriter
        //                          .create<ekl::IntroOp>(
        //                              reduce.getLoc(),
        //                              createUnrealizedCast(
        //                                  rewriter,
        //                                  getTypeBound(rhs.getType()),
        //                                  {body.getArguments().back()},
        //                                  reduce.getLoc()))
        //                          .getResult();

        //     rewriter.inlineBlockBefore(
        //         reduce.getReduction(),
        //         reduce,
        //         {lhs, rhs});
        //     const auto res = reduce->getPrevNode()->getOperand(0);
        //     rewriter.eraseOp(reduce->getPrevNode());
        //     rewriter.replaceOp(reduce, res);
        // }

        // return success();
    }

private:
    LogicalResult
    generate(ekl::AssocOp op, ConversionPatternRewriter &rewriter) const
    {
        if (!llvm::isa<ScalarType>(getTypeBound(op.getMapExpression())))
            return failure();

        const auto tensorTy = llvm::cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getType()));
        auto generate = rewriter.create<tensor::GenerateOp>(
            op.getLoc(),
            tensorTy,
            ValueRange{});

        auto &body = generate.getBody().emplaceBlock();
        rewriter.setInsertionPointToStart(&body);

        SmallVector<Value> args;
        for (auto arg : op.getMap()->getArguments()) {
            const auto idx =
                body.addArgument(rewriter.getIndexType(), arg.getLoc());
            const auto cast = createUnrealizedCast(
                rewriter,
                getTypeBound(arg.getType()),
                {idx},
                arg.getLoc());
            args.push_back(
                rewriter.create<ekl::IntroOp>(arg.getLoc(), cast).getResult());
        }
        rewriter.inlineBlockBefore(op.getMap(), &body, body.end(), args);
        rewriter.setInsertionPoint(&body.back());
        const auto eval =
            rewriter
                .create<ekl::EvalOp>(
                    body.back().getLoc(),
                    body.back().getOperand(0),
                    getTypeBound(body.back().getOperand(0).getType()))
                .getResult();
        rewriter.replaceOpWithNewOp<tensor::YieldOp>(
            &body.back(),
            createUnrealizedCast(
                rewriter,
                tensorTy.getElementType(),
                {eval},
                body.back().getLoc()));
        rewriter.replaceOp(op, generate);
        return success();
    }
};

struct ConvertReduce : OpConversionPattern<ekl::ReduceOp> {
    using OpConversionPattern<ekl::ReduceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::ReduceOp op,
        ekl::ReduceOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto inTy =
            llvm::cast<RankedTensorType>(adaptor.getArray().getType());

        const ReassociationIndices reassoc = llvm::to_vector(
            llvm::iota_range<int64_t>(0, inTy.getRank(), false));
        const auto collapsed = rewriter
                                   .create<tensor::CollapseShapeOp>(
                                       op.getLoc(),
                                       adaptor.getArray(),
                                       ArrayRef<ReassociationIndices>{reassoc})
                                   .getResult();

        const auto init = rewriter
                              .create<tensor::EmptyOp>(
                                  op.getLoc(),
                                  ArrayRef<int64_t>{},
                                  inTy.getElementType())
                              .getResult();

        auto reduce = rewriter.create<linalg::ReduceOp>(
            op.getLoc(),
            TypeRange{init.getType()},
            ValueRange{collapsed},
            ValueRange{init},
            ArrayRef<int64_t>{0});

        auto &combiner = reduce.getCombiner().emplaceBlock();
        rewriter.setInsertionPointToStart(&combiner);
        const auto lhs = rewriter
                             .create<ekl::IntroOp>(
                                 op.getLoc(),
                                 combiner.addArgument(
                                     inTy.getElementType(),
                                     op.getBody()->getArgument(0).getLoc()))
                             .getResult();
        const auto rhs = rewriter
                             .create<ekl::IntroOp>(
                                 op.getLoc(),
                                 combiner.addArgument(
                                     inTy.getElementType(),
                                     op.getBody()->getArgument(1).getLoc()))
                             .getResult();
        rewriter.inlineBlockBefore(
            op.getBody(),
            &combiner,
            combiner.end(),
            {lhs, rhs});
        const auto yield = &combiner.back();
        rewriter.setInsertionPoint(yield);
        const auto eval = rewriter
                              .create<ekl::EvalOp>(
                                  op.getLoc(),
                                  yield->getOperand(0),
                                  inTy.getElementType())
                              .getResult();
        rewriter.replaceOpWithNewOp<linalg::YieldOp>(yield, eval);
        rewriter.setInsertionPointAfter(op);
        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
            op,
            reduce.getResult(0),
            ValueRange{});
        return success();
    }
};

struct ConvertSubscript : OpConversionPattern<ekl::SubscriptOp> {
    using OpConversionPattern<ekl::SubscriptOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::SubscriptOp op,
        ekl::SubscriptOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        const auto scalarTy = llvm::dyn_cast_if_present<ScalarType>(
            getTypeConverter()->convertType(op.getType()));
        if (!scalarTy) return failure();

        rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
            op,
            adaptor.getArray(),
            adaptor.getSubscripts());
        return success();
    }
};

struct ConvertWrite : OpConversionPattern<ekl::WriteOp> {
    using OpConversionPattern<ekl::WriteOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::WriteOp op,
        ekl::WriteOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
            op,
            Type{},
            adaptor.getValue(),
            adaptor.getReference(),
            true,
            true);
        return success();
    }
};

} // namespace

void ConvertEKLToLinalgPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    auto stdConverter = createStandardConverter();

    stdConverter.addConversion([&](ekl::ArrayType arrayTy) {
        return mlir::RankedTensorType::get(
            toShape(arrayTy.getExtents()),
            stdConverter.convertType(arrayTy.getScalarType()));
    });
    stdConverter.addConversion([&](ekl::ReferenceType refTy) {
        return mlir::MemRefType::get(
            toShape(refTy.getExtents()),
            stdConverter.convertType(refTy.getScalarType()));
    });

    auto converter = createEKLConverter(stdConverter);

    messner::populateConvertEKLToLinalgPatterns(converter, patterns);

    const auto isInAssoc = [](Operation *op) {
        return op->getParentOfType<ekl::AssocOp>();
    };

    target.addDynamicallyLegalOp<ekl::LiteralOp>([](ekl::LiteralOp op) {
        return !llvm::isa_and_present<ArrayType>(getTypeBound(op.getType()));
    });
    target.addDynamicallyLegalOp<ekl::UnifyOp>([](ekl::UnifyOp op) {
        return !llvm::isa_and_present<ArrayType>(getTypeBound(op.getType()))
            || !llvm::isa_and_present<ScalarType>(
                getTypeBound(op.getOperand()));
    });
    target.addIllegalOp<ekl::ZipOp>();
    target.addIllegalOp<ekl::ChoiceOp>();
    target.addIllegalOp<ekl::StackOp>();
    target.addIllegalOp<tensor::ConcatOp>();
    target.addIllegalOp<ekl::AssocOp>();
    target.addDynamicallyLegalOp<ekl::ReduceOp>(isInAssoc);
    target.addDynamicallyLegalOp<ekl::SubscriptOp>(isInAssoc);
    target.addIllegalOp<ekl::WriteOp>();

    target.addLegalDialect<ekl::EKLDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void messner::populateConvertEKLToLinalgPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertLiteral,
        ConvertUnify,
        ConvertZip,
        ConvertChoice,
        ConvertStack,
        ConvertAssoc,
        ConvertReduce,
        ConvertSubscript,
        ConvertWrite>(typeConverter, patterns.getContext());

    tensor::populateDecomposeTensorConcatPatterns(patterns);
}

std::unique_ptr<Pass> messner::createConvertEKLToLinalgPass()
{
    return std::make_unique<ConvertEKLToLinalgPass>();
}
