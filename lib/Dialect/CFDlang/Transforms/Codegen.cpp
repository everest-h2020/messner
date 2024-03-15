#include "PassDetail.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/CFDlang/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::cfdlang;

#define DEBUG_TYPE "cfdlang-codegen"

namespace {

class MulLowering : public OpConversionPattern<MulOp> {
public:
    using OpConversionPattern<MulOp>::OpConversionPattern;

    virtual LogicalResult match(MulOp op) const override { return success(); }

    virtual void rewrite(
        MulOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter
    ) const override final {
        auto out_shape = op.getResult().cast<Atom>().getShape();
        auto out = rewriter.create<tensor::EmptyOp>(
            op.getLoc(),
            out_shape,
            rewriter.getF64Type()
        );

        SmallVector<Value, 2> in;
        in.push_back(adaptor.getLhs());
        in.push_back(adaptor.getRhs());

        SmallVector<AffineMap, 3> maps;
        maps.push_back(AffineMap::getMultiDimIdentityMap(
            out_shape.size(),
            rewriter.getContext()
        ));
        maps.push_back(AffineMap::getMultiDimIdentityMap(
            out_shape.size(),
            rewriter.getContext()
        ));
        maps.push_back(AffineMap::getMultiDimIdentityMap(
            out_shape.size(),
            rewriter.getContext()
        ));

        SmallVector<utils::IteratorType, 4> iterators(out_shape.size(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            (Type)RankedTensorType::get(out_shape, rewriter.getF64Type()),
            in,
            (Value)out,
            maps,
            iterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value res =
                    builder.create<arith::MulFOp>(loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, res);
            }
        );
    }
};

class ProductLowering : public OpConversionPattern<ProductOp> {
public:
    using OpConversionPattern<ProductOp>::OpConversionPattern;

    virtual LogicalResult match(ProductOp op) const override {
        return success();
    }

    virtual void rewrite(
        ProductOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter
    ) const override final {
        auto out_shape = op.getResult().cast<Atom>().getShape();
        auto out = rewriter.create<tensor::EmptyOp>(
            op.getLoc(),
            out_shape,
            rewriter.getF64Type()
        );

        auto lhs_rank = adaptor.getLhs().getType().cast<ShapedType>().getRank();
        auto rhs_rank = adaptor.getRhs().getType().cast<ShapedType>().getRank();

        SmallVector<Value, 2> in;
        in.push_back(adaptor.getLhs());
        in.push_back(adaptor.getRhs());

        SmallVector<AffineMap, 3> maps;
        SmallVector<AffineExpr, 4> lhsMap;
        for (unsigned i = 0; i < lhs_rank; ++i) {
            lhsMap.push_back(getAffineDimExpr(i, rewriter.getContext()));
        }
        maps.push_back(
            AffineMap::get(out_shape.size(), 0, lhsMap, rewriter.getContext())
        );
        maps.push_back(AffineMap::getMinorIdentityMap(
            out_shape.size(),
            rhs_rank,
            rewriter.getContext()
        ));
        maps.push_back(AffineMap::getMultiDimIdentityMap(
            out_shape.size(),
            rewriter.getContext()
        ));

        SmallVector<utils::IteratorType, 4> iterators(out_shape.size(), utils::IteratorType::parallel);

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            (Type)RankedTensorType::get(out_shape, rewriter.getF64Type()),
            in,
            (Value)out,
            maps,
            iterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value res =
                    builder.create<arith::MulFOp>(loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, res);
            }
        );
    }
};

class ContractLowering : public OpConversionPattern<ContractOp> {
public:
    using OpConversionPattern<ContractOp>::OpConversionPattern;

    virtual LogicalResult match(ContractOp op) const override {
        return success();
    }

    virtual void rewrite(
        ContractOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter
    ) const override final {
        auto out_shape = op.getResult().cast<Atom>().getShape();
        auto init = rewriter.create<tensor::EmptyOp>(
            op.getLoc(),
            out_shape,
            rewriter.getF64Type()
        );
        auto zero = rewriter.create<arith::ConstantFloatOp>(
            op.getLoc(),
            APFloat::getZero(APFloat::IEEEdouble()),
            rewriter.getF64Type()
        );
        auto out =
            rewriter.create<linalg::FillOp>(op.getLoc(), zero.getResult(), init.getResult()).result();

        auto indices = op.getIndicesAttr().getValues();
        auto in_shape =
            adaptor.getOperand().getType().cast<ShapedType>().getShape();

        SmallVector<unsigned, 4> outDims;
        for (unsigned dim = 0; dim < in_shape.size(); ++dim) {
            if (std::find(indices.begin(), indices.end(), dim + 1) ==
                indices.end()) {
                outDims.push_back(dim);
            }
        }

        SmallVector<AffineExpr, 4> ins, outs;
        ins.resize(in_shape.size());
        SmallVector<utils::IteratorType, 4> iterators;
        unsigned dim = 0;
        for (; dim < out_shape.size(); ++dim) {
            iterators.push_back(utils::IteratorType::parallel);
            auto here = getAffineDimExpr(dim, rewriter.getContext());
            outs.push_back(here);
            ins[outDims[dim]] = here;
        }

        for (auto it = indices.begin(); it != indices.end(); ++it, ++dim) {
            iterators.push_back(utils::IteratorType::reduction);
            auto here = getAffineDimExpr(dim, rewriter.getContext());
            ins[*it - 1] = here;
            ++it;
            ins[*it - 1] = here;
        }

        SmallVector<AffineMap, 2> maps;
        maps.push_back(AffineMap::get(
            out_shape.size() + indices.size() / 2,
            /*symbols=*/0,
            ins,
            rewriter.getContext()
        ));
        maps.push_back(AffineMap::get(
            out_shape.size() + indices.size() / 2,
            0,
            outs,
            rewriter.getContext()
        ));

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op,
            (Type)RankedTensorType::get(out_shape, rewriter.getF64Type()),
            (Value)adaptor.getOperand(),
            out,
            maps,
            iterators,
            [&](OpBuilder& builder, Location loc, ValueRange args) {
                Value add =
                    builder.create<arith::AddFOp>(loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, add);
            }
        );
    }
};

class CastRemoval : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    virtual LogicalResult matchAndRewrite(
        UnrealizedConversionCastOp op,
        PatternRewriter& rewriter
    ) const override {
        auto par1 = op.getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
        if (!par1) return failure();
        auto par2 =
            par1.getOperand().getDefiningOp<UnrealizedConversionCastOp>();
        if (!par2) return failure();

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
            op,
            par2.getOperand(0)
        );

        return success();
    }
};

class AllocaCastRemoval : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    virtual LogicalResult matchAndRewrite(
        UnrealizedConversionCastOp op,
        PatternRewriter& rewriter
    ) const override {
        auto alloca = op.getOperand(0).getDefiningOp<memref::AllocaOp>();
        if (!alloca) return failure();

        rewriter.replaceOpWithNewOp<memref::AllocaOp>(
            op,
            op.getResult(0).getType().cast<MemRefType>()
        );

        return success();
    }
};

class CopyCastRemoval : public OpRewritePattern<memref::CopyOp> {
public:
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    virtual LogicalResult matchAndRewrite(
        memref::CopyOp op,
        PatternRewriter& rewriter
    ) const override {
        auto tar = op.getTarget().getDefiningOp<UnrealizedConversionCastOp>();
        if (!tar) return failure();
        auto src = op.getSource().getDefiningOp<bufferization::ToMemrefOp>();
        if (!src) return failure();
        auto src2 =
            src.getOperand().getDefiningOp<UnrealizedConversionCastOp>();
        if (!src2) return failure();

        rewriter.replaceOpWithNewOp<memref::CopyOp>(
            op,
            rewriter.create<bufferization::ToMemrefOp>(
                src2.getLoc(),
                MemRefType::get(
                    op.getOperand(0).getType().cast<MemRefType>().getShape(),
                    rewriter.getF64Type()
                ),
                src2.getOperand(0)
            ),
            tar.getOperand(0)
        );

        return success();
    }
};

class CodegenPass : public CodegenBase<CodegenPass> {
public:
    virtual void runOnOperation() override {
        TypeConverter typeConverter;

        typeConverter.addConversion([](Type type) -> std::optional<Type> {
            return type;
        });
        typeConverter.addConversion([](ScalarType type) -> std::optional<Type> {
            return Float64Type::get(type.getContext());
        });
        typeConverter.addConversion(
            [&](RankedTensorType type) -> std::optional<Type> {
                return RankedTensorType::get(
                    type.getShape(),
                    typeConverter.convertType(type.getElementType())
                );
            }
        );
        typeConverter.addConversion([&](MemRefType type) -> std::optional<Type> {
            return MemRefType::get(
                type.getShape(),
                typeConverter.convertType(type.getElementType())
            );
        });

        auto addUnrealizedCast = [](OpBuilder& builder,
                                    Type type,
                                    ValueRange inputs,
                                    Location loc) {
            auto cast =
                builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
            return std::optional<Value>(cast.getResult(0));
        };
        typeConverter.addSourceMaterialization(addUnrealizedCast);
        typeConverter.addTargetMaterialization(addUnrealizedCast);

        RewritePatternSet patterns(&getContext());
        ConversionTarget target(getContext());

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns,
            typeConverter
        );
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });

        populateCallOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
            return typeConverter.isLegal(op);
        });

        populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return typeConverter.isLegal(op);
        });

        populateReconcileUnrealizedCastsPatterns(patterns);

        patterns.add<MulLowering, ProductLowering, ContractLowering>(
            typeConverter,
            &getContext()
        );

        target.addLegalDialect<
            memref::MemRefDialect,
            arith::ArithDialect,
            bufferization::BufferizationDialect,
            linalg::LinalgDialect>();
        target.addLegalOp<ModuleOp>();
        if (failed(
                applyFullConversion(getOperation(), target, std::move(patterns))
            ))
            signalPassFailure();

        RewritePatternSet patt2(&getContext());
        patt2.add<CastRemoval>(&getContext());
        patt2.add<CopyCastRemoval>(&getContext());
        patt2.add<AllocaCastRemoval>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patt2))))
            signalPassFailure();
    }

    virtual void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<linalg::LinalgDialect>();
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::cfdlang::createCodegenPass() {
    return std::make_unique<CodegenPass>();
}