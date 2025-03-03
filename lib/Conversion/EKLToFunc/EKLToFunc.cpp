/// Implements the ConvertEKLToFuncPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToFunc/EKLToFunc.h"

#include "messner/Dialect/EKL/Analysis/Extents.h"
#include "messner/Dialect/EKL/IR/Base.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOFUNC
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

//===----------------------------------------------------------------------===//

[[nodiscard]]
static ArrayRef<int64_t> toShape(ExtentRange extents)
{
    return {reinterpret_cast<const int64_t *>(extents.data()), extents.size()};
}

namespace {

struct ConvertEKLToFuncPass
        : messner::impl::ConvertEKLToFuncBase<ConvertEKLToFuncPass> {
    using ConvertEKLToFuncBase::ConvertEKLToFuncBase;

    void runOnOperation() override;
};

struct ConvertKernel : OpConversionPattern<ekl::KernelOp> {
    using OpConversionPattern<ekl::KernelOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ekl::KernelOp op,
        ekl::KernelOp::Adaptor,
        ConversionPatternRewriter &rewriter) const final
    {
        auto module = op->getParentOfType<ModuleOp>();

        const auto argTys  = llvm::to_vector(llvm::map_range(
            op.getBody()->getArgumentTypes(),
            [&](Type inTy) { return convertArgType(inTy); }));
        const auto argLocs = llvm::to_vector(llvm::map_range(
            op.getBody()->getArguments(),
            [&](BlockArgument inArg) { return inArg.getLoc(); }));

        rewriter.setInsertionPointToEnd(module.getBody());
        auto func = rewriter.create<func::FuncOp>(
            op.getLoc(),
            op.getSymName(),
            rewriter.getFunctionType(argTys, {}));
        const auto args =
            func.getBody().emplaceBlock().addArguments(argTys, argLocs);
        rewriter.setInsertionPointToStart(&func.getBody().front());

        SmallVector<Value> params(argTys.size());
        for (auto &&[i, arg] : llvm::enumerate(args))
            params[i] = convertArg(
                rewriter,
                argLocs[i],
                op.getBody()->getArgument(i).getType(),
                arg);

        auto retOp = rewriter.create<func::ReturnOp>(op.getLoc());
        rewriter.inlineBlockBefore(op.getBody(), retOp, params);

        rewriter.eraseOp(op);
        return success();
    }

private:
    [[nodiscard]] Type convertArgType(Type argTy) const
    {
        argTy = getTypeBound(argTy);
        if (argTy.isSignlessIntOrIndexOrFloat()) return argTy;

        if (const auto intTy = llvm::dyn_cast<mlir::IntegerType>(argTy))
            return mlir::IntegerType::get(getContext(), intTy.getWidth());

        if (const auto idxTy = llvm::dyn_cast<ekl::IndexType>(argTy))
            return mlir::IndexType::get(getContext());

        if (const auto arrayTy = llvm::dyn_cast<ekl::ArrayType>(argTy))
            return mlir::MemRefType::get(
                toShape(arrayTy.getExtents()),
                convertArgType(arrayTy.getScalarType()));

        if (const auto refTy = llvm::dyn_cast<ekl::ReferenceType>(argTy))
            return mlir::MemRefType::get(
                toShape(refTy.getExtents()),
                convertArgType(refTy.getScalarType()));

        llvm_unreachable("invalid ABI type encountered");
    }

    [[nodiscard]] Value
    convertArg(OpBuilder &builder, Location loc, Type resultTy, Value input)
        const
    {
        resultTy = getTypeBound(resultTy);

        if (const auto arrayTy = llvm::dyn_cast<ekl::ArrayType>(resultTy)) {
            input =
                builder
                    .create<bufferization::ToTensorOp>(loc, input, true, false)
                    .getResult();
        }

        if (input.getType() != resultTy) {
            input =
                builder.create<UnrealizedConversionCastOp>(loc, resultTy, input)
                    .getResult(0);
        }

        return builder.create<ekl::IntroOp>(loc, input).getResult();
    }
};

} // namespace

void ConvertEKLToFuncPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    messner::populateConvertEKLToFuncPatterns(converter, patterns);

    target.addLegalDialect<ekl::EKLDialect>();
    target.addIllegalOp<ekl::KernelOp>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<BuiltinDialect>();

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();

    getOperation()->walk([](ekl::ProgramOp op) -> WalkResult {
        if (op.getBody()->getOps<ekl::KernelOp>().empty()) op->erase();
        return WalkResult::skip();
    });
}

void messner::populateConvertEKLToFuncPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertKernel>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> messner::createConvertEKLToFuncPass()
{
    return std::make_unique<ConvertEKLToFuncPass>();
}
