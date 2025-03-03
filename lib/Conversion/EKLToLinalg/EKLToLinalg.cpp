/// Implements the ConvertEKLToLinalgPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToLinalg/EKLToLinalg.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::ekl;

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

} // namespace

void ConvertEKLToLinalgPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    messner::populateConvertEKLToLinalgPatterns(converter, patterns);

    target.addLegalDialect<linalg::LinalgDialect>();

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
    // TODO: Implement.
}

std::unique_ptr<Pass> messner::createConvertEKLToLinalgPass()
{
    return std::make_unique<ConvertEKLToLinalgPass>();
}
