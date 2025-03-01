/// Implements the ConvertEKLToArithPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Conversion/EKLToArith/EKLToArith.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::ekl;

//===- Generated includes -------------------------------------------------===//

namespace messner {

#define GEN_PASS_DEF_CONVERTEKLTOARITH
#include "messner/Conversion/Passes.h.inc"

} // namespace messner

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

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    messner::populateConvertEKLToArithPatterns(converter, patterns);

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
    // TODO: Implement.
}

std::unique_ptr<Pass> messner::createConvertEKLToArithPass()
{
    return std::make_unique<ConvertEKLToArithPass>();
}
