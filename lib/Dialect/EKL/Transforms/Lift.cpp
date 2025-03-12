/// Implements the LiftPass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ImplicitCast.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Traits.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "messner/Dialect/EKL/Transforms/Passes.h"
#include "messner/Dialect/EKL/Transforms/TypeCheck.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::ekl;

#define DEBUG_TYPE "ekl-lift"

//===- Generated includes -------------------------------------------------===//

namespace mlir::ekl {

#define GEN_PASS_DEF_LIFT
#include "messner/Dialect/EKL/Transforms/Passes.h.inc"

} // namespace mlir::ekl

//===----------------------------------------------------------------------===//

namespace {

struct LiftPass : ekl::impl::LiftBase<LiftPass> {
    using LiftBase::LiftBase;

    void runOnOperation() override final;
};

} // namespace

//===----------------------------------------------------------------------===//
// populateLiftPatterns implementation
//===----------------------------------------------------------------------===//

namespace {} // namespace

void mlir::ekl::populateLiftPatterns(RewritePatternSet &patterns) {}

//===----------------------------------------------------------------------===//
// LiftPass implementation
//===----------------------------------------------------------------------===//

void LiftPass::runOnOperation()
{
    RewritePatternSet patterns(&getContext());

    populateDecayNumberPatterns(patterns);

    if (failed(applyPatternsGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::ekl::createLiftPass()
{
    return std::make_unique<LiftPass>();
}
