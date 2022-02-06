#pragma once

#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir::cfdlang {

std::unique_ptr<OperationPass<ModuleOp>> createContractionFactorizationPass();

} // namespace mlir::cfdlang

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/CFDlang/Passes.h.inc"

//===----------------------------------------------------------------------===//
