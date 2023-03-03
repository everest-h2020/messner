#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::cfdlang {

std::unique_ptr<OperationPass<ModuleOp>> createContractionFactorizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createKernelizePass();
std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass();

} // namespace mlir::cfdlang

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/CFDlang/Passes.h.inc"

//===----------------------------------------------------------------------===//
