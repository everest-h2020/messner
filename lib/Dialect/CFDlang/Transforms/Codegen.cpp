#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Dialect/CFDlang/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::cfdlang;

#define DEBUG_TYPE "cfdlang-codegen"

namespace {

class CodegenPass
        : public CodegenBase<CodegenPass> {
public:
    virtual void runOnOperation() override
    {

    }
};

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::cfdlang::createCodegenPass() {
    return std::make_unique<CodegenPass>();
}