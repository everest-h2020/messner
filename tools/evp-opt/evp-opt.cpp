/** Main entry point for the evp-opt optimizer driver.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/CFDlang/IR/Base.h"
#include "mlir/Dialect/TeIL/IR/Base.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

using namespace mlir;

int main(int argc, char *argv[])
{
    registerAllPasses();

    DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<cfdlang::CFDlangDialect>();
    registry.insert<teil::TeILDialect>();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "evp-tools optimizer driver\n", registry)
    );
}
