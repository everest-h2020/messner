/// Main entry point for the evp-opt optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

int main(int argc, char *argv[])
{
    registerAllPasses();

    DialectRegistry registry;

    // TODO: Register dialects.
    // registerAllDialects(registry);

    return asMainReturnCode(
        MlirOptMain(argc, argv, "messner optimizer driver\n", registry));
}
