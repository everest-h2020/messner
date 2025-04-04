/// Main entry point for the messner optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Tools/InitRegistry.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

using namespace mlir;

int main(int argc, char *argv[])
{
    DialectRegistry registry;

    registerAllDialects(registry);
    messner::registerAllDialects(registry);
    registerAllExtensions(registry);
    messner::registerAllExtensions(registry);
    registerAllPasses();
    messner::registerAllPasses(registry);

    return asMainReturnCode(
        MlirOptMain(argc, argv, "messner optimizer driver\n", registry));
}
