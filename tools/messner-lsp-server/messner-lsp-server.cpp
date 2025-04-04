/// Main entry point for the messner MLIR language server.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Tools/InitRegistry.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>

using namespace mlir;

[[nodiscard]] static int asMainReturnCode(LogicalResult result)
{
    return llvm::succeeded(result) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char *argv[])
{
    DialectRegistry registry;

    registerAllDialects(registry);
    messner::registerAllDialects(registry);

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
