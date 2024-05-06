/// Main entry point for the messner MLIR language server.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

static int asMainReturnCode(LogicalResult r)
{
    return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char *argv[])
{
    DialectRegistry registry;

#ifndef NDEBUG
    registerAllDialects(registry);
#endif
    // TODO: Register dialects.
    registry.insert<ekl::EKLDialect>();

    return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
