/// Main entry point for the messner translation driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Target/EKL/Import.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv)
{
    // TODO: Register translations.
#ifndef NDEBUG
    registerAllTranslations();
#endif
    ekl::registerImport();

    return failed(mlirTranslateMain(argc, argv, "messner translation driver"));
}
