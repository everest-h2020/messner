/// Main entry point for the messner translation driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv)
{
    // TODO: Register translations.
    // registerAllTranslations();

    return failed(mlirTranslateMain(argc, argv, "messner translation driver"));
}
