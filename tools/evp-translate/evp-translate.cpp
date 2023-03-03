/** Main entry point for the evp-translate translation driver.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/CFDlang/Export.h"
#include "mlir/Target/CFDlang/Import.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv)
{
    registerAllTranslations();

    cfdlang::registerExport();
    cfdlang::registerImport();

    return failed(
        mlirTranslateMain(argc, argv, "evp-tools translation driver")
    );
}
