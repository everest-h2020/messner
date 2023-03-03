/** Implements the CFDlang import translation.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/Import.h"

#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Target/CFDlang/Utils/ParseDriver.h"
#include "mlir/Target/CFDlang/CLI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "Tokenizer.h"

using namespace mlir;
using namespace mlir::cfdlang;

static OwningOpRef<cfdlang::ProgramOp> importProgram(
    llvm::SourceMgr &source,
    MLIRContext *context
)
{
    // Invoke the parser and convert to nullptr-style success indicator.
    ImportContext importContext(context, source);
    cfdlang::detail::ParseDriver driver(importContext);
    if (failed(driver.parse())) {
        return {};
    }

    return driver.takeResult();
}

static OwningOpRef<ModuleOp> importModule(
    llvm::SourceMgr &source,
    MLIRContext *context
)
{
    // Import as a ProgramOp.
    if (auto result = importProgram(source, context)) {
        // Wrap the result in a ModuleOp.
        OpBuilder builder(context);
        auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
        builder.setInsertionPointToStart(module.getBody());
        builder.insert(result.release());
        return OwningOpRef<ModuleOp>(module);
    }

    return {};
}

void mlir::cfdlang::registerImport()
{
    TranslateToMLIRRegistration registration(
        "import-cfdlang",
        [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
            // Load dialects.
            context->loadDialect<CFDlangDialect>();
            // Call importer.
            return ::importModule(sourceMgr, context);
        }
    );
}
