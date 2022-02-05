/** Implements the CFDlang import translation.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/Import.h"

#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Target/CFDlang/CLI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"

using namespace mlir;
using namespace mlir::cfdlang;

static OwningOpRef<cfdlang::ProgramOp> importProgram(
    llvm::SourceMgr &source,
    MLIRContext *context
)
{
    // TODO: Implement.
    return {};
}

static OwningModuleRef importModule(
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
        return OwningModuleRef(module);
    }

    return {};
}

void mlir::cfdlang::registerImport()
{
    TranslateToMLIRRegistration registration(
        "import-cfdlang",
        [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
            return ::importModule(sourceMgr, context);
        }
    );
}
