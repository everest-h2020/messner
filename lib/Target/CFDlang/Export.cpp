/** Implements the CFDlang export translation.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/Export.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Target/CFDlang/Utils/PrintDriver.h"
#include "mlir/Target/CFDlang/CLI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::cfdlang;

static LogicalResult exportProgram(ProgramOp program, raw_ostream &output)
{
    cfdlang::detail::PrintDriver driver(output);
    driver.print(program);
    return success();
}

static LogicalResult exportModule(ModuleOp module, raw_ostream &output)
{
    auto programs = module.getBody()->getOps<ProgramOp>();

    // Find the first program that we should export.
    auto filter = [](ProgramOp program) {
        return translated_program_name.empty()
            || program.getName().value_or("").equals(translated_program_name);
    };
    auto first = std::find_if(programs.begin(), programs.end(), filter);

    if (first == programs.end()) {
        // No programs to export.
        llvm::WithColor::error()
            << "export-cfdlang: no program matching '"
            << translated_program_name
            << "' found!\n";
        return failure();
    }

    // Export the first program.
    if (failed(exportProgram(*first, output))) {
        return failure();
    }

    // Find another program to export.
    auto second = std::find_if(std::next(first), programs.end(), filter);
    if (second != programs.end()) {
        // Warn user but do not fail.
        llvm::WithColor::warning()
            << "export-cfdlang: multiple programs matching '"
            << translated_program_name
            << "' found!\n";
        (*first).emitRemark()
            << "this program was exported.\n";
        (*second).emitWarning()
            << "this and following matches were not exported.\n";
    }

    return success();
}

void mlir::cfdlang::registerExport()
{
    TranslateFromMLIRRegistration registration(
        "export-cfdlang",
        "Converts the MLIR CFDlang dialect into CFDlang source",
        [](ModuleOp module, raw_ostream &output) {
            return exportModule(module, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<cfdlang::CFDlangDialect>();
        }
    );
}
