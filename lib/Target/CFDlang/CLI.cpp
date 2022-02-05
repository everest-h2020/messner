/** Implements common CFDlang translation CLI options.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/CLI.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::cfdlang;

std::string cfdlang::translated_program_name;

// BUG: While registration of the translation can be requested explicitly, these
//      CLI options will always be added to all linked apps. A solution for more
//      consistency would be to auto-register the translations too, since that
//      makes sense anyway for our usage.

static llvm::cl::opt<std::string, true> programName(
    "cfdlang-program-name",
    llvm::cl::desc("Specifies a CFDlang program name for import/export"),
    llvm::cl::location(cfdlang::translated_program_name)
);
