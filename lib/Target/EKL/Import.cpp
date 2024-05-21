/// Implements the main entry point for importing EKL sources into MLIR.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Target/EKL/Import.h"
#include "messner/Dialect/EKL/Transforms/TypeCheck.h"

#include "Lexer.hpp"
#include "ParseDriver.h"
#include "Parser.hpp"
#include "messner/Dialect/EKL/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Importer implementation
//===----------------------------------------------------------------------===//

[[nodiscard]] static OwningOpRef<ProgramOp>
import(MLIRContext *context, const std::shared_ptr<llvm::SourceMgr> &sourceMgr)
{
    // Create the driver, lexer and parser.
    ParseDriver driver(context, sourceMgr);
    ekl::detail::LexerImpl lexer(
        driver,
        reflex::Input(driver.getSource().data(), driver.getSource().size()));
    ekl::detail::ParserImpl parser(driver, lexer);

    // Parse the input.
    parser.parse();

    // Take the result, which may be empty if parsing failed.
    return driver.takeResult();
}

OwningOpRef<ProgramOp> mlir::ekl::importAndTypeCheck(
    MLIRContext *context,
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr)
{
    auto result = import(context, sourceMgr);
    if (!result) return {};

    if (failed(verify(*result)) || failed(typeCheck(*result))) {
        result.release();
        return {};
    }

    return result;
}

void mlir::ekl::registerImport()
{
    TranslateToMLIRRegistration registration(
        "import-ekl",
        "Converts EKL source into the MLIR ekl dialect",
        [](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
           MLIRContext *context) -> OwningOpRef<ProgramOp> {
            context->loadAllAvailableDialects();

            // Install a rich diagnostic handler for the parsing process.
            DiagHandler diagHandler(context, sourceMgr);

            return import(context, sourceMgr);
        },
        [](DialectRegistry &registry) { registry.insert<EKLDialect>(); });
}
