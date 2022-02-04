/** Implements the TeIL dialect types.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/TeIL/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::teil;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/TeIL/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TeILDialect
//===----------------------------------------------------------------------===//

void TeILDialect::registerTypes()
{
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "mlir/Dialect/TeIL/IR/Types.cpp.inc"
    >();
}

void TeILDialect::printType(Type type, DialectAsmPrinter& printer) const
{
    if (failed(generatedTypePrinter(type, printer))) {
        llvm_unreachable("unexpected 'teil' type kind");
    }
}

Type TeILDialect::parseType(DialectAsmParser& parser) const
{
    StringRef typeTag;
    if (parser.parseKeyword(&typeTag)) {
        return Type();
    }

    Type genType;
    auto parseResult = generatedTypeParser(
        parser.getBuilder().getContext(),
        parser,
        typeTag,
        genType
    );
    if (parseResult.hasValue()) {
        return genType;
    }

    parser.emitError(parser.getNameLoc(), "unknown teil type: ") << typeTag;
    return Type();
}
