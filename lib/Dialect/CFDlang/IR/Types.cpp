/** Implementation of the CFDlang dialect ops.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::cfdlang;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CFDlang/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CFDlangDialect
//===----------------------------------------------------------------------===//

void CFDlangDialect::registerTypes()
{
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "mlir/Dialect/CFDlang/IR/Types.cpp.inc"
    >();
}

/*
void CFDlangDialect::printType(Type type, DialectAsmPrinter &printer) const
{
    if (failed(generatedTypePrinter(type, printer))) {
        llvm_unreachable("unexpected 'cfdlang' type kind");
    }
}

Type CFDlangDialect::parseType(DialectAsmParser &parser) const
{
    StringRef typeTag;
    if (parser.parseKeyword(&typeTag)) {
        return nullptr;
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

    parser.emitError(parser.getNameLoc(), "unknown cfdlang type: ") << typeTag;
    return nullptr;
}
*/
