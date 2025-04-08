/// Implementation of the Ref dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Types.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/Ref/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RefDialect implementation
//===----------------------------------------------------------------------===//

auto RefDialect::parseType(DialectAsmParser &parser) const -> Type
{
    if (std::string kindStr; !parser.parseOptionalString(&kindStr)) {
        const auto maybeKind = symbolizeReferenceKind(kindStr);
        if (!maybeKind) {
            parser.emitError(parser.getNameLoc(), "invalid reference kind '")
                << kindStr << "'";
            return nullptr;
        }

        Type cellType;
        if (parser.parseComma() || parser.parseType(cellType)) return nullptr;
        return ReferenceType::get(parser.getContext(), cellType, *maybeKind);
    }

    StringRef keyword;
    Type result;
    if (const auto maybeError = generatedTypeParser(parser, &keyword, result);
        maybeError.has_value()) {
        if (maybeError.value()) return nullptr;
        return result;
    }

    if (const auto maybeKind = symbolizeReferenceKind(keyword); maybeKind) {
        Type cellType;
        if (parser.parseComma() || parser.parseType(cellType)) return nullptr;
        return ReferenceType::get(parser.getContext(), cellType, *maybeKind);
    }

    parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
    return nullptr;
}

void RefDialect::printType(Type type, DialectAsmPrinter &os) const
{
    if (const auto refTy = llvm::dyn_cast<ReferenceType>(type); refTy) {
        os << refTy.getKind() << ", " << refTy.getCellType();
        return;
    }

    const auto ok = generatedTypePrinter(type, os);
    assert(succeeded(ok));
}

void RefDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "messner/Dialect/Ref/IR/Types.cpp.inc"
        >();
}
