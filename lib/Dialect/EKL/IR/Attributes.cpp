/// Implements the EKL dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Attributes.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"

#include <bit>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

static ParseResult parseSourceRange(
    AsmParser &parser,
    unsigned &startLine,
    unsigned &startColumn,
    unsigned &endLine,
    unsigned &endColumn)
{
    if (parser.parseInteger(startLine) || parser.parseColon()
        || parser.parseInteger(startColumn))
        return failure();

    endLine   = startLine;
    endColumn = endLine;
    if (parser.parseOptionalColon()) return success();

    if (parser.parseInteger(endColumn)) return failure();
    if (parser.parseOptionalColon()) return success();

    endLine = endColumn;
    return parser.parseInteger(endColumn);
}

static void printSourceRange(
    AsmPrinter &printer,
    unsigned startLine,
    unsigned startColumn,
    unsigned endLine,
    unsigned endColumn)
{
    printer << startLine << ":" << startColumn;
    if (endLine == startLine) {
        if (endColumn == startColumn) return;
        printer << ":" << endColumn;
        return;
    }
    printer << ":" << endLine << ":" << endColumn;
}

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "messner/Dialect/EKL/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayAttr implementation
//===----------------------------------------------------------------------===//

LogicalResult ekl::ArrayAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayType arrayType,
    mlir::ArrayAttr flattened)
{
    if (!arrayType) return emitError() << "expected array type";
    if (!flattened) return emitError() << "expected array attribute";

    const auto numElements = arrayType.getNumElements();
    assert(succeeded(numElements));
    if (flattened.size() != 1 && flattened.size() != *numElements)
        return emitError() << "expected " << *numElements
                           << " elements, but got " << flattened.size();

    for (auto [idx, value] : llvm::enumerate(flattened.getValue())) {
        const auto scalarAttr = llvm::dyn_cast<ScalarAttr>(value);
        if (!scalarAttr)
            return emitError() << "element #" << idx << " (" << value
                               << ") is not an EKL scalar value";

        if (!isSubtype(scalarAttr.getType(), arrayType.getScalarType()))
            return emitError()
                << "type " << scalarAttr.getType() << " of element #" << idx
                << " is not a subtype of " << arrayType.getScalarType();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// InitializerAttr implementation
//===----------------------------------------------------------------------===//

LogicalResult ekl::InitializerAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseArrayAttr flattened,
    ExtentRange extents)
{
    const auto scalarTy =
        llvm::dyn_cast<ScalarType>(flattened.getElementType());
    if (!scalarTy) {
        return emitError()
            << flattened.getElementType() << " is not an EKL scalar type";
    }

    if (failed(ArrayType::verify(emitError, scalarTy, extents)))
        return failure();

    const auto numElements = flatten(extents);
    assert(succeeded(numElements));
    if (static_cast<std::size_t>(flattened.getSize()) != *numElements) {
        return emitError() << "expected " << *numElements
                           << " elements, but got " << flattened.getSize();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// SourceLocationAttr implementation
//===----------------------------------------------------------------------===//

SourceLocationAttr SourceLocationAttr::fromLocation(OpaqueLoc loc)
{
    if (!loc || loc.getUnderlyingTypeID() != getTypeID()) return {};
    return SourceLocationAttr(
        std::bit_cast<const ImplType *>(loc.getUnderlyingLocation()));
}

OpaqueLoc SourceLocationAttr::toLocation() const
{
    return OpaqueLoc::get(
        std::bit_cast<std::uintptr_t>(getImpl()),
        getTypeID(),
        toStartLocation());
}

//===----------------------------------------------------------------------===//
// EKLDialect
//===----------------------------------------------------------------------===//

Attribute EKLDialect::parseAttribute(DialectAsmParser &parser, Type type) const
{
    if (!parser.parseOptionalColon())
        return IdentityAttr::get(parser.getContext());
    if (!parser.parseOptionalStar())
        return ExtentAttr::get(parser.getContext());
    if (!parser.parseOptionalEllipsis())
        return EllipsisAttr::get(parser.getContext());
    if (!parser.parseOptionalQuestion())
        return ErrorAttr::get(parser.getContext());

    StringRef keyword;
    Attribute result;
    const auto maybe = generatedAttributeParser(parser, &keyword, type, result);
    if (maybe.has_value()) {
        if (maybe.value()) return nullptr;
        return result;
    }

    if (keyword.consume_front("_")) {
        extent_t value;
        if (keyword.consumeInteger(10, value) || !keyword.empty()) {
            parser.emitError(parser.getNameLoc(), "expected index literal");
            return nullptr;
        }

        return IndexAttr::get(parser.getContext(), value);
    }

    parser.emitError(parser.getNameLoc(), "unknown attrribute type: ")
        << keyword;
    return nullptr;
}

void EKLDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const
{
    if (llvm::isa<IdentityAttr>(attr)) {
        os << ":";
        return;
    }
    if (llvm::isa<ExtentAttr>(attr)) {
        os << "*";
        return;
    }
    if (llvm::isa<EllipsisAttr>(attr)) {
        os << "...";
        return;
    }
    if (llvm::isa<ErrorAttr>(attr)) {
        os << "?";
        return;
    }
    if (const auto indexAttr = llvm::dyn_cast<IndexAttr>(attr)) {
        os << "_" << indexAttr.getValue();
        return;
    }

    const auto ok = generatedAttributePrinter(attr, os);
    assert(succeeded(ok));
}

void EKLDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "messner/Dialect/EKL/IR/Attributes.cpp.inc"
        >();
}
