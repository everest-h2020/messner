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
    mlir::ArrayAttr stack)
{
    if (!arrayType) return emitError() << "expected array type";
    if (!stack) return emitError() << "expected stack attribute";
    if (stack.empty()) return emitError() << "stack can't be empty";

    const auto verifyCovariant = [&](Type type) -> LogicalResult {
        if (!isSubtype(type, arrayType.getScalarType())) {
            auto diag = emitError() << "type mismatch";
            diag.attachNote()
                << type << " is not a subtype of " << arrayType.getScalarType();
            return diag;
        }

        return success();
    };

    const auto scalar = llvm::dyn_cast<ScalarAttr>(*stack.begin());
    if (scalar) {
        if (failed(verifyCovariant(scalar.getType()))) return failure();
        // Attribute is a valid splat.
        return success();
    }
    if (arrayType.isScalar()) {
        // Scalars must always be stored as a splat.
        return emitError() << "expected splat";
    }

    const auto subExtents    = arrayType.getExtents().drop_front();
    const auto verifyElement = [&](Attribute attr) -> LogicalResult {
        const auto arrayAttr = llvm::dyn_cast<ekl::ArrayAttr>(attr);
        if (!arrayAttr) return emitError() << "expected array attribute";
        if (failed(verifyCovariant(arrayAttr.getType().getScalarType())))
            return failure();
        if (arrayAttr.getType().getExtents() != subExtents) {
            auto diag  = emitError() << "extent mismatch";
            auto &note = diag.attachNote() << "[";
            llvm::interleaveComma(arrayAttr.getType().getExtents(), note);
            note << "] != [";
            llvm::interleaveComma(subExtents, note);
            note << "]";
            return diag;
        }

        return success();
    };

    if (failed(verifyElement(*stack.begin()))) return failure();
    if (stack.size() == 1) {
        // Attribute is a valid broadcast.
        return success();
    }
    if (stack.size() != arrayType.getExtent(0)) {
        auto diag = emitError() << "extent mismatch";
        diag.attachNote() << stack.size() << " != " << arrayType.getExtent(0);
        return diag;
    }

    for (auto element : stack.getValue().drop_front())
        if (failed(verifyElement(element))) return failure();

    return success();
}

[[nodiscard]] static Attribute
subscript(ekl::ArrayAttr root, ExtentRange indices)
{
    assert(root.getType().isInBounds(indices));

    while (!indices.empty()) {
        const auto stack = root.getStack().getValue();
        assert(!stack.empty());

        if (stack.size() > 1) {
            // Descend into the stacked child array.
            root    = llvm::cast<ekl::ArrayAttr>(stack[indices.front()]);
            indices = indices.drop_front();
            continue;
        }

        if (const auto bcast = llvm::dyn_cast<ekl::ArrayAttr>(stack.front())) {
            // Descend into implicitly broadcasted child array.
            root    = bcast;
            indices = indices.drop_front();
            continue;
        }

        // The result will be a splat, we just have to drop all the extents that
        // we'd still need to apply an index to.
        const auto splat = llvm::cast<ScalarAttr>(stack.front());
        const auto extents =
            root.getType().getExtents().drop_front(indices.size());

        // Automatically decay to a scalar if needed.
        if (extents.empty()) return splat;
        // Make a new splat attribute.
        return ekl::ArrayAttr::get(root.getType().cloneWith(extents), splat);
    }

    // Automatically decay to a scalar if needed.
    if (root.getType().isScalar()) return *root.getStack().begin();
    return root;
}

Attribute ekl::ArrayAttr::subscript(ExtentRange indices) const
{
    return ::subscript(*this, indices);
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
