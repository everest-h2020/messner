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

    // The element type must be a subtype of the declared type.
    const auto verifyType = [&](Type type) -> LogicalResult {
        if (!isSubtype(type, arrayType.getScalarType())) {
            auto diag = emitError() << "type mismatch";
            diag.attachNote()
                << type << " is not a subtype of " << arrayType.getScalarType();
            return diag;
        }

        return success();
    };

    if (const auto splat = llvm::dyn_cast<ScalarAttr>(*stack.begin())) {
        // Correctly typed splats are always valid.
        if (failed(verifyType(splat.getType()))) return failure();
        return success();
    }
    if (arrayType.isScalar()) {
        // Scalars must always be stored as a splat.
        return emitError() << "expected splat";
    }

    // The elements in the stack must match the remaining extents.
    const auto subExtents   = arrayType.getExtents().drop_front();
    // Match them under the assumption that they are implicitly broadcasted.
    const auto matchExtents = [&](ExtentRange extents) -> LogicalResult {
        if (extents.empty()) return success();
        if (extents.size() != subExtents.size()) return failure();
        for (auto [have, want] : llvm::zip_equal(extents, subExtents))
            if (have != want && have != 1) return failure();
        return success();
    };
    // A stack element must match have matching type and extents.
    const auto verifyElement = [&](Attribute attr) -> LogicalResult {
        // Accept scalars, because they broadcast to anything.
        if (const auto scalarAttr = llvm::dyn_cast<ScalarAttr>(attr)) {
            if (failed(verifyType(scalarAttr.getType()))) return failure();
            return success();
        }

        // Otherwise, it must be a matching ArrayAttr.
        const auto arrayAttr = llvm::dyn_cast<ekl::ArrayAttr>(attr);
        if (!arrayAttr) return emitError() << "expected array attribute";
        if (failed(verifyType(arrayAttr.getType().getScalarType())))
            return failure();
        if (failed(matchExtents(arrayAttr.getType().getExtents()))) {
            auto diag  = emitError() << "extent mismatch";
            auto &note = diag.attachNote() << "[";
            llvm::interleaveComma(arrayAttr.getType().getExtents(), note);
            note << "] is not broadcastable to [";
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

[[nodiscard]] static Attribute subscript(
    mlir::ArrayAttr root,
    ScalarType type,
    ExtentRange extents,
    ExtentRange indices)
{
    ScalarAttr splatValue{};
    while (!indices.empty()) {
        // Step down into the next extent.
        const auto index = indices.front();
        extents          = extents.drop_front();
        indices          = indices.drop_front();

        const auto stack = root.getValue();
        assert(!stack.empty());

        if (stack.size() > 1) {
            // The current extent is not broadcasted.
            if ((splatValue = llvm::dyn_cast<ScalarAttr>(stack[index]))) {
                // Don't descend into a splat, return it.
                break;
            }

            // Otherwise, the extent must be a child array, descend into it.
            root = llvm::cast<ekl::ArrayAttr>(stack[index]).getStack();
            continue;
        }

        if (const auto bcast = llvm::dyn_cast<ekl::ArrayAttr>(stack.front())) {
            // The extent is stored as a broadcasted array, descend into it.
            root = bcast.getStack();
            continue;
        }

        // Don't descend into a splat, return it.
        splatValue = llvm::cast<ScalarAttr>(stack.front());
        break;
    }

    if (splatValue) {
        // The loop was exited early due to a splat being found. Simulate the
        // remaining descent by dropping the unaddressed extents.
        extents = extents.drop_front(indices.size());

        // Decay to a scalar if necessary.
        if (extents.empty()) return splatValue;

        // If a splat value was found, a splat array must be created so that the
        // remaining extents are correctly declared.
        return ekl::ArrayAttr::get(splatValue, extents);
    }

    // Automatically decay to a scalar if needed.
    if (extents.empty()) return *root.begin();
    return ekl::ArrayAttr::get(ArrayType::get(type, extents), root);
}

Attribute ekl::ArrayAttr::subscript(ExtentRange indices) const
{
    assert(getType().isInBounds(indices));

    return ::subscript(
        getStack(),
        getType().getScalarType(),
        getType().getExtents(),
        indices);
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
