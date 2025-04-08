/// Implementation of the EKL dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Attributes.h"

#include "messner/Dialect/EKL/Analysis/Shape.h"
#include "messner/Dialect/EKL/IR/Diagnostics.h"
#include "messner/Dialect/EKL/IR/Dialect.h"
#include "messner/Dialect/EKL/IR/TypeSystem.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "messner/Dialect/EKL/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RationalAttr implementation
//===----------------------------------------------------------------------===//

auto RationalAttr::get(MLIRContext *context, Rational value) -> RationalAttr
{
    value.shrinkToFit();
    return Base::get(context, std::move(value));
}

//===----------------------------------------------------------------------===//
// ArrayAttr implementation
//===----------------------------------------------------------------------===//

auto ekl::ArrayAttr::get(ArrayType arrayType, TupleAttr stack) -> ArrayAttr
{
    assert(arrayType);
    assert(stack);

    return Base::get(arrayType.getContext(), arrayType, stack);
}

auto ekl::ArrayAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayType arrayType,
    TupleAttr stack) -> LogicalResult
{
    if (!arrayType) return emitError() << "expected array type";
    if (!stack) return emitError() << "expected stack attribute";
    if (stack.empty()) return emitError() << "stack can't be empty";

    // Covariant type verification.
    const auto &typeSystem     = getTypeSystem(arrayType.getContext());
    const auto verifyCovariant = [&](Type type) -> LogicalResult {
        if (!typeSystem.isSubtype(type, arrayType.getScalarType())) {
            auto diag = emitError() << "type mismatch";
            diag.attachNote()
                << type << " is not a subtype of " << arrayType.getScalarType();
            return diag;
        }

        return success();
    };

    // Handle scalar case.
    if (arrayType.getNumExtents() == 0) {
        const auto splatValue = llvm::dyn_cast<ScalarAttr>(*stack.begin());
        if (!splatValue || stack.size() != 1)
            return emitError() << "expected splat";
        return verifyCovariant(splatValue.getType());
    }

    // Broadcast shape verification.
    const auto shape       = arrayType.getShape().drop_front();
    const auto verifyShape = [&](ShapeRef found) -> LogicalResult {
        const auto maybeBcast = broadcast(shape, found);
        if (failed(maybeBcast) || ShapeRef(*maybeBcast) != shape) {
            auto diag = emitError() << "shape mismatch";
            diag.attachNote() << found << " is not broadcastable to " << shape;
            return diag;
        }
        return success();
    };

    // Stack element verification.
    const auto verifyElement = [&](Attribute attr) -> LogicalResult {
        // Scalars are always broadcastable, so must only be covariant.
        if (const auto scalarAttr = llvm::dyn_cast<ScalarAttr>(attr);
            scalarAttr)
            return verifyCovariant(scalarAttr.getType());

        // Arrays must be covariant and broadcast correctly.
        const auto arrayAttr = llvm::dyn_cast<ekl::ArrayAttr>(attr);
        if (!arrayAttr) return emitError() << "expected array attribute";
        if (failed(verifyCovariant(arrayAttr.getType().getScalarType())))
            return failure();
        return verifyShape(arrayAttr.getType().getShape());
    };

    // Handle broadcast in the stacking dimension.
    if (stack.size() == 1) return verifyElement(*stack.begin());

    // Verify all the elements of the stack expression.
    if (stack.size() != arrayType.getExtent(0))
        return emitError()
            << "expected " << arrayType.getExtent(0) << " elements";
    for (auto element : stack)
        if (failed(verifyElement(element))) return failure();
    return success();
}

//===----------------------------------------------------------------------===//
// SliceAttr implementation
//===----------------------------------------------------------------------===//

auto SliceAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    Slice value) -> LogicalResult
{
    if (value.getStride() == 0)
        return emitError() << "requires non-zero stride";
    return success();
}

//===----------------------------------------------------------------------===//
// EKLDialect implementation
//===----------------------------------------------------------------------===//

auto EKLDialect::parseAttribute(DialectAsmParser &parser, Type type) const
    -> Attribute
{
    if (!parser.parseOptionalColon())
        return SliceAttr::get(parser.getContext(), Slice::id());
    if (!parser.parseOptionalStar()) return AxisAttr::get(parser.getContext());
    if (!parser.parseOptionalEllipsis())
        return EllipsisAttr::get(parser.getContext());

    StringRef keyword;
    Attribute result;
    if (const auto maybeError =
            generatedAttributeParser(parser, &keyword, type, result);
        maybeError.has_value()) {
        if (maybeError.value()) return nullptr;
        return result;
    }

    parser.emitError(parser.getNameLoc(), "unknown attribute: ") << keyword;
    return nullptr;
}

void EKLDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const
{
    llvm::TypeSwitch<Attribute>(attr)
        .Case([&](SliceAttr sliceAttr) {
            if (sliceAttr.getValue() == Slice::id()) {
                os << ":";
                return;
            }

            os << SliceAttr::getMnemonic();
            sliceAttr.print(os);
        })
        .Case([&](AxisAttr) { os << "*"; })
        .Case([&](EllipsisAttr) { os << "..."; })
        .Default([&](Attribute attr) {
            const auto ok = generatedAttributePrinter(attr, os);
            assert(succeeded(ok));
        });
}

void EKLDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "messner/Dialect/EKL/IR/Attributes.cpp.inc"
        >();
}
