/// Implementation of the EKL dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Types.h"

#include "messner/Dialect/EKL/Analysis/Offset.h"
#include "messner/Dialect/EKL/IR/Assembly.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/EKL/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IndexType implementation
//===----------------------------------------------------------------------===//

auto ekl::IndexType::parse(AsmParser &parser) -> Type
{
    Extent bound   = unbounded;
    bool isFromEnd = false;

    if (!parser.parseOptionalLess()) {
        isFromEnd = !parser.parseOptionalMinus();

        const auto maybeBound = FieldParser<Extent>::parse(parser);
        if (failed(maybeBound)) return {};
        bound = *maybeBound;
    }
    return get(parser.getContext(), bound, isFromEnd);
}

void ekl::IndexType::print(AsmPrinter &printer) const
{
    if (!isFromEnd() && !isBounded()) return;

    printer << "<";
    if (isFromEnd()) printer << "-";
    printer << getBound() << ">";
}

//===----------------------------------------------------------------------===//
// ArrayType implementation
//===----------------------------------------------------------------------===//

auto ArrayType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ScalarType scalarType,
    ShapeRef shape) -> LogicalResult
{
    if (!scalarType) return emitError() << "scalar type required";
    const auto flattened = flatten(shape);
    if (failed(flattened)) return emitError() << "shape is too large";
    if (!flattened->isBounded()) return emitError() << "shape is unbounded";
    if (*flattened == Extent{})
        return emitError() << "shape is trivially empty";
    return success();
}

//===----------------------------------------------------------------------===//
// ReferenceType implementation
//===----------------------------------------------------------------------===//

auto ReferenceType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ref::ReferenceKind,
    ScalarType scalarType,
    ShapeRef shape) -> LogicalResult
{
    if (!shape.empty() && shape.front() == unbounded)
        shape = shape.drop_front();
    return ArrayType::verify(emitError, scalarType, shape);
}

//===----------------------------------------------------------------------===//
// SliceType implementation
//===----------------------------------------------------------------------===//

auto SliceType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    std::optional<Index>,
    std::optional<Index>,
    Offset stride) -> LogicalResult
{
    if (stride == 0) return emitError() << "non-zero stride required";
    return success();
}

auto SliceType::parse(AsmParser &parser) -> Type
{
    std::optional<Index> begin, end;
    Offset stride = 1;
    if (!parser.parseOptionalLess()) {
        if (parseSlice(parser, begin, end, stride) || parser.parseGreater())
            return {};
    }
    return get(parser.getContext(), begin, end, stride);
}

void SliceType::print(AsmPrinter &printer) const
{
    if (!getBegin() && !getEnd() && getStride() == 1) return;
    printer << "<";
    printSlice(printer, getBegin(), getEnd(), getStride());
    printer << ">";
}

//===----------------------------------------------------------------------===//
// EKLDialect implementation
//===----------------------------------------------------------------------===//

auto EKLDialect::parseType(DialectAsmParser &parser) const -> Type
{
    if (!parser.parseOptionalQuestion())
        return ExpressionType::get(parser.getContext());
    if (!parser.parseOptionalStar()) return AxisType::get(parser.getContext());
    if (!parser.parseOptionalEllipsis())
        return EllipsisType::get(parser.getContext());

    StringRef keyword;
    Type result;
    if (const auto maybeError = generatedTypeParser(parser, &keyword, result);
        maybeError.has_value()) {
        if (maybeError.value()) return nullptr;
        return result;
    }

    parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
    return nullptr;
}

void EKLDialect::printType(Type type, DialectAsmPrinter &os) const
{
    llvm::TypeSwitch<Type>(type)
        .Case([&](ExpressionType) { os << "?"; })
        .Case([&](AxisType) { os << "*"; })
        .Case([&](EllipsisType) { os << "..."; })
        .Default([&](Type type) {
            const auto ok = generatedTypePrinter(type, os);
            assert(succeeded(ok));
        });
}

void EKLDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "messner/Dialect/EKL/IR/Types.cpp.inc"
        >();
}
