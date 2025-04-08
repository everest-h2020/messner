/// Declares the custom assembly format for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/Analysis/Index.h"
#include "messner/Dialect/EKL/Analysis/Offset.h"
#include "messner/Dialect/EKL/Analysis/Rational.h"
#include "messner/Dialect/EKL/Analysis/Shape.h"
#include "messner/Dialect/EKL/Analysis/Slice.h"
#include "messner/Dialect/EKL/IR/Dialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Field serialization
//===----------------------------------------------------------------------===//

auto operator<<(llvm::raw_ostream &os, const Extent &extent)
    -> llvm::raw_ostream &;

auto operator<<(llvm::raw_ostream &os, const Offset &offset)
    -> llvm::raw_ostream &;

auto operator<<(llvm::raw_ostream &os, const Index &index)
    -> llvm::raw_ostream &;

auto operator<<(llvm::raw_ostream &os, const Slice &slice)
    -> llvm::raw_ostream &;

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

auto parseOffset(AsmParser &parser, Offset &result) -> ParseResult;

auto parseOptionalOffset(AsmParser &parser, Offset &result)
    -> OptionalParseResult;

auto parseIndex(AsmParser &parser, Index &result) -> ParseResult;

auto parseOptionalIndex(AsmParser &parser, Index &result)
    -> OptionalParseResult;

auto parseStaticExtent(AsmParser &parser, Extent &result) -> ParseResult;

auto parseSlice(
    AsmParser &parser,
    std::optional<Index> &begin,
    std::optional<Index> &end,
    Offset &stride) -> ParseResult;

void printSlice(
    AsmPrinter &printer,
    const std::optional<Index> &begin,
    const std::optional<Index> &end,
    const Offset &stride);

auto parseRational(AsmParser &parser, Rational &rational) -> ParseResult;

void printRational(AsmPrinter &printer, const Rational &rational);

auto parseStaticShape(AsmParser &parser, ShapeBuilder &shape) -> ParseResult;

void printStaticShape(AsmPrinter &printer, ShapeRef shape);

auto parseReferenceShape(AsmParser &parser, ShapeBuilder &shape) -> ParseResult;

void printReferenceShape(AsmPrinter &printer, ShapeRef shape);

auto parseKeywordOrString(AsmParser &parser, StringAttr &result) -> ParseResult;

void printKeywordOrString(AsmPrinter &printer, StringAttr str);

auto parseOptionalTuple(AsmParser &parser, TupleAttr &result) -> ParseResult;

void printOptionalTuple(AsmPrinter &printer, TupleAttr args);

auto parseOptionalExprType(AsmParser &parser, Type &type) -> ParseResult;

void printOptionalExprType(AsmPrinter &printer, Type type);

auto parseOperand(
    OpAsmParser &parser,
    OpAsmParser::UnresolvedOperand &operand,
    Type &type) -> ParseResult;

void printOperand(OpAsmPrinter &printer, Operation *, Value operand, Type);

auto parseOperandList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) -> ParseResult;

void printOperandList(
    OpAsmPrinter &printer,
    Operation *op,
    ValueRange operands,
    TypeRange);

auto parseResultType(AsmParser &parser, Type &type) -> ParseResult;

void printResultType(AsmPrinter &printer, Operation *, Type type);

} // namespace mlir::ekl

namespace mlir {

//===----------------------------------------------------------------------===//
// Field deserialization
//===----------------------------------------------------------------------===//

template<>
struct FieldParser<ekl::Extent> {
    static auto parse(AsmParser &parser) -> FailureOr<ekl::Extent>;
};

template<>
struct FieldParser<ekl::Offset> {
    static auto parse(AsmParser &parser) -> FailureOr<ekl::Offset>;
};

template<>
struct FieldParser<ekl::Index> {
    static auto parse(AsmParser &parser) -> FailureOr<ekl::Index>;
};

template<>
struct FieldParser<ekl::Slice> {
    static auto parse(AsmParser &parser) -> FailureOr<ekl::Slice>;
};

} // namespace mlir

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// operator<<
//===----------------------------------------------------------------------===//

inline auto operator<<(llvm::raw_ostream &os, const Extent &extent)
    -> llvm::raw_ostream &
{
    if (!extent.isBounded())
        os << "?";
    else
        os << extent.getValue();
    return os;
}

inline auto operator<<(llvm::raw_ostream &os, const Offset &offset)
    -> llvm::raw_ostream &
{
    os << offset.getValue();
    return os;
}

inline auto operator<<(llvm::raw_ostream &os, const Index &index)
    -> llvm::raw_ostream &
{
    const auto value = index.getOffset();
    if (index.isFromEnd()) {
        os << "end";
        if (value == 0) return os;
        if (value > 0) os << "+";
    }
    os << value;
    return os;
}

//===----------------------------------------------------------------------===//
// parseOffset
//===----------------------------------------------------------------------===//

inline auto parseOffset(AsmParser &parser, Offset &result) -> ParseResult
{
    const auto maybe = parseOptionalOffset(parser, result);
    if (!maybe.has_value())
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected offset value");
    return *maybe;
}

//===----------------------------------------------------------------------===//
// parseIndex
//===----------------------------------------------------------------------===//

inline auto parseIndex(AsmParser &parser, Index &result) -> ParseResult
{
    const auto maybe = parseOptionalIndex(parser, result);
    if (!maybe.has_value())
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected index value");
    return *maybe;
}

} // namespace mlir::ekl

namespace mlir {

//===----------------------------------------------------------------------===//
// FieldParser<ekl::Offset> implementation
//===----------------------------------------------------------------------===//

inline auto FieldParser<ekl::Offset>::parse(AsmParser &parser)
    -> FailureOr<ekl::Offset>
{
    ekl::Offset result;
    if (ekl::parseOffset(parser, result)) return failure();
    return result;
}

//===----------------------------------------------------------------------===//
// FieldParser<ekl::Index> implementation
//===----------------------------------------------------------------------===//

inline auto FieldParser<ekl::Index>::parse(AsmParser &parser)
    -> FailureOr<ekl::Index>
{
    ekl::Index result;
    if (ekl::parseIndex(parser, result)) return failure();
    return result;
}

//===----------------------------------------------------------------------===//
// FieldParser<ekl::Slice> implementation
//===----------------------------------------------------------------------===//

inline auto FieldParser<ekl::Slice>::parse(AsmParser &parser)
    -> FailureOr<ekl::Slice>
{
    std::optional<ekl::Index> begin, end;
    ekl::Offset stride;
    if (parseSlice(parser, begin, end, stride)) return failure();
    return ekl::Slice(begin, end, stride);
}

} // namespace mlir
