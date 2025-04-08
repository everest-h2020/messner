/// Implements the custom assembly format for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Assembly.h"

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/IR/Dialect.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Field serialization
//===----------------------------------------------------------------------===//

auto mlir::ekl::operator<<(llvm::raw_ostream &os, const Slice &slice)
    -> llvm::raw_ostream &
{
    if (slice.getStride() >= 0) {
        if (slice.getBegin() != Index::begin()) os << slice.getBegin();
        os << ":";
        if (slice.getEnd() != Index::end()) os << slice.getEnd();
    } else {
        if (slice.getBegin() != Index::rbegin()) os << slice.getBegin();
        os << ":";
        if (slice.getEnd() != Index::rend()) os << slice.getEnd();
    }

    if (slice.getStride() != 1) os << ":" << slice.getStride();
    return os;
}

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

auto mlir::ekl::parseOptionalOffset(AsmParser &parser, Offset &result)
    -> OptionalParseResult
{
    Offset::value_type value;
    const auto maybeValue = parser.parseOptionalInteger(value);
    if (!maybeValue.has_value()) return std::nullopt;
    if (*maybeValue) return failure();

    result = Offset(value);
    return success();
}

auto mlir::ekl::parseOptionalIndex(AsmParser &parser, Index &result)
    -> OptionalParseResult
{
    auto isFromEnd = !parser.parseOptionalKeyword("end");
    if (isFromEnd) std::ignore = parser.parseOptionalPlus();

    Offset offset;
    const auto maybeOffset = parseOptionalOffset(parser, offset);
    if (!maybeOffset.has_value()) return std::nullopt;
    if (failed(*maybeOffset)) return failure();

    result = Index(offset, isFromEnd);
    return success();
}

auto mlir::ekl::parseStaticExtent(AsmParser &parser, Extent &result)
    -> ParseResult
{
    const auto loc = parser.getCurrentLocation();
    Extent::value_type value;
    if (parser.parseInteger(value)) return failure();
    if (value == 0 || value > Extent::value_max)
        return parser.emitError(loc, "expected in-bounds extent");
    result = Extent(value);
    return success();
}

auto mlir::ekl::parseSlice(
    AsmParser &parser,
    std::optional<Index> &begin,
    std::optional<Index> &end,
    Offset &stride) -> ParseResult
{
    const auto parseIndex = [&](Index &result) -> ParseResult {
        const auto maybeIndex = FieldParser<Index>::parse(parser);
        if (failed(maybeIndex)) return failure();
        result = *maybeIndex;
        return success();
    };
    const auto parseStride = [&]() -> ParseResult {
        const auto loc = parser.getCurrentLocation();
        Offset::value_type strideValue;
        if (parser.parseInteger(strideValue)) return failure();
        if (strideValue == 0)
            return parser.emitError(loc, "expected non-zero stride");
        stride = Offset(strideValue);
        return success();
    };

    stride = 1;

    if (!parser.parseOptionalColon()) {
        if (!parser.parseOptionalColon()) return parseStride();
        const auto maybeEnd = parseOptionalIndex(parser, end.emplace());
        if (!maybeEnd.has_value())
            end.reset();
        else if (*maybeEnd)
            return failure();
        if (!parser.parseOptionalColon()) return parseStride();
        return success();
    }
    if (parseIndex(begin.emplace())) return failure();
    if (parser.parseColon()) return failure();
    if (!parser.parseOptionalColon()) return parseStride();
    const auto maybeEnd = parseOptionalIndex(parser, end.emplace());
    if (!maybeEnd.has_value())
        end.reset();
    else if (*maybeEnd)
        return failure();
    if (!parser.parseOptionalColon()) return parseStride();
    return success();
}

void mlir::ekl::printSlice(
    AsmPrinter &printer,
    const std::optional<Index> &begin,
    const std::optional<Index> &end,
    const Offset &stride)
{
    if (begin) printer << *begin;
    printer << ":";
    if (end) printer << *end;
    if (stride != 1) printer << ":" << stride;
}

auto mlir::ekl::parseRational(AsmParser &parser, Rational &rational)
    -> ParseResult
{
    auto maybeNumber = Rational::parseField(parser);
    if (failed(maybeNumber)) return failure();

    rational = std::move(*maybeNumber);
    return success();
}

void mlir::ekl::printRational(AsmPrinter &printer, const Rational &rational)
{
    rational.printField(printer.getStream());
}

auto mlir::ekl::parseStaticShape(AsmParser &parser, ShapeBuilder &shape)
    -> ParseResult
{
    return parser.parseCommaSeparatedList(
        AsmParser::Delimiter::OptionalSquare,
        [&]() -> ParseResult {
            return parseStaticExtent(parser, shape.emplace_back());
        });
}

void mlir::ekl::printStaticShape(AsmPrinter &printer, ShapeRef shape)
{
    if (shape.empty()) return;
    printer << "[";
    llvm::interleaveComma(shape, printer);
    printer << "]";
}

auto mlir::ekl::parseReferenceShape(AsmParser &parser, ShapeBuilder &shape)
    -> ParseResult
{
    if (parser.parseOptionalLSquare()) return success();

    if (!parser.parseOptionalRSquare()) return success();
    if (!parser.parseOptionalEllipsis()) {
        shape.emplace_back(unbounded);
        if (!parser.parseOptionalRSquare()) return success();
        if (parser.parseComma()) return failure();
    }

    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            return parseStaticExtent(parser, shape.emplace_back());
        }))
        return failure();

    return parser.parseRSquare();
}

void mlir::ekl::printReferenceShape(AsmPrinter &printer, ShapeRef shape)
{
    if (shape.empty()) return;

    printer << "[";
    if (!shape.empty() && shape.front() == unbounded) {
        printer << "...";
        shape = shape.drop_front();
        if (!shape.empty()) printer << ", ";
    }
    llvm::interleaveComma(shape, printer);
    printer << "]";
}

auto mlir::ekl::parseKeywordOrString(AsmParser &parser, StringAttr &result)
    -> ParseResult
{
    std::string str;
    if (parser.parseKeywordOrString(&str)) return failure();
    result = StringAttr::get(parser.getContext(), str);
    return success();
}

void mlir::ekl::printKeywordOrString(AsmPrinter &printer, StringAttr str)
{
    printer.printKeywordOrString(str);
}

auto mlir::ekl::parseOptionalTuple(AsmParser &parser, TupleAttr &result)
    -> ParseResult
{
    SmallVector<Attribute> args;
    if (!parser.parseOptionalLParen()) {
        if (parser.parseOptionalRParen()) {
            if (parser.parseCommaSeparatedList([&]() {
                    return parser.parseAttribute(args.emplace_back());
                }))
                return failure();
            if (parser.parseRParen()) return failure();
        }
    }

    result = TupleAttr::get(parser.getContext(), args);
    return success();
}

void mlir::ekl::printOptionalTuple(AsmPrinter &printer, TupleAttr args)
{
    assert(args);

    if (args.empty()) return;
    printer << "(";
    llvm::interleaveComma(args, printer);
    printer << ")";
}

auto mlir::ekl::parseOptionalExprType(AsmParser &parser, Type &type)
    -> ParseResult
{
    if (parser.parseOptionalColon()) {
        type = ExpressionType::get(parser.getContext());
        return success();
    }

    return parser.parseType(type);
}

void mlir::ekl::printOptionalExprType(AsmPrinter &printer, Type type)
{
    assert(type);

    if (llvm::isa<ExpressionType>(type)) return;
    printer << ": " << type;
}

auto mlir::ekl::parseOperand(
    OpAsmParser &parser,
    OpAsmParser::UnresolvedOperand &operand,
    Type &type) -> ParseResult
{
    if (parser.parseOperand(operand)) return failure();
    return parseOptionalExprType(parser, type);
}

void mlir::ekl::printOperand(
    OpAsmPrinter &printer,
    Operation *,
    Value operand,
    Type)
{
    printer.printOperand(operand);
    printOptionalExprType(printer, operand.getType());
}

auto mlir::ekl::parseOperandList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) -> ParseResult
{
    return parser.parseCommaSeparatedList([&]() {
        return parseOperand(
            parser,
            operands.emplace_back(),
            types.emplace_back());
    });
}

void mlir::ekl::printOperandList(
    OpAsmPrinter &printer,
    Operation *op,
    ValueRange operands,
    TypeRange)
{
    llvm::interleaveComma(operands, printer, [&](Value operand) {
        printOperand(printer, op, operand, operand.getType());
    });
}

auto mlir::ekl::parseResultType(AsmParser &parser, Type &type) -> ParseResult
{
    type = ExpressionType::get(parser.getContext());
    if (parser.parseOptionalArrow()) return success();
    return parser.parseType(type);
}

void mlir::ekl::printResultType(AsmPrinter &printer, Operation *, Type type)
{
    if (llvm::isa<ExpressionType>(type)) return;
    printer << "-> " << type;
}

//===----------------------------------------------------------------------===//
// mlir::FieldParser<Extent> implementation
//===----------------------------------------------------------------------===//

auto mlir::FieldParser<Extent>::parse(AsmParser &parser) -> FailureOr<Extent>
{
    if (!parser.parseOptionalQuestion()) return Extent(unbounded);

    Extent::value_type value;
    const auto loc = parser.getCurrentLocation();
    if (parser.parseInteger(value)) return failure();
    if (value > Extent::value_max)
        return parser.emitError(loc, "extent value out of range");

    return Extent(value);
}

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

auto FuncOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
    const auto fnTypeBuilder = [](Builder &builder,
                                  ArrayRef<Type> inputs,
                                  ArrayRef<Type> results,
                                  function_interface_impl::VariadicFlag flag,
                                  std::string &error) -> FunctionType {
        if (flag.isVariadic()) {
            error.assign("variadic functions are not supported");
            return nullptr;
        }

        return builder.getFunctionType(inputs, results);
    };

    return function_interface_impl::parseFunctionOp(
        parser,
        result,
        false,
        getFunctionTypeAttrName(result.name),
        fnTypeBuilder,
        getArgAttrsAttrName(result.name),
        getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &printer)
{
    function_interface_impl::printFunctionOp(
        printer,
        *this,
        false,
        getFunctionTypeAttrName(),
        getArgAttrsAttrName(),
        getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

auto KernelOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
    StringAttr nameAttr;
    SmallVector<OpAsmParser::Argument> args;
    auto &definition = *result.addRegion();

    if (parser.parseSymbolName(
            nameAttr,
            SymbolTable::getSymbolAttrName(),
            result.attributes)
        || parser
               .parseArgumentList(args, AsmParser::Delimiter::Paren, true, true)
        || parser.parseOptionalAttrDictWithKeyword(result.attributes)
        || parser.parseRegion(definition, args, true))
        return failure();

    if (definition.empty()) {
        assert(args.empty());
        definition.emplaceBlock();
    }

    auto anyArgAttrs       = false;
    const auto allArgAttrs = llvm::map_to_vector(
        args,
        [&](const OpAsmParser::Argument &arg) -> Attribute {
            anyArgAttrs |= arg.attrs && !arg.attrs.empty();
            return arg.attrs;
        });
    if (anyArgAttrs) {
        result.addAttribute(
            getArgAttrsAttrName(result.name),
            TupleAttr::get(parser.getContext(), allArgAttrs));
    }

    return success();
}

void KernelOp::print(OpAsmPrinter &printer)
{
    printer << " ";
    printer.printSymbolName(getSymName());

    const auto allArgAttrs = getArgAttrs().value_or(TupleAttr{});
    printer << "(";
    auto index = 0U;
    llvm::interleaveComma(
        getDefinition().getArguments(),
        printer,
        [&](const BlockArgument &arg) {
            const auto argAttrs =
                allArgAttrs ? llvm::cast<DictionaryAttr>(allArgAttrs[index++])
                                  .getValue()
                            : ArrayRef<NamedAttribute>{};
            printer.printRegionArgument(arg, argAttrs);
        });
    printer << ")";

    printer.printOptionalAttrDictWithKeyword(
        (*this)->getAttrs(),
        getAttributeNames());

    printer << " ";
    printer.printRegion(getDefinition(), false, false);
}

//===----------------------------------------------------------------------===//
// StaticOp implementation
//===----------------------------------------------------------------------===//

auto StaticOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
    StringAttr nameAttr;
    if (impl::parseOptionalVisibilityKeyword(parser, result.attributes)
        || parser.parseSymbolName(
            nameAttr,
            getSymNameAttrName(result.name),
            result.attributes))
        return failure();

    ReferenceType type;
    if (parser.parseColonType(type)) return failure();
    result.addAttribute(getTypeAttrName(result.name), TypeAttr::get(type));

    if (parser.parseOptionalEqual()) return success();
    ArrayAttr initializerAttr;
    if (parser.parseCustomAttributeWithFallback(
            initializerAttr,
            type.getCellType()))
        return failure();
    result.addAttribute(getInitializerAttrName(result.name), initializerAttr);

    return parser.parseOptionalAttrDict(result.attributes);
}

void StaticOp::print(OpAsmPrinter &printer)
{
    printer << " ";
    if (const auto maybeVisibility = getSymVisibility();
        maybeVisibility && *maybeVisibility != "public")
        printer << *maybeVisibility << " ";

    printer.printSymbolName(getSymName());
    printer << " : " << getType();

    if (const auto maybeInit = getInitializer(); maybeInit) {
        printer << " = ";
        if (maybeInit->getType() == getType().getCellType())
            printer.printStrippedAttrOrType(*maybeInit);
        else
            printer << *maybeInit;
    }

    printer.printOptionalAttrDict((*this)->getAttrs(), getAttributeNames());
}
