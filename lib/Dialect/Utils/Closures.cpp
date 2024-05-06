/// Implements the closure utility functions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Utils/Closures.h"

#include "mlir/IR/Builders.h"

#include <cassert>

using namespace mlir;
using namespace mlir::messner;

/// Parses a mandatory type in a closure.
///
/// This parser implements the following grammar:
///
/// ```
/// type
/// ```
///
/// @param  [in]        parser  OpAsmParser.
/// @param  [out]       type    Result type.
/// @param              mapType An optional ClosureTypeMapFn.
/// @param              segment The current ClosureSegment.
/// @param              index   The index in @p segment .
///
/// @return ParseResult
static ParseResult parseType(
    OpAsmParser &parser,
    Type &type,
    ClosureTypeMapFn mapType,
    ClosureSegment segment,
    unsigned index)
{
    if (parser.parseType(type)) return failure();

    // The type was produced from the input, so it is subject to mapping.
    type = mapType ? mapType(segment, index, type) : type;
    // We do not allow mapping to fail here.
    assert(type);

    return success();
}

/// Parses an optional colon type in a closure.
///
/// This parser implements the following grammar:
///
/// ```
/// opt-type    ::= [ `:` type ]
/// ```
///
/// @param  [in]        parser  OpAsmParser.
/// @param  [out]       type    Result type.
/// @param              mapType An optional ClosureTypeMapFn.
/// @param              segment The current ClosureSegment.
/// @param              index   The index in @p segment .
///
/// @return ParseResult
static ParseResult parseOptionalType(
    OpAsmParser &parser,
    Type &type,
    ClosureTypeMapFn mapType,
    ClosureSegment segment,
    unsigned index)
{
    const auto typeLoc = parser.getCurrentLocation();
    Type givenTy{};
    if (!parser.parseOptionalColon())
        if (parser.parseType(givenTy)) return failure();

    // The type was either absent or produced from the input, so it is subject
    // to mapping.
    type = mapType ? mapType(segment, index, givenTy) : givenTy;

    // If we don't end up with a type, it was either required or rejected by the
    // mapping function.
    if (!type) {
        if (!givenTy) return parser.emitError(typeLoc, "expected type");
        return parser.emitError(typeLoc, "invalid type ") << givenTy;
    }

    return success();
}

/// Prints an optional colon type in a closure.
///
/// This printer implements the following grammar:
///
/// ```
/// opt-type    ::= [ `:` type ]
/// ```
///
/// @param  [in]        printer OpAsmPrinter.
/// @param              type    Type.
/// @param              mapType An optional ClosureTypeMapFn.
/// @param              segment The current ClosureSegment.
/// @param              index   The index in @p segment .
static void printOptionalType(
    OpAsmPrinter &printer,
    Type type,
    ClosureTypeMapFn mapType,
    ClosureSegment segment,
    unsigned index)
{
    // The type is passed to the mapping function to determine what should
    // actually be printed. If nullptr is returned, the type is omitted.
    if ((type = mapType ? mapType(segment, index, type) : type))
        printer << ": " << type;
}

ParseResult mlir::messner::parseOptionalCaptureList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &bodyArguments,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captureOperands,
    SmallVectorImpl<Type> &captureTypes,
    ClosureTypeMapFn mapType)
{
    if (parser.parseOptionalLSquare()) return success();

    unsigned captureIdx = 0;
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            if (parser.parseArgument(bodyArguments.emplace_back())
                || parser.parseEqual()
                || parser.parseOperand(captureOperands.emplace_back())
                || parseOptionalType(
                    parser,
                    bodyArguments.back().type,
                    mapType,
                    ClosureSegment::Captures,
                    captureIdx))
                return failure();

            captureTypes.emplace_back(bodyArguments.back().type);
            return success();
        }))
        return failure();

    return parser.parseRSquare();
}

void mlir::messner::printOptionalCaptureList(
    OpAsmPrinter &printer,
    Block::BlockArgListType bodyArguments,
    OperandRange captureOperands,
    ClosureTypeMapFn mapType)
{
    assert(bodyArguments.size() >= captureOperands.size());

    if (captureOperands.empty()) return;

    printer << "[";

    unsigned captureIdx = 0;
    llvm::interleaveComma(captureOperands, printer, [&](Value op) {
        printer << bodyArguments[captureIdx] << "=" << op;
        printOptionalType(
            printer,
            op.getType(),
            mapType,
            ClosureSegment::Captures,
            captureIdx++);
    });

    printer << "]";
}

ParseResult mlir::messner::parseArgumentList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &bodyArguments,
    ClosureTypeMapFn mapType)
{
    unsigned inputIdx = 0;
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
            if (parser.parseArgument(bodyArguments.emplace_back()))
                return failure();

            return parseOptionalType(
                parser,
                bodyArguments.back().type,
                mapType,
                ClosureSegment::Inputs,
                inputIdx++);
        });
}

void mlir::messner::printArgumentList(
    OpAsmPrinter &printer,
    Block::BlockArgListType argumentRange,
    ClosureTypeMapFn mapType)
{
    printer << "(";

    unsigned inputIdx = 0;
    llvm::interleaveComma(argumentRange, printer, [&](BlockArgument &arg) {
        printer << arg;
        printOptionalType(
            printer,
            arg.getType(),
            mapType,
            ClosureSegment::Inputs,
            inputIdx++);
    });

    printer << ")";
}

OptionalParseResult mlir::messner::parseOptionalYieldTypes(
    OpAsmParser &parser,
    SmallVectorImpl<Type> &types,
    ClosureTypeMapFn mapType)
{
    if (parser.parseOptionalArrow()) return std::nullopt;

    if (parser.parseOptionalLParen()) {
        // A tuple of results must be delimited by parenthesis, so this is just
        // a single type.
        return parseType(
            parser,
            types.emplace_back(),
            mapType,
            ClosureSegment::Results,
            0);
    }

    if (!parser.parseOptionalRParen()) {
        // The result type list is empty.
        return success();
    }

    unsigned resultIdx = 0;
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
            return parseType(
                parser,
                types.emplace_back(),
                mapType,
                ClosureSegment::Results,
                resultIdx++);
        }))
        return failure();

    return parser.parseRParen();
}

void mlir::messner::printOptionalYieldTypes(
    OpAsmPrinter &printer,
    TypeRange types,
    ClosureTypeMapFn mapType)
{
    // Types need to be mapped in advance to determine elision.
    SmallVector<Type> mappedTys;
    if (mapType) {
        mappedTys = llvm::to_vector(types);
        for (auto [idx, ty] : llvm::enumerate(mappedTys))
            ty = mapType(ClosureSegment::Results, idx, ty);

        // Only elide all types or none.
        const auto numElided = llvm::count(mappedTys, Type{});
        if (static_cast<std::size_t>(numElided) == types.size()) return;

        types = mappedTys;
    }

    if (types.empty()) return;

    printer << " -> ";

    if (types.size() == 1) {
        printer << types[0];
        return;
    }

    printer << "(";
    llvm::interleaveComma(types, printer);
    printer << ")";
}

OptionalParseResult mlir::messner::parseOptionalDelegate(
    OpAsmParser &parser,
    FunctionType type,
    SmallVectorImpl<Type> &resultTypes,
    Region &body,
    OperationName yieldOp,
    ClosureTypeMapFn mapType)
{
    assert(type);
    assert(body.empty());

    if (parser.parseOptionalLBrace()) return std::nullopt;

    // Parse the payload operation.
    const auto opName = parser.parseCustomOperationName();
    NamedAttrList opAttrs;
    std::optional<Location> opLoc;
    if (failed(opName) || parser.parseOptionalAttrDict(opAttrs)
        || parser.parseOptionalLocationSpecifier(opLoc) || parser.parseRBrace())
        return failure();

    // Determine the payload location.
    const auto unknownLoc = parser.getBuilder().getUnknownLoc();
    const auto loc        = opLoc.value_or(unknownLoc);

    // Determine what types we will expect the payload to yield.;
    const auto hasResultTypes =
        parseOptionalYieldTypes(parser, resultTypes, mapType);
    if (hasResultTypes.has_value() && hasResultTypes.value()) return failure();
    const auto yieldTypes = hasResultTypes.has_value()
                              ? ArrayRef<Type>(resultTypes)
                              : type.getResults();

    // Prepare the block for insertion.
    auto &block = body.emplaceBlock();
    SmallVector<Location> argLocs(yieldTypes.size(), unknownLoc);
    block.addArguments(yieldTypes, argLocs);

    OpBuilder builder(&block, block.begin());

    auto op = builder.create(
        loc,
        builder.getStringAttr(opName->getStringRef()),
        block.getArguments(),
        yieldTypes,
        opAttrs);
    builder.create(
        loc,
        builder.getStringAttr(yieldOp.getStringRef()),
        op->getResults());

    return success();
}

Operation *mlir::messner::matchDelegate(Region &body, TypeRange argumentTypes)
{
    // Must have single block.
    if (body.empty()) return nullptr;
    auto &block = body.front();
    if (&block != &body.back()) return nullptr;

    // Must be (op, yield).
    if (block.empty()) return nullptr;
    auto op    = &block.front();
    auto yield = &block.back();
    if (!yield || yield != op->getNextNode()) return nullptr;

    // Sanity checks for the yield operation.
    assert(yield->getNumRegions() == 0);
    assert(yield->getNumResults() == 0);
    assert(yield->hasTrait<OpTrait::IsTerminator>());

    // The op must not have any nested regions.
    if (op->getNumRegions() != 0) return nullptr;

    // The op must accept all expected arguments in order.
    if (block.getArgumentTypes() != argumentTypes) return nullptr;
    if (op->getOperands() != block.getArguments()) return nullptr;

    // The yield must accept all results of op in order.
    if (yield->getOperands() != op->getResults()) return nullptr;
    // The yield must not have any extra attributes.
    if (!yield->getAttrs().empty()) return nullptr;

    return op;
}

void mlir::messner::printDelegate(
    OpAsmPrinter &printer,
    Operation *delegate,
    ClosureTypeMapFn mapType)
{
    assert(delegate);

    printer << "{ ";
    printer << delegate->getName().getStringRef();
    printer.printOptionalAttrDict(delegate->getAttrs());
    if (!llvm::isa<UnknownLoc>(delegate->getLoc()))
        printer.printOptionalLocationSpecifier(delegate->getLoc());
    printer << " }";

    printOptionalYieldTypes(printer, delegate->getResultTypes(), mapType);
}

ParseResult mlir::messner::parseClosure(
    OpAsmParser &parser,
    FunctionType type,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captureOperands,
    SmallVectorImpl<Type> &captureTypes,
    SmallVectorImpl<Type> &resultTypes,
    Region &body,
    std::optional<OperationName> yieldOp,
    ClosureTypeMapFn mapType)
{
    assert(body.empty());

    if (type && yieldOp) {
        const auto maybeDelegate = parseOptionalDelegate(
            parser,
            type,
            resultTypes,
            body,
            *yieldOp,
            mapType);
        if (maybeDelegate.has_value()) return maybeDelegate.value();
    }

    SmallVector<OpAsmParser::Argument> bodyArguments;

    if (parseOptionalCaptureList(
            parser,
            bodyArguments,
            captureOperands,
            captureTypes,
            mapType)
        || parseArgumentList(parser, bodyArguments, mapType))
        return failure();

    const auto hasResultTypes =
        parseOptionalYieldTypes(parser, resultTypes, mapType);
    if (hasResultTypes.has_value() && hasResultTypes.value()) return failure();
    if (!hasResultTypes.has_value() && type)
        resultTypes.assign(type.getResults().begin(), type.getResults().end());

    return parser.parseRegion(body, bodyArguments, true);
}

void mlir::messner::printClosure(
    OpAsmPrinter &printer,
    TypeRange inputTypes,
    OperandRange captureOperands,
    TypeRange resultTypes,
    Region &body,
    ClosureTypeMapFn mapType)
{
    if (const auto delegate = matchDelegate(body, inputTypes)) {
        printDelegate(printer, delegate, mapType);
        return;
    }

    printOptionalCaptureList(
        printer,
        body.getArguments(),
        captureOperands,
        mapType);

    printArgumentList(
        printer,
        body.getArguments().drop_front(captureOperands.size()),
        mapType);

    printOptionalYieldTypes(printer, resultTypes, mapType);

    printer << " ";
    printer.printRegion(body, false);
}
