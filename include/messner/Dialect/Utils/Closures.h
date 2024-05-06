/// Declares helper functions to work with closure-like operations.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/OpImplementation.h"

#include <cassert>

namespace mlir::messner {

/// Enumeration of value ranges present in a closure.
enum class ClosureSegment {
    /// Range of captured arguments.
    Captures,
    /// Range of input arguments.
    Inputs,
    /// Range of results.
    Results
};

/// Reference to a function that maps a type during closure parsing.
///
/// When printing or parsing a closure capture, argument or result type, the
/// user may want to override the default behavior to better suit their dialect.
///
/// During parsing, the mapping function is applied to every type that is parsed
/// from the input, including absent types (`nullptr`). The mapping function
/// must produce the concrete type to use, or `nullptr` to indicate failure.
///
/// During printing, the mapping function is applied to every type that is
/// written to the output. It must produce the concrete type to output, or
/// `nullptr` to indicate omission. It must be the inverse of the parsing map
/// function in that regard.
using ClosureTypeMapFn =
    llvm::function_ref<Type(ClosureSegment, unsigned, Type)>;

/// Parses an optional delimited list of capture operands.
///
/// This parser implements the following grammar:
///
/// ```
/// capture-list    ::= [ `[` capture { `,` capture } `]` ]
/// capture         ::= ssa-id `=` value-use [ `:` type ]
/// ```
///
/// @param  [in]        parser          OpAsmParser.
/// @param  [out]       bodyArguments   Arguments to the closure body.
/// @param  [out]       captureOperands Captured operands.
/// @param  [out]       captureTypes    Types of @p captureOperands
/// @param              mapType         An optional ClosureTypeMapFn.
///
/// @return ParseResult
ParseResult parseOptionalCaptureList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &bodyArguments,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captureOperands,
    SmallVectorImpl<Type> &captureTypes,
    ClosureTypeMapFn mapType = {});

/// Prints an optional delimited list of capture operands.
///
/// This printer implements the following grammar:
///
/// ```
/// capture-list    ::= [ `[` capture { `,` capture } `]` ]
/// capture         ::= ssa-id `=` value-use [ `:` type ]
/// ```
///
/// @param  [in]        printer         OpAsmPrinter.
/// @param  [in]        bodyArguments   Arguments to the closure body.
/// @param              captureOperands Captured operands.
/// @param              mapType         An optional ClosureTypeMapFn.
///
/// @pre    `bodyArguments.size() >= captureOperands.size()`
void printOptionalCaptureList(
    OpAsmPrinter &printer,
    Block::BlockArgListType bodyArguments,
    OperandRange captureOperands,
    ClosureTypeMapFn mapType = {});

/// Parses a list of closure arguments.
///
/// This parser implements the following grammar:
///
/// ```
/// arg-list    ::= `(` [ arg { `,` arg } ] `)`
/// arg         ::= ssa-id [ `:` type ]
/// ```
///
/// @param  [in]        parser          OpAsmParser.
/// @param  [out]       bodyArguments   Arguments to the closure body.
/// @param              mapType         An optional ClosureTypeMapFn.
///
/// @return ParseResult
ParseResult parseArgumentList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &bodyArguments,
    ClosureTypeMapFn mapType = {});

/// Prints a list of closure arguments.
///
/// This printer implements the following grammar:
///
/// ```
/// arg-list    ::= `(` [ arg { `,` arg } ] `)`
/// arg         ::= ssa-id [ `:` type ]
/// ```
///
/// @param  [in]        printer         OpAsmPrinter.
/// @param  [in]        argumentRange   Arguments minus captures.
/// @param              mapType         An optional ClosureTypeMapFn.
void printArgumentList(
    OpAsmPrinter &printer,
    Block::BlockArgListType argumentRange,
    ClosureTypeMapFn mapType = {});

/// Parses an optional yield type list.
///
/// This parser implements the following grammar:
///
/// ```
/// yield-types         ::= [ `->` type-or-type-list ]
/// type-or-type-list   ::= type | `(` [ type { `,` type } ] `)`
/// ```
///
/// @param  [in]        parser  OpAsmParser.
/// @param  [out]       types   Result types.
/// @param              mapType An optional ClosureTypeMapFn.
///
/// @return OptionalParseResult
OptionalParseResult parseOptionalYieldTypes(
    OpAsmParser &parser,
    SmallVectorImpl<Type> &types,
    ClosureTypeMapFn mapType = {});

/// Prints an optional yield type list.
///
/// If @p mapType elides all types, the whole construct is elided, otherwise the
/// shortest possible form is printed.
///
/// This parser implements the following grammar:
///
/// ```
/// yield-types         ::= [ `->` type-or-type-list ]
/// type-or-type-list   ::= type | `(` [ type { `,` type } ] `)`
/// ```
///
/// @param  [in]        printer OpAsmPrinter.
/// @param              types   Types.
/// @param              mapType An optional ClosureTypeMapFn.
void printOptionalYieldTypes(
    OpAsmPrinter &printer,
    TypeRange types,
    ClosureTypeMapFn mapType = {});

/// Parses an optional delegate operation.
///
/// This parser implements the following grammar:
///
/// ```
/// op-delegate     ::= `{` op-name attr-dict loc `}` yield-types
/// yield-types     ::= [ `->` type-or-type-list ]
/// ```
///
/// @param  [in]        parser      OpAsmParser.
/// @param              type        Expected delegate type.
/// @param  [out]       resultTypes Result types.
/// @param  [out]       body        Body region.
/// @param              yieldOp     Body yield operation.
/// @param              mapType     An optional ClosureTypeMapFn.
///
/// @pre    `type`
/// @pre    `body.empty()`
///
/// @return OptionalParseResult
OptionalParseResult parseOptionalDelegate(
    OpAsmParser &parser,
    FunctionType type,
    SmallVectorImpl<Type> &resultTypes,
    Region &body,
    OperationName yieldOp,
    ClosureTypeMapFn mapType = {});

/// Matches the contents of @p body against a single delegate operation.
///
/// A delegate operation takes all of the arguments of @p body in the order
/// they are defined in, and produces a range of results that are consumed by
/// the only other operation in @p body , its terminator.
///
/// @param  [in]        body          Body region.
/// @param              argumentTypes Allowed argument types.
///
/// @pre    Any terminator in @p body must be a trivial yield.
///
/// @retval Operation*  Single delegate operation in @p body .
/// @retval nullptr     @p body is not a regular delegate.
[[nodiscard]] Operation *matchDelegate(Region &body, TypeRange argumentTypes);

/// Prints a delegate operation.
///
/// This printer implements the following grammar:
///
/// ```
/// op-delegate     ::= `{` op-name attr-dict loc `}` yield-types
/// yield-types     ::= [ `->` type-or-type-list ]
/// ```
///
/// @param  [in]        printer     OpAsmPrinter.
/// @param  [in]        delegate    The delegate operation.
/// @param              mapType     An optional ClosureTypeMapFn.
///
/// @pre    @p delegate must be a regular delegate or round-tripping is broken.
void printDelegate(
    OpAsmPrinter &printer,
    Operation *delegate,
    ClosureTypeMapFn mapType = {});

/// Parses a closure.
///
/// This parser implements the following grammar:
///
/// ```
/// closure         ::= op-delegate | closure-region
/// op-delegate     ::= `{` op-name attr-dict loc `}` yield-types
/// closure-region  ::= capture-list `(` arg-list `)` yield-types $body
/// capture-list    ::= [ `[` capture { `,` capture } `]` ]
/// capture         ::= ssa-id `=` value-use [ `:` type ]
/// arg-list        ::= [ arg { `,` arg } ]
/// arg             ::= ssa-id [ `:` type ]
/// yield-types     ::= [ `->` type-or-type-list ]
/// ```
///
/// @param  [in]        parser          OpAsmParser.
/// @param  [in]        type            Expected delegate type.
/// @param  [out]       captureOperands Captured operands.
/// @param  [out]       captureTypes    Types of @p captureOperands
/// @param  [out]       resultTypes     Result types.
/// @param  [out]       body            Closure body.
/// @param              yieldOp         Body yield operation, if any.
/// @param              mapType         An optional ClosureTypeMapFn.
///
/// @pre    `body.empty()`
///
/// @return ParseResult
ParseResult parseClosure(
    OpAsmParser &parser,
    FunctionType type,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captureOperands,
    SmallVectorImpl<Type> &captureTypes,
    SmallVectorImpl<Type> &resultTypes,
    Region &body,
    std::optional<OperationName> yieldOp,
    ClosureTypeMapFn mapType = {});

/// Prints a closure.
///
/// This printer implements the following grammar:
///
/// ```
/// closure         ::= op-delegate | closure-region
/// op-delegate     ::= `{` op-name attr-dict loc `}` yield-types
/// closure-region  ::= capture-list `(` arg-list `)` yield-types $body
/// capture-list    ::= [ `[` capture { `,` capture } `]` ]
/// capture         ::= ssa-id `=` value-use [ `:` type ]
/// arg-list        ::= [ arg { `,` arg } ]
/// arg             ::= ssa-id [ `:` type ]
/// yield-types     ::= [ `->` type-or-type-list ]
/// ```
///
/// @param  [in]        printer         OpAsmPrinter.
/// @param              inputTypes      Delegate input types.
/// @param              captureOperands Captured operands.
/// @param              resultTypes     Result types.
/// @param  [in]        body            Closure body.
/// @param              mapType         An optional ClosureTypeMapFn.
void printClosure(
    OpAsmPrinter &printer,
    TypeRange inputTypes,
    OperandRange captureOperands,
    TypeRange resultTypes,
    Region &body,
    ClosureTypeMapFn mapType = {});

} // namespace mlir::messner
