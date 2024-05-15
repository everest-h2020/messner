/// Implements the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

static ParseResult parseSymbolDeclaration(
    OpAsmParser &parser,
    StringAttr &sym_visibility,
    StringAttr &sym_name)
{
    // (`public` | `private` | `nested`)?
    std::string keyword;
    const auto loc = parser.getCurrentLocation();
    if (!parser.parseOptionalKeywordOrString(&keyword) && keyword != "public") {
        if (!llvm::is_contained({"private", "nested"}, keyword))
            return parser.emitError(loc, "expected public, private or nested");

        sym_visibility = parser.getBuilder().getStringAttr(keyword);
    }

    // $sym_name
    return parser.parseSymbolName(sym_name);
}

static void printSymbolDeclaration(
    OpAsmPrinter &printer,
    Operation *,
    StringAttr sym_visibility,
    StringAttr sym_name)
{
    // (`public` | `private` | `nested`)?
    if (sym_visibility && sym_visibility.getValue() != "public") {
        printer.printKeywordOrString(sym_visibility.getValue());
        printer << " ";
    }

    // $sym_name
    printer.printSymbolName(sym_name.getValue());
}

static ParseResult
parseKernelBody(OpAsmParser &parser, Region &body, NamedAttrList &attributes)
{
    // `(` [ block-arg { `,` block-arg } ] `)`
    SmallVector<OpAsmParser::Argument> arguments;
    if (parser
            .parseArgumentList(arguments, OpAsmParser::Delimiter::Paren, true))
        return failure();

    // Ensure that all argument types are wrapped in the ExpressionType.
    for (auto &arg : arguments)
        if (!llvm::isa<ExpressionType>(arg.type))
            arg.type = ExpressionType::get(parser.getContext(), arg.type);

    // attr-dict-with-keyword $body
    if (parser.parseOptionalAttrDictWithKeyword(attributes)
        || parser.parseRegion(body, arguments, true))
        return failure();

    // NOTE: Weirdly, the parser may leave the body block empty.
    if (arguments.empty() && body.empty()) body.emplaceBlock();
    return success();
}

static void printKernelBody(
    OpAsmPrinter &printer,
    Operation *,
    Region &body,
    DictionaryAttr attributes)
{
    // `(` [ block-arg { `,` block-arg } ] `)`
    printer << "(";
    llvm::interleaveComma(
        body.getArguments(),
        printer,
        [&](BlockArgument &arg) {
            // block-arg ::= ssa-id [ `:` type ] [ `loc` `(` loc `)` ]
            printer << arg << ": "
                    << llvm::cast<ExpressionType>(arg.getType()).getTypeBound();
            printer.printOptionalLocationSpecifier(arg.getLoc());
        });
    printer << ")";

    // attr-dict-with-keyword
    printer.printOptionalAttrDictWithKeyword(
        attributes.getValue(),
        {"sym_name"});

    // $body
    printer << " ";
    printer.printRegion(body, false);
}

static ParseResult parseOptionalExprOperand(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &operand,
    Type &type)
{
    operand.emplace();

    // [ value-use ... ]
    const auto maybe = parser.parseOptionalOperand(*operand);
    if (!maybe.has_value()) {
        operand.reset();
        return success();
    }
    if (maybe.value()) {
        operand.reset();
        return failure();
    }

    // [ `:` type ]
    Type bound = nullptr;
    if (!parser.parseOptionalColon()) {
        if (parser.parseType(bound)) return failure();
    }

    type = ExpressionType::get(parser.getContext(), bound);
    return success();
}

static ParseResult parseExprOperand(
    OpAsmParser &parser,
    OpAsmParser::UnresolvedOperand &operand,
    Type &type)
{
    std::optional<OpAsmParser::UnresolvedOperand> maybe;
    if (parseOptionalExprOperand(parser, maybe, type)) return failure();
    if (!maybe)
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected SSA operand");

    operand = *maybe;
    return success();
}

static void printExprOperand(
    OpAsmPrinter &printer,
    Operation *,
    Value operand,
    ExpressionType type)
{
    // value-use
    printer << operand;

    // [ `:` type ]
    if (!type.isUnbounded()) printer << ": " << type.getTypeBound();
}

static void printOptionalExprOperand(
    OpAsmPrinter &printer,
    Operation *op,
    Value operand,
    Type type)
{
    if (!operand) return;
    printExprOperand(printer, op, operand, llvm::cast<ExpressionType>(type));
}

static ParseResult parseExprOperand(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types)
{
    // [ expr-operand ... ]
    std::optional<OpAsmParser::UnresolvedOperand> operand;
    Type type;
    if (parseOptionalExprOperand(parser, operand, type)) return failure();
    if (!operand) return success();
    operands.push_back(*operand);
    types.push_back(type);

    // { `,` expr-operand }
    while (!parser.parseOptionalComma())
        if (parseExprOperand(
                parser,
                operands.emplace_back(),
                types.emplace_back()))
            return failure();

    return success();
}

static void printExprOperand(
    OpAsmPrinter &printer,
    Operation *op,
    OperandRange operands,
    TypeRange types)
{
    const auto printOperand = [&]() {
        printExprOperand(
            printer,
            op,
            operands.front(),
            llvm::cast<ExpressionType>(types.front()));
        operands = operands.drop_front(1);
        types    = types.drop_front(1);
    };

    // [ expr-operand ... ]
    if (operands.empty()) return;
    printOperand();

    // { `,` expr-operand }
    while (!operands.empty()) {
        printer << ", ";
        printOperand();
    }
}

static ParseResult parseExprResult(OpAsmParser &parser, Type &result)
{
    // [ `->` type ]
    Type bound = nullptr;
    if (!parser.parseOptionalArrow()) {
        if (parser.parseType(bound)) return failure();
    }

    result = ExpressionType::get(parser.getContext(), bound);
    return success();
}

static void
printExprResult(OpAsmPrinter &printer, Operation *, ExpressionType result)
{
    // [ `->` type ]
    if (!result.isUnbounded()) printer << "-> " << result.getTypeBound();
}

static ParseResult parseFunctor(OpAsmParser &parser, Region &body)
{
    SmallVector<OpAsmParser::Argument> arguments;

    // [ `(` ... { `,` ... } `)` ]
    if (parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::OptionalParen,
            [&]() -> ParseResult {
                // `ssa-id`
                if (parser.parseArgument(arguments.emplace_back()))
                    return failure();

                // [ `:` type ]
                Type argTy{};
                if (!parser.parseOptionalColon()) {
                    if (parser.parseType(argTy)) return failure();
                }
                arguments.back().type =
                    ExpressionType::get(parser.getContext(), argTy);

                // [ `loc` `(` loc `)` ]
                return parser.parseOptionalLocationSpecifier(
                    arguments.back().sourceLoc);
            }))
        return failure();

    // $body
    return parser.parseRegion(body, arguments);
}

static void printFunctor(OpAsmPrinter &printer, Operation *, Region &body)
{
    // [ `(` ... { `,` ... } `)` ]
    if (body.getNumArguments() > 0) {
        printer << "(";
        llvm::interleaveComma(
            body.getArguments(),
            printer,
            [&](BlockArgument arg) {
                // ssa-id
                printer << arg;

                // [ `:` type ]
                const auto exprTy = llvm::cast<ExpressionType>(arg.getType());
                if (!exprTy.isUnbounded())
                    printer << ": " << exprTy.getTypeBound();
            });
        printer << ") ";
    }

    // $body
    printer.printRegion(body, false);
}

static ParseResult
parseElseBranch(OpAsmParser &parser, Region &body, Type &resultType)
{
    if (parser.parseOptionalKeyword("else")) return success();

    if (failed(parseFunctor(parser, body))) return failure();

    if (!parser.parseOptionalArrow()) {
        if (parser.parseType(resultType)) return failure();
    }

    return success();
}

static void printElseBranch(
    OpAsmPrinter &printer,
    Operation *op,
    Region &body,
    Type resultType)
{
    if (!resultType && body.empty()) return;

    printer << "else ";

    printFunctor(printer, op, body);

    if (resultType) printer << " -> " << resultType;
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EvalOp implementation
//===----------------------------------------------------------------------===//

LogicalResult EvalOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type resultTy;
    return adaptor.require(getInput(), getType(), resultTy);
}

OpFoldResult EvalOp::fold(EvalOp::FoldAdaptor adaptor)
{
    // NOTE: Attributes folded in this way can't actually be materialized.
    return adaptor.getInput();
}

//===----------------------------------------------------------------------===//
// DefineOp implementation
//===----------------------------------------------------------------------===//

LogicalResult DefineOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location>,
    ValueRange operands,
    DictionaryAttr,
    OpaqueProperties,
    RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes)
{
    inferredReturnTypes.push_back(
        ExpressionType::get(context, operands.front().getType()));
    return success();
}

OpFoldResult DefineOp::fold(DefineOp::FoldAdaptor adaptor)
{
    return adaptor.getInput();
}

//===----------------------------------------------------------------------===//
// ConstexprOp implementation
//===----------------------------------------------------------------------===//

LogicalResult ConstexprOp::verifyRegions()
{
    // Check that all children are pure.
    for (auto &op : getExpression().getOps())
        if (!isPure(&op))
            return op.emitOpError("is impure").attachNote(getLoc())
                << "required by this operation";

    return success();
}

OpFoldResult ConstexprOp::fold(ConstexprOp::FoldAdaptor)
{
    // Get the yielded value.
    auto yieldOp     = llvm::cast<YieldOp>(&getExpression().front().back());
    const auto yield = llvm::cast<Expression>(yieldOp.getValue());

    // Fold to that value if it is constant.
    LiteralAttr literal;
    if (matchPattern(yield, m_Constant(&literal))) return literal;

    return {};
}

LogicalResult ConstexprOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Get the yielded type.
    auto yieldOp       = llvm::cast<YieldOp>(&getExpression().front().back());
    const auto yield   = llvm::cast<Expression>(yieldOp.getValue());
    const auto yieldTy = adaptor.getType(yield);
    if (!yieldTy) return success();

    return adaptor.refineBound(getResult(), yieldTy);
}

//===----------------------------------------------------------------------===//
// GlobalOp implementation
//===----------------------------------------------------------------------===//

void GlobalOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    ABIReferenceType type,
    InitializerAttr initializer,
    SymbolTable::Visibility visibility)
{
    state.addAttribute(
        getSymNameAttrName(state.name),
        builder.getStringAttr(name));
    state.addAttribute(getTypeAttrName(state.name), TypeAttr::get(type));

    if (initializer)
        state.addAttribute(getInitializerAttrName(state.name), initializer);

    switch (visibility) {
    case SymbolTable::Visibility::Nested:
        state.addAttribute(
            getSymVisibilityAttrName(state.name),
            builder.getStringAttr("nested"));
        break;
    case SymbolTable::Visibility::Private:
        state.addAttribute(
            getSymVisibilityAttrName(state.name),
            builder.getStringAttr("private"));
        break;
    default: break;
    }
}

LogicalResult GlobalOp::verify()
{
    if (const auto initAttr = getInitializer()) {
        const auto initTy = initAttr->getType();
        if (initTy != getType().getArrayType())
            return emitOpError("expected initializer of type ")
                << getType().getArrayType() << " but got " << initTy;
    }

    return success();
}

LogicalResult GlobalOp::canonicalize(GlobalOp op, PatternRewriter &rewriter)
{
    // Initializers on non-readable non-public symbols are useless.
    if (op.getVisibility() != SymbolTable::Visibility::Public
        && !op.getType().isReadable() && op.getInitializer()) {
        rewriter.updateRootInPlace(op, [&]() { op.removeInitializerAttr(); });
        return success();
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name)
{
    state.addRegion()->emplaceBlock();
    state.addAttribute(
        getSymNameAttrName(state.name),
        builder.getStringAttr(name));
}

LogicalResult KernelOp::verify()
{
    for (auto arg : getBody().getArguments()) {
        const auto exprTy = llvm::dyn_cast<ExpressionType>(arg.getType());
        if (exprTy && llvm::isa_and_present<ABIType>(exprTy.getTypeBound()))
            continue;

        return emitOpError("type ")
            << arg.getType() << " of argument #" << arg.getArgNumber()
            << " is not an ABI-compatible expression type";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ReadOp implementation
//===----------------------------------------------------------------------===//

LogicalResult ReadOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The reference operand must be a readable reference.
    ReadableRefType refTy;
    if (auto contra =
            adaptor.require(getReference(), refTy, "readable reference"))
        return contra;

    // Result type is the referenced array type, decaying to a scalar.
    return adaptor.refineBound(
        getResult(),
        decayToScalar(refTy.getArrayType()));
}

//===----------------------------------------------------------------------===//
// WriteOp implementation
//===----------------------------------------------------------------------===//

LogicalResult WriteOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The reference operand must be a writable reference.
    WritableRefType refTy;
    if (auto contra =
            adaptor.require(getReference(), refTy, "writable reference"))
        return contra;

    // The second operand must be a value that is assignable to that reference.
    Type valueTy;
    if (auto contra =
            adaptor.require(getValue(), refTy.getArrayType(), valueTy))
        return contra;

    // There are no results to this operation.
    return success();
}

//===----------------------------------------------------------------------===//
// IfOp implementation
//===----------------------------------------------------------------------===//

static LogicalResult verifyFunctor(Operation *op, Region &functor)
{
    if (functor.empty()) return success();

    for (auto &&[idx, arg] : llvm::enumerate(functor.getArguments()))
        if (!llvm::isa<ExpressionType>(arg.getType()))
            return op->emitOpError()
                << "argument #" << idx << " of region #"
                << functor.getRegionNumber() << " must be an expression type";

    return success();
}

LogicalResult IfOp::verify()
{
    if (getThenBranch().getNumArguments() != 0)
        return emitOpError() << "no arguments allowed in then branch";
    if (getElseBranch().getNumArguments() != 0)
        return emitOpError() << "no arguments allowed in else branch";

    if (failed(verifyFunctor(*this, getThenBranch()))
        || failed(verifyFunctor(*this, getElseBranch())))
        return failure();

    if (!getResult()) return success();

    if (getElseBranch().empty())
        return emitOpError() << "else branch is required";
    auto yield = llvm::cast<YieldOp>(getElseBranch().front().back());
    if (!yield.getValue())
        return yield.emitOpError("operand is required").attachNote(getLoc())
            << "required by this operation";

    return success();
}

LogicalResult IfOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // We need to ensure that the closures are in a verified state.
    if (failed(verify())) return failure();

    // The condition must be a boolean.
    BoolType boolTy;
    if (auto contra = adaptor.require(getCondition(), boolTy, "bool"))
        return contra;

    // Handle the statement case.
    if (!getResult()) return success();

    // Unify the types of both branch expressions.
    const auto trueVal =
        llvm::cast<YieldOp>(&getThenBranch().front().back()).getValue();
    const auto falseVal =
        llvm::cast<YieldOp>(&getElseBranch().front().back()).getValue();
    Type unifiedTy;
    if (auto contra = adaptor.unify(ValueRange({trueVal, falseVal}), unifiedTy))
        return contra;

    return adaptor.refineBound(llvm::cast<Expression>(getResult()), unifiedTy);
}

//===----------------------------------------------------------------------===//
// PanicOp implementation
//===----------------------------------------------------------------------===//

// TODO: PanicOp.

//===----------------------------------------------------------------------===//
// UnifyOp implementation
//===----------------------------------------------------------------------===//

[[nodiscard]] static ScalarAttr unify(NumberAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](NumberType) { return input; })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ScalarAttr unify(ekl::IntegerAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](NumberType) {
            return NumberAttr::get(
                input.getContext(),
                Number(input.getValue()));
        })
        .Case([&](ekl::IntegerType type) {
            if (type == input.getType()) return input;
            return llvm::cast<ekl::IntegerAttr>(
                mlir::IntegerAttr::get(type, input.getValue()));
        })
        .Case([&](FloatType type) {
            llvm::APFloat value(type.getFloatSemantics());
            value.convertFromAPInt(
                input.getValue(),
                input.getType().isSigned(),
                llvm::APFloat::roundingMode::NearestTiesToEven);
            return FloatAttr::get(type, value);
        })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ScalarAttr unify(FloatAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](NumberType) {
            return NumberAttr::get(
                input.getContext(),
                Number(input.getValue()));
        })
        .Case([&](FloatType type) {
            if (type == input.getType()) return input;
            auto value = input.getValue();
            bool losesInfo;
            value.convert(
                type.getFloatSemantics(),
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &losesInfo);
            return FloatAttr::get(type, value);
        })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ScalarAttr unify(ekl::IndexAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](NumberType) {
            return NumberAttr::get(
                input.getContext(),
                Number(input.getValue()));
        })
        .Case([&](ekl::IntegerType type) {
            return llvm::cast<ekl::IntegerAttr>(mlir::IntegerAttr::get(
                type,
                llvm::APInt(64U, input.getValue())));
        })
        .Case([&](FloatType type) {
            llvm::APFloat value(type.getFloatSemantics());
            value.convertFromAPInt(
                llvm::APInt(64U, input.getValue()),
                false,
                llvm::APFloat::roundingMode::NearestTiesToEven);
            return FloatAttr::get(type, value);
        })
        .Case([&](ekl::IndexType) { return input; })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ScalarAttr unify(BoolAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](BoolType) { return input; })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ScalarAttr unify(ScalarAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarAttr, ScalarAttr>(input)
        .Case([&](NumberAttr attr) { return unify(attr, output); })
        .Case([&](ekl::IntegerAttr attr) { return unify(attr, output); })
        .Case([&](FloatAttr attr) { return unify(attr, output); })
        .Case([&](ekl::IndexAttr attr) { return unify(attr, output); })
        .Case([&](BoolAttr attr) { return unify(attr, output); })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ekl::ArrayAttr unify(LiteralAttr input, ArrayType output)
{
    return llvm::TypeSwitch<LiteralAttr, ekl::ArrayAttr>(input)
        .Case([&](ScalarAttr attr) {
            return ekl::ArrayAttr::get(output, {attr});
        })
        .Case([&](ekl::ArrayAttr attr) {
            return ekl::ArrayAttr::get(output, attr.getFlattened());
        })
        .Default(ekl::ArrayAttr{});
}

[[nodiscard]] static LiteralAttr unify(LiteralAttr input, LiteralType output)
{
    return llvm::TypeSwitch<LiteralType, LiteralAttr>(output)
        .Case([&](ScalarType type) -> LiteralAttr {
            if (const auto attr = llvm::dyn_cast<ScalarAttr>(input))
                return unify(attr, type);
            return {};
        })
        .Case([&](ArrayType type) { return unify(input, type); })
        .Default(LiteralAttr{});
}

OpFoldResult UnifyOp::fold(UnifyOp::FoldAdaptor adaptor)
{
    const auto output =
        llvm::dyn_cast_if_present<LiteralType>(getTypeBound(getType()));
    const auto input =
        llvm::dyn_cast_if_present<LiteralAttr>(adaptor.getInput());
    if (output && input) return ::unify(input, output);
    return {};
}

LogicalResult UnifyOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type resultTy;
    return adaptor.require(getInput(), getType().getTypeBound(), resultTy);
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

void BroadcastOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value input,
    ExtentRange extents,
    Type resultBound)
{
    const auto extentsAttr = DenseI64ArrayAttr::get(
        builder.getContext(),
        ArrayRef<int64_t>(
            reinterpret_cast<const int64_t *>(extents.data()),
            extents.size()));

    state.addOperands({input});
    state.addAttribute(getExtentsAttrName(state.name), extentsAttr);
    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});
}

LogicalResult BroadcastOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // TODO: Find a safer way to do this.
    const auto extents = ExtentRange(
        std::bit_cast<const extent_t *>(getExtents().data()),
        getExtents().size());

    ArrayType resultTy;
    if (auto contra = adaptor.broadcast(getInput(), extents, resultTy))
        return contra;

    return adaptor.refineBound(getResult(), resultTy);
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

LogicalResult CoerceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    return adaptor.coerce(getInput(), getType());
}

//===----------------------------------------------------------------------===//
// LiteralOp implementation
//===----------------------------------------------------------------------===//

void LiteralOp::build(
    OpBuilder &builder,
    OperationState &state,
    LiteralAttr value)
{
    state.addTypes(
        {ExpressionType::get(builder.getContext(), value.getType())});
    state.addAttribute(getLiteralAttrName(state.name), value);
}

LogicalResult LiteralOp::verify()
{
    if (getType().getTypeBound() != getLiteral().getType())
        return emitOpError() << "expected " << getLiteral().getType()
                             << ", but got " << getType().getTypeBound();

    return success();
}

OpFoldResult LiteralOp::fold(LiteralOp::FoldAdaptor) { return getLiteral(); }

LogicalResult LiteralOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location>,
    ValueRange,
    DictionaryAttr attributes,
    OpaqueProperties,
    RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes)
{
    const auto literal = attributes.getAs<LiteralAttr>(getAttributeNames()[0]);
    if (!literal) return failure();

    inferredReturnTypes.push_back(
        ExpressionType::get(context, literal.getType()));
    return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp implementation
//===----------------------------------------------------------------------===//

void GetGlobalOp::build(
    OpBuilder &builder,
    OperationState &state,
    GlobalOp global)
{
    state.addTypes(
        {ExpressionType::get(builder.getContext(), global.getType())});
    state.addAttribute(
        getGlobalNameAttrName(state.name),
        FlatSymbolRefAttr::get(global.getNameAttr()));
}

LogicalResult GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    auto targetOp =
        symbolTable.lookupNearestSymbolFrom(*this, getGlobalNameAttr());
    auto globalOp = llvm::dyn_cast_if_present<GlobalOp>(targetOp);
    if (!globalOp) {
        auto diag = emitOpError("'")
                 << getGlobalName() << "' does not reference a global symbol";
        if (targetOp)
            diag.attachNote(targetOp->getLoc())
                << "references this declaration";

        return diag;
    }

    if (!isSubtype(getType().getTypeBound(), globalOp.getType())) {
        auto diag = emitOpError()
                 << getType().getTypeBound() << " is not a subtype of "
                 << globalOp.getType();
        diag.attachNote(targetOp->getLoc()) << "symbol declared here";
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// SubscriptOp implementation
//===----------------------------------------------------------------------===//

[[nodiscard]] Operation *getOwner(Value value)
{
    if (const auto argument = llvm::dyn_cast<BlockArgument>(value))
        return argument.getOwner()->getParentOp();
    return llvm::cast<OpResult>(value).getOwner();
}

[[nodiscard]] static bool isIndexArg(Value value)
{
    assert(value);

    // Must be a block argument.
    const auto argument = llvm::dyn_cast<BlockArgument>(value);
    if (!argument) return false;

    // Must be from the map region of an AssocOp or ReduceOp.
    const auto owner = argument.getOwner()->getParentOp();
    if (!owner) return false;
    if (llvm::isa<AssocOp>(owner)) return true;
    if (llvm::isa<ReduceOp>(owner))
        return argument.getOwner()->getParent()->getRegionNumber() == 0;

    return false;
}

static FailureOr<ekl::IndexType> meetIndexBound(
    AbstractTypeChecker &typeChecker,
    Expression expr,
    uint64_t bound)
{
    // Update the bound on the index value, which will fail if there is already
    // a different bound.
    const auto type = ekl::IndexType::get(expr.getContext(), bound);
    if (failed(typeChecker.meetBound(expr, type))) return failure();

    // This invalidates the owner as well.
    typeChecker.invalidate(getOwner(expr));
    return type;
}

static Contradiction typeCheckSubscripts(
    TypeCheckingAdaptor &adaptor,
    ValueRange subscripts,
    SmallVectorImpl<Type> &bounds)
{
    // Check the known types of all subscript operands.
    auto unbounded = false;
    std::optional<Value> ellipsis;
    for (auto subscript : subscripts) {
        auto bound = adaptor.getType(llvm::cast<Expression>(subscript));
        bounds.push_back(bound);

        if (!bound) {
            if (isIndexArg(subscript)) {
                // Will be inferred later.
                continue;
            }

            // Definitely stays unbounded.
            unbounded = true;
            continue;
        }

        if (llvm::isa<EllipsisType>(bound)) {
            // There may only be a single ellipsis.
            if (ellipsis) {
                auto diag = adaptor.emitError() << "more than one ellipsis";
                diag.attachNote(subscript.getLoc()) << "found here";
                diag.attachNote(ellipsis->getLoc())
                    << "previous ellipsis was here";
                return diag;
            }
            ellipsis = subscript;
            continue;
        }

        if (llvm::isa<ExtentType, IdentityType>(bound)) continue;
        if (llvm::isa_and_present<ekl::IndexType>(getScalarType(bound)))
            continue;

        // Type is not a valid indexer.
        auto diag = adaptor.emitError()
                 << "expected indexer, but got " << bound;
        diag.attachNote(subscript.getLoc()) << "for this subscript";
        return diag;
    }

    if (ellipsis && unbounded) {
        // We can't resolve the ellipsis yet.
        return Contradiction::indeterminate();
    }

    return Contradiction::none();
}

LogicalResult SubscriptOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Array operand must be an array or reference type.
    ContiguousType arrayTy;
    if (auto contra =
            adaptor.require(getArray(), arrayTy, "array or reference"))
        return contra;

    // Type check the subscripts first, to allow ellipsis resolving.
    SmallVector<Type> subscriptTys;
    if (auto contra =
            typeCheckSubscripts(adaptor, getSubscripts(), subscriptTys))
        return contra;

    // Infer the result extents.
    auto sourceDim = 0U;
    SmallVector<uint64_t> extents;
    for (auto [idx, value] : llvm::enumerate(getSubscripts())) {
        auto bound = subscriptTys[idx];
        if (!bound) {
            assert(isIndexArg(value));

            // The subscript type checker let this through because we can infer
            // it from the array extents.
            const auto meet = meetIndexBound(
                typeChecker,
                llvm::cast<Expression>(value),
                arrayTy.getExtent(sourceDim) - 1UL);
            if (failed(meet)) return failure();
            bound = *meet;
            assert(bound);
        }
        if (llvm::isa<ExtentType>(bound)) {
            // We insert a new unit dimension.
            extents.push_back(1UL);
            continue;
        }
        if (llvm::isa<EllipsisType>(bound)) {
            // Count the number of remaining subscripts that will bind to a
            // source dimension.
            const auto remaining = static_cast<size_t>(llvm::count_if(
                ArrayRef<Type>(subscriptTys).drop_front(idx + 1),
                [](Type type) {
                    return !type || !llvm::isa<ExtentType>(type);
                }));
            // Insert the identity indexer until we bind enough dimensions.
            while (remaining < (arrayTy.getNumExtents() - sourceDim))
                extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        // For all other kinds of subscripts, we bind 1 source dimension.
        if (sourceDim == arrayTy.getNumExtents()) {
            auto diag = emitError() << "exceeded number of array extents ("
                                    << arrayTy.getNumExtents() << ")";
            diag.attachNote(value.getLoc()) << "with this subscript";
            return diag;
        }
        if (llvm::isa<IdentityType>(bound)) {
            // We map this dimension using the identity.
            extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        const auto indexTy = llvm::cast<ekl::IndexType>(getScalarType(bound));

        // Handle statically known index bounds.
        if (!indexTy.isUnbounded()
            && indexTy.getUpperBound() >= arrayTy.getExtent(sourceDim)) {
            auto diag = emitOpError()
                     << "index out of bounds (" << indexTy.getUpperBound()
                     << " >= " << arrayTy.getExtent(sourceDim) << ")";
            diag.attachNote(value.getLoc()) << "for this subscript";
            return diag;
        }

        // Insert the indexer's extents here, and skip this dimension in the
        // source.
        ++sourceDim;
        concat(extents, getExtents(bound).value_or(ExtentRange{}));
    }

    // For partial subscripting, we append all the remaining dimensions.
    concat(extents, arrayTy.getExtents().drop_front(sourceDim));

    // The result is an array with the inferred extents, decaying to a scalar.
    return adaptor.refineBound(
        getResult(),
        decayToScalar(applyReference(
            arrayTy,
            ArrayType::get(
                llvm::cast<ScalarType>(arrayTy.getScalarType()),
                extents))));
}

//===----------------------------------------------------------------------===//
// YieldOp implementation
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    // Propagate this event to the parent.
    if (const auto parent = (*this)->getParentOp())
        typeChecker.invalidate(parent);

    return success();
}

//===----------------------------------------------------------------------===//
// StackOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult StackOp::fold(StackOp::FoldAdaptor adaptor)
{
    const auto arrayTy =
        llvm::dyn_cast_if_present<ArrayType>(getTypeBound(getType()));
    if (!arrayTy) return {};

    SmallVector<Attribute> flattened;
    for (auto op : adaptor.getOperands()) {
        if (const auto attr = llvm::dyn_cast_if_present<ScalarAttr>(op)) {
            flattened.push_back(attr);
            continue;
        }
        if (const auto attr = llvm::dyn_cast_if_present<ekl::ArrayAttr>(op)) {
            if (attr.isSplat())
                flattened.append(
                    *attr.getType().getNumElements(),
                    attr.getSplatValue());
            else
                flattened.append(
                    attr.getFlattened().begin(),
                    attr.getFlattened().end());
            continue;
        }
        return {};
    }

    return ekl::ArrayAttr::get(arrayTy, flattened);
}

LogicalResult StackOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The operands must all unify or broadcast together.
    Type unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(getOperands(), unifiedTy))
        return contra;

    // Its easier to just decay pseudo-scalar arrays now.
    unifiedTy = decayToScalar(unifiedTy);

    // The unified type must have a scalar type.
    const auto scalarTy = getScalarType(unifiedTy);
    if (!scalarTy)
        return emitError()
            << "can't stack values of non-scalar type " << unifiedTy;

    // The result extents are determined by the unified extents, prepended by
    // the number of stacked atoms.
    const auto resultExtents = concat(
        {static_cast<uint64_t>(getOperands().size())},
        getExtents(unifiedTy).value_or(ExtentRange{}));
    return adaptor.refineBound(
        getResult(),
        ArrayType::get(scalarTy, resultExtents));
}

//===----------------------------------------------------------------------===//
// AssocOp implementation
//===----------------------------------------------------------------------===//

static LogicalResult typeCheckMap(
    TypeCheckingAdaptor &adaptor,
    Region &map,
    Type &yieldTy,
    SmallVectorImpl<extent_t> &extents,
    bool &validExtents)
{
    // Ensure that all the arguments are indices and infer the extents.
    validExtents = true;
    for (auto arg : map.getArguments()) {
        ekl::IndexType indexTy;
        if (failed(
                adaptor.require(llvm::cast<Expression>(arg), indexTy, "index")))
            return failure();
        if (!indexTy || indexTy.isUnbounded())
            validExtents = false;
        else
            extents.push_back(indexTy.getUpperBound() + 1UL);
    }

    // Obtain the yielded value type.
    auto yield = llvm::cast<YieldOp>(&map.front().back());
    yieldTy    = adaptor.getType(llvm::cast<Expression>(yield.getValue()));
    return success();
}

LogicalResult AssocOp::verify() { return verifyFunctor(*this, getMap()); }

OpFoldResult AssocOp::fold(AssocOp::FoldAdaptor)
{
    // Can't fold untyped operation.
    if (!getType().getTypeBound()) return {};

    // See if the yielded value was folded.
    auto yield      = llvm::cast<YieldOp>(getMap().front().back());
    const auto expr = llvm::cast<Expression>(yield.getValue());
    Attribute value;
    if (!matchPattern(expr, m_Constant(&value))) return {};

    // If the extents are empty, we can return anything.
    if (getMap().getNumArguments() == 0) return value;

    // Otherwise, we must have yielded a scalar, which we can make a splat of.
    const auto splatValue = llvm::cast<ScalarAttr>(value);
    return ekl::ArrayAttr::get(
        llvm::cast<ArrayType>(getType().getTypeBound()),
        {splatValue});
}

LogicalResult AssocOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Type-check the map functor.
    Type yieldTy;
    SmallVector<extent_t> extents;
    bool validExtents;
    if (failed(typeCheckMap(adaptor, getMap(), yieldTy, extents, validExtents)))
        return failure();

    // Only if all of the above is known can we deduce the result type.
    if (!validExtents || !yieldTy) return success();

    // Refine the bound on the result.
    if (extents.empty()) return adaptor.refineBound(getResult(), yieldTy);

    const auto yieldExtents = getExtents(yieldTy);
    if (failed(yieldExtents)) {
        auto diag = emitError() << "expected scalar or array value";
        diag.attachNote(getMap().front().back().getOperand(0).getLoc())
            << "for this value";
        return diag;
    }
    concat(extents, *yieldExtents);
    return adaptor.refineBound(
        getResult(),
        ArrayType::get(getScalarType(yieldTy), extents));
}

static constexpr StringRef indexLetters = "ijklmnopqrstuvwxyz";

void AssocOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setName)
{
    for (auto &&[arg, name] : llvm::zip(region.getArguments(), indexLetters))
        setName(arg, StringRef(&name, 1));
}

//===----------------------------------------------------------------------===//
// ZipOp implementation
//===----------------------------------------------------------------------===//

void ZipOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange operands,
    Type resultBound)
{
    auto &combinator = state.addRegion()->emplaceBlock();

    state.addOperands(operands);
    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});

    SmallVector<Type> types(
        operands.size(),
        ExpressionType::get(builder.getContext()));
    SmallVector<Location> locs(operands.size(), builder.getUnknownLoc());
    combinator.addArguments(types, locs);
}

void ZipOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange operands,
    Location combineLoc,
    OperationName combineOp,
    Type resultBound)
{
    build(builder, state, operands, resultBound);
    auto &combinator = state.regions.front()->front();

    // Insert the combinator op, assuming that it is a regular ExpressionOp.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&combinator);
    const auto result = builder
                            .create(
                                combineLoc,
                                combineOp.getIdentifier(),
                                combinator.getArguments(),
                                {ExpressionType::get(builder.getContext())})
                            ->getResult(0);
    builder.create<YieldOp>(combineLoc, result);
}

LogicalResult ZipOp::verify()
{
    if (getCombinator().getNumArguments() != getNumOperands())
        return emitOpError() << "expected " << getNumOperands()
                             << " arguments to combinator block";

    return verifyFunctor(*this, getCombinator());
}

LogicalResult ZipOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The operands must broadcast together.
    SmallVector<Type> operandTys;
    if (auto contra = adaptor.broadcast(getOperands(), operandTys))
        return contra;

    // Refine the bounds on the combinator.
    for (auto [arg, ty] :
         llvm::zip_equal(getCombinator().getArguments(), operandTys)) {
        if (failed(adaptor.refineBound(
                llvm::cast<Expression>(arg),
                getScalarType(ty))))
            return failure();
    }

    // Get the value type that the combinator yields.
    auto yield = llvm::cast<Expression>(
        llvm::cast<YieldOp>(&getCombinator().front().back()).getValue());

    // If there are no operands, we can produce anything.
    if (getNumOperands() == 0) {
        const auto resultTy = adaptor.getType(yield);
        if (!resultTy) return success();
        return adaptor.refineBound(getResult(), resultTy);
    }

    // Otherwise, we must create an array.
    ScalarType scalarTy;
    if (auto contra = adaptor.require(yield, scalarTy, "scalar")) return contra;
    return adaptor.refineBound(
        getResult(),
        ArrayType::get(scalarTy, *getExtents(operandTys.front())));
}

//===----------------------------------------------------------------------===//
// ReduceOp implementation
//===----------------------------------------------------------------------===//

void ReduceOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type resultBound)
{
    state.addRegion()->emplaceBlock();
    auto &reduction = state.addRegion()->emplaceBlock();

    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});

    const auto exprTy = ExpressionType::get(builder.getContext());
    const auto loc    = builder.getUnknownLoc();
    reduction.addArguments({exprTy, exprTy}, {loc, loc});
}

void ReduceOp::build(
    OpBuilder &builder,
    OperationState &state,
    Location reduceLoc,
    OperationName reduceOp,
    Type resultBound)
{
    build(builder, state, resultBound);
    auto &reduction = state.regions.back()->front();

    // Insert the reduction op, assuming that it is a regular ExpressionOp.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&reduction);
    const auto result = builder
                            .create(
                                reduceLoc,
                                reduceOp.getIdentifier(),
                                reduction.getArguments(),
                                {ExpressionType::get(builder.getContext())})
                            ->getResult(0);
    builder.create<YieldOp>(reduceLoc, result);
}

LogicalResult ReduceOp::verify()
{
    if (getReduction().getNumArguments() != 2)
        return emitOpError() << "expected 2 arguments to reduction block";

    if (failed(verifyFunctor(*this, getMap()))
        || failed(verifyFunctor(*this, getReduction())))
        return failure();

    return success();
}

LogicalResult ReduceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Type-check the map functor.
    ScalarType scalarTy;
    SmallVector<extent_t> extents;
    bool validExtents;
    if (failed(
            typeCheckMap(adaptor, getMap(), scalarTy, extents, validExtents)))
        return failure();
    if (!scalarTy) return success();

    // Refine the bounds on the reduction functor.
    if (failed(adaptor.refineBound(
            llvm::cast<Expression>(getReduction().getArgument(0)),
            scalarTy)))
        return failure();
    if (failed(adaptor.refineBound(
            llvm::cast<Expression>(getReduction().getArgument(1)),
            scalarTy)))
        return failure();

    // Check that the reduction functor yields a compatible scalar type.
    ScalarType resultTy;
    auto yield = llvm::cast<YieldOp>(&getReduction().front().back());
    if (auto contra = adaptor.require(
            llvm::cast<Expression>(yield.getValue()),
            scalarTy,
            resultTy))
        return contra;

    // The result does not depend on the extents.
    return adaptor.refineBound(getResult(), resultTy);
}

static constexpr StringRef reductionLetters = "abcdefgh";

void ReduceOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setName)
{
    if (region.getRegionNumber() == 0) {
        for (auto &&[arg, name] :
             llvm::zip(region.getArguments(), reductionLetters))
            setName(arg, StringRef(&name, 1));
        return;
    }

    assert(region.getRegionNumber() == 1);
    setName(region.getArgument(0), "lhs");
    setName(region.getArgument(1), "rhs");
}

//===----------------------------------------------------------------------===//
// ChoiceOp implementation
//===----------------------------------------------------------------------===//

LogicalResult ChoiceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(
            {getTrueValue(), getFalseValue()},
            unifiedTy))
        return contra;

    ArrayType resultTy;
    if (auto contra =
            adaptor.broadcast(getCondition(), *getExtents(unifiedTy), resultTy))
        return contra;

    // The condition must be bool.
    if (!llvm::isa<BoolType>(resultTy.getScalarType())) {
        auto diag = emitError()
                 << "expected bool, got " << resultTy.getScalarType();
        diag.attachNote(getCondition().getLoc()) << "for this value";
        return diag;
    }

    // The result of the choice uses the unified scalar type.
    resultTy = resultTy.cloneWith(getScalarType(unifiedTy));
    // Get rid of pseudo-scalar arrays now.
    return adaptor.refineBound(getResult(), decayToScalar(resultTy));
}

//===----------------------------------------------------------------------===//
// CompareOp implementation
//===----------------------------------------------------------------------===//

void CompareOp::build(
    OpBuilder &builder,
    OperationState &state,
    RelationKind kind,
    Value lhs,
    Value rhs,
    Type resultBound)
{
    state.addOperands({lhs, rhs});
    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});
    state.addAttribute(
        getKindAttrName(state.name),
        RelationKindAttr::get(builder.getContext(), kind));
}

LogicalResult CompareOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(getOperands(), unifiedTy))
        return contra;

    // Unified scalar type must be a number (or bool for eq/ne).
    const auto scalarTy = getScalarType(unifiedTy);
    if (!llvm::isa<NumericType>(scalarTy)) {
        switch (getKind()) {
        case RelationKind::Equivalent:
        case RelationKind::Antivalent:
            if (llvm::isa<BoolType>(scalarTy)) {
                // Equivalence/Antivalence of booleans is also well-defined.
                break;
            }
            [[fallthrough]];

        default:
            return emitError() << "can't relate values of type " << unifiedTy;
        }
    }

    // The result is boolean with the unified extents.
    return adaptor.refineBound(
        getResult(),
        applyExtents(unifiedTy, BoolType::get(getContext())));
}

//===----------------------------------------------------------------------===//
// Logical operator implementation
//===----------------------------------------------------------------------===//

static LogicalResult typeCheckLogicalOp(TypeCheckingAdaptor &adaptor)
{
    // The operands must all unify or broadcast together.
    Type unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(
            adaptor.getParent()->getOperands(),
            unifiedTy))
        return contra;

    // Its easier to just decay pseudo-scalar arrays now.
    unifiedTy = decayToScalar(unifiedTy);

    // The unified scalar type must be bool.
    const auto scalarTy = getScalarType(unifiedTy);
    if (!llvm::isa_and_present<BoolType>(scalarTy))
        return adaptor.getParent()->emitError()
            << "expected bool, but got " << unifiedTy;

    // The result type is the unified type.
    return adaptor.refineBound(
        llvm::cast<Expression>(adaptor.getParent()->getResult(0)),
        unifiedTy);
}

LogicalResult LogicalNotOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckLogicalOp(adaptor);
}

LogicalResult LogicalOrOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckLogicalOp(adaptor);
}

LogicalResult LogicalAndOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckLogicalOp(adaptor);
}

//===----------------------------------------------------------------------===//
// Arithmetic operator implementation
//===----------------------------------------------------------------------===//

static LogicalResult typeCheckArithmeticOp(
    TypeCheckingAdaptor &adaptor,
    function_ref<uint64_t(ArrayRef<uint64_t>)> combineIndexBounds)
{
    // The operands must all unify or broadcast together.
    Type unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(
            adaptor.getParent()->getOperands(),
            unifiedTy))
        return contra;

    // Its easier to just decay pseudo-scalar arrays now.
    unifiedTy = decayToScalar(unifiedTy);

    // The unified scalar type must be numeric.
    const auto scalarTy = getScalarType(unifiedTy);
    if (!llvm::isa<NumericType>(scalarTy))
        return adaptor.getParent()->emitError()
            << "expected number, but got " << unifiedTy;

    // Arithmetic operations need to properly update the upper bounds on the
    // types of index values they produce.
    if (llvm::isa<ekl::IndexType>(scalarTy)) {
        const auto operandUpperBounds = llvm::to_vector(llvm::map_range(
            adaptor.getParent()->getOperands(),
            [&](Value operand) {
                return llvm::cast<ekl::IndexType>(
                           getScalarType(adaptor.getType(
                               llvm::cast<Expression>(operand))))
                    .getUpperBound();
            }));
        return adaptor.refineBound(
            llvm::cast<Expression>(adaptor.getParent()->getResult(0)),
            applyExtents(
                unifiedTy,
                ekl::IndexType::get(
                    scalarTy.getContext(),
                    combineIndexBounds(operandUpperBounds))));
    }

    // The result type is the unified type.
    return adaptor.refineBound(
        llvm::cast<Expression>(adaptor.getParent()->getResult(0)),
        unifiedTy);
}

LogicalResult NegateOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            return bounds[0] == 0 ? 0 : ekl::IndexType::kUnbounded;
        });
}

LogicalResult AddOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            return bounds[0] + bounds[1];
        });
}

LogicalResult SubtractOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t { return bounds[0]; });
}

LogicalResult MultiplyOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            return bounds[0] * bounds[1];
        });
}

LogicalResult DivideOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t { return bounds[0]; });
}

LogicalResult RemainderOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t { return bounds[1] - 1; });
}

LogicalResult PowerOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(adaptor, [](ArrayRef<uint64_t>) -> uint64_t {
        return ekl::IndexType::kUnbounded;
    });
}

//===----------------------------------------------------------------------===//
// MinOp implementation
//===----------------------------------------------------------------------===//

LogicalResult MinOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            return std::min(bounds[0], bounds[1]);
        });
}

//===----------------------------------------------------------------------===//
// MaxOp implementation
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);
    return typeCheckArithmeticOp(
        adaptor,
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            return std::max(bounds[0], bounds[1]);
        });
}

//===----------------------------------------------------------------------===//
// Tensor operator implementation
//===----------------------------------------------------------------------===//

LogicalResult TensorProductOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The inputs must be tensors that unify to the same scalar type.
    ekl::TensorType lhsTy;
    if (auto contra = adaptor.require(getLhs(), lhsTy, "tensor")) return contra;
    ekl::TensorType rhsTy;
    if (auto contra = adaptor.require(getRhs(), rhsTy, "tensor")) return contra;
    NumericType scalarTy;
    if (auto contra = adaptor.unify(
            {lhsTy.getScalarType(), rhsTy.getScalarType()},
            scalarTy))
        return contra;

    // The result extents are the concatenation of the operand extents.
    const auto resultExtents = concat(lhsTy.getExtents(), rhsTy.getExtents());

    if (llvm::isa<ekl::IndexType>(scalarTy)) {
        // Arithmetic on indices requires computing the upper bound.
        const auto resultUpperBound =
            llvm::cast<IndexType>(lhsTy.getScalarType()).getUpperBound()
            * llvm::cast<IndexType>(rhsTy.getScalarType()).getUpperBound();
        return adaptor.refineBound(
            getResult(),
            ArrayType::get(
                ekl::IndexType::get(scalarTy.getContext(), resultUpperBound)));
    }

    return adaptor.refineBound(
        getResult(),
        ArrayType::get(scalarTy, resultExtents));
}

//===----------------------------------------------------------------------===//
// EKLDialect
//===----------------------------------------------------------------------===//

void EKLDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"
        >();
}
