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
#include <bit>

using namespace mlir;
using namespace mlir::ekl;

std::function<void(OpBuilder &, Location, ValueRange)>
mlir::ekl::getFunctorBuilder(
    OperationName op,
    std::optional<Location> opLoc,
    Type resultBound,
    DictionaryAttr attributes)
{
    return [=](OpBuilder &builder, Location loc, ValueRange args) {
        const auto result =
            builder
                .create(
                    opLoc.value_or(loc),
                    op.getIdentifier(),
                    args,
                    {ExpressionType::get(builder.getContext(), resultBound)},
                    attributes ? attributes.getValue()
                               : DictionaryAttr::ValueType{})
                ->getResult(0);
        builder.create<YieldOp>(loc, result);
    };
}

//===----------------------------------------------------------------------===//
// Shared directives
//===----------------------------------------------------------------------===//

static ParseResult parseTypeBound(OpAsmParser &parser, Type &type)
{
    // [ `:` type ]
    type = Type{};
    if (!parser.parseOptionalColon() && failed(parser.parseType(type)))
        return failure();

    type = ExpressionType::get(parser.getContext(), type);
    return success();
}

static void printTypeBound(OpAsmPrinter &printer, Operation *, Type type)
{
    const auto bound = llvm::cast<ExpressionType>(type).getTypeBound();

    // [ `:` type ]
    if (bound) printer << ": " << bound;
}

static OptionalParseResult parseOptionalOperand(
    OpAsmParser &parser,
    OpAsmParser::UnresolvedOperand &operand,
    Type &type)
{
    // [ value-use ... ]
    const auto maybe = parser.parseOptionalOperand(operand);
    if (!maybe.has_value()) return maybe;
    if (failed(*maybe)) return failure();

    // [ `:` type ]
    return parseTypeBound(parser, type);
}

static ParseResult parseOperand(
    OpAsmParser &parser,
    OpAsmParser::UnresolvedOperand &operand,
    Type &type)
{
    const auto maybe = parseOptionalOperand(parser, operand, type);
    if (!maybe.has_value())
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected SSA operand");
    return *maybe;
}

static void
printOperand(OpAsmPrinter &printer, Operation *op, Value operand, Type type)
{
    // value-use
    printer << operand;

    // [ `:` type ]
    printTypeBound(printer, op, type);
}

static ParseResult parseOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types)
{
    // [ expr-operand ... ]
    {
        OpAsmParser::UnresolvedOperand operand;
        Type type;
        const auto maybe = parseOptionalOperand(parser, operand, type);
        if (!maybe.has_value()) return success();
        if (failed(*maybe)) return failure();
        operands.push_back(operand);
        types.push_back(type);
    }

    // { `,` expr-operand }
    while (!parser.parseOptionalComma())
        if (parseOperand(parser, operands.emplace_back(), types.emplace_back()))
            return failure();

    return success();
}

static void printOperands(
    OpAsmPrinter &printer,
    Operation *op,
    OperandRange operands,
    TypeRange types)
{
    // [ expr-operand { `,` expr-operand } ]
    auto typeIt = types.begin();
    llvm::interleaveComma(operands, printer, [&](Value operand) {
        printOperand(printer, op, operand, *typeIt++);
    });
}

static ParseResult parseResult(OpAsmParser &parser, Type &result)
{
    // [ `->` type ]
    Type bound = nullptr;
    if (!parser.parseOptionalArrow()) {
        if (parser.parseType(bound)) return failure();
    }

    result = ExpressionType::get(parser.getContext(), bound);
    return success();
}

static void printResult(OpAsmPrinter &printer, Operation *, Type result)
{
    const auto bound = llvm::cast<ExpressionType>(result).getTypeBound();

    // [ `->` type ]
    if (bound) printer << "-> " << bound;
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
                if (failed(parseTypeBound(parser, arguments.back().type)))
                    return failure();

                // [ `loc` `(` attr `)` ]
                return parser.parseOptionalLocationSpecifier(
                    arguments.back().sourceLoc);
            }))
        return failure();

    // $body
    return parser.parseRegion(body, arguments);
}

static void printFunctor(OpAsmPrinter &printer, Operation *op, Region &body)
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
                printTypeBound(printer, op, arg.getType());

                // [ `loc` `(` attr `)` ]
                printer.printOptionalLocationSpecifier(arg.getLoc());
            });
        printer << ") ";
    }

    // $body
    printer.printRegion(body, false);
}

//===----------------------------------------------------------------------===//
// StaticOp directives
//===----------------------------------------------------------------------===//

static void
parseOptionalAccessModifier(OpAsmParser &parser, AccessModifier &modifier)
{
    // (`local` | `import` | `export`)?
    StringRef keyword{};
    if (parser.parseOptionalKeyword(&keyword, {"local", "import", "export"}))
        keyword = "local";

    modifier = llvm::StringSwitch<AccessModifier>(keyword)
                   .Case("import", AccessModifier::Import)
                   .Case("export", AccessModifier::Export)
                   .Default(AccessModifier::Local);
}

static ParseResult parseSymbolDecl(
    OpAsmParser &parser,
    StringAttr &sym_visibility,
    UnitAttr &isOwned,
    StringAttr &sym_name)
{
    // (`local` | `import` | `export`)?
    AccessModifier modifier;
    parseOptionalAccessModifier(parser, modifier);
    if (modifier != AccessModifier::Import)
        isOwned = parser.getBuilder().getUnitAttr();
    if (modifier != AccessModifier::Export)
        sym_visibility = parser.getBuilder().getStringAttr("private");

    // $sym_name
    return parser.parseSymbolName(sym_name);
}

static void printSymbolDecl(
    OpAsmPrinter &printer,
    Operation *,
    StringAttr sym_visibility,
    UnitAttr isOwned,
    StringAttr sym_name)
{
    // (`local` | `import` | `export`)?
    const auto isPublic =
        !sym_visibility || sym_visibility.getValue() == "public";
    if (isPublic) printer << "export ";
    if (!isOwned) printer << "import ";

    // $sym_name
    printer.printSymbolName(sym_name.getValue());
}

//===----------------------------------------------------------------------===//
// KernelOp directives
//===----------------------------------------------------------------------===//

static ParseResult parseSymbolDecl(
    OpAsmParser &parser,
    StringAttr &sym_visibility,
    StringAttr &sym_name)
{
    // (`public` | `private` | `nested`)?
    StringRef keyword{};
    if (parser.parseOptionalKeyword(&keyword, {"public", "private", "nested"}))
        keyword = "public";
    if (!keyword.empty())
        sym_visibility = parser.getBuilder().getStringAttr(keyword);

    // $sym_name
    return parser.parseSymbolName(sym_name);
}

static void printSymbolDecl(
    OpAsmPrinter &printer,
    Operation *,
    StringAttr sym_visibility,
    StringAttr sym_name)
{
    // (`public` | `private` | `nested`)?
    if (sym_visibility && sym_visibility.getValue() != "public")
        printer << sym_visibility.getValue() << " ";

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
        {"sym_name", "sym_visibility"});

    // $body
    printer << " ";
    printer.printRegion(body, false);
}

//===----------------------------------------------------------------------===//
// LiteralOp directives
//===----------------------------------------------------------------------===//

static ParseResult
parseLiteralResult(OpAsmParser &parser, LiteralAttr attr, Type &type)
{
    type = attr.getType();
    if (!parser.parseOptionalArrow()) {
        if (parser.parseType(type)) return failure();
    }

    type = ExpressionType::get(parser.getContext(), type);
    return success();
}

static void printLiteralResult(
    OpAsmPrinter &printer,
    Operation *,
    LiteralAttr attr,
    ExpressionType type)
{
    if (type.getTypeBound() == attr.getType()) return;
    printer << "-> " << type.getTypeBound();
}

//===----------------------------------------------------------------------===//
// IfOp directives
//===----------------------------------------------------------------------===//

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
// ProgramOp implementation
//===----------------------------------------------------------------------===//

LogicalResult ProgramOp::verifyRegions()
{
    // All descendants must be symbol declarations.
    for (auto &op : this->getOps())
        if (!op.hasTrait<ekl::OpTrait::IsSymbol>()) {
            auto diag = op.emitOpError("is not a symbol declaration");
            diag.attachNote(getLoc()) << "required by this operation";
            return diag;
        }

    return success();
}

//===----------------------------------------------------------------------===//
// IntroOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult IntroOp::fold(IntroOp::FoldAdaptor adaptor)
{
    // If the input value is a compatible LiteralAttr, it is materialized by the
    // dialect. Otherwise, it will be passed along by the folder, but there is
    // no guarantee this op will be deleted.
    return adaptor.getValue();
}

bool IntroOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    const auto in  = inputs.front();
    const auto out = llvm::dyn_cast<ExpressionType>(outputs.front());
    return out && out.getTypeBound() == in;
}

LogicalResult IntroOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location>,
    ValueRange operands,
    DictionaryAttr,
    OpaqueProperties,
    RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes)
{
    // Wrap the type in an ExpressionType.
    const auto type = operands.front().getType();
    inferredReturnTypes.push_back(ExpressionType::get(context, type));
    return success();
}

//===----------------------------------------------------------------------===//
// EvalOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult EvalOp::fold(EvalOp::FoldAdaptor adaptor)
{
    if (!isSpeculatable(*this)) return {};

    // Since the result type of the op is not an ExpressionType, the dialect
    // constant materializer will not be able to materialize any attribute
    // returned by this operation.
    return adaptor.getExpression();
}

Speculation::Speculatability EvalOp::getSpeculatability()
{
    // Speculation is not allowed until UB can be excluded.
    return isFullyTyped() ? Speculation::Speculatable
                          : Speculation::NotSpeculatable;
}

bool EvalOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    const auto in  = llvm::dyn_cast<ExpressionType>(inputs.front());
    const auto out = outputs.front();
    return in && (!in.getTypeBound() || isSubtype(in.getTypeBound(), out));
}

LogicalResult EvalOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // If the expression is typed, it must be a subtype of the result type.
    Type resultTy;
    return adaptor.require(getExpression(), getType(), resultTy);
}

//===----------------------------------------------------------------------===//
// StaticOp implementation
//===----------------------------------------------------------------------===//

void StaticOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    ReferenceType type,
    AccessModifier access,
    ekl::ArrayAttr initializer)
{
    state.addAttribute(
        getSymNameAttrName(state.name),
        builder.getStringAttr(name));
    state.addAttribute(getTypeAttrName(state.name), TypeAttr::get(type));
    if (initializer)
        state.addAttribute(getInitializerAttrName(state.name), initializer);

    const auto setPrivate = [&]() {
        state.addAttribute(
            getSymVisibilityAttrName(state.name),
            builder.getStringAttr("private"));
    };
    const auto setOwned = [&]() {
        state.addAttribute(
            getIsOwnedAttrName(state.name),
            builder.getUnitAttr());
    };

    switch (access) {
    case AccessModifier::Local:
        setOwned();
        setPrivate();
        break;
    case AccessModifier::Import:
        // By default, variables are not owned.
        setPrivate();
        break;
    case AccessModifier::Export:
        setOwned();
        // By default, variables are public.
        break;
    }
}

LogicalResult StaticOp::verify()
{
    if (isPublic() && !isOwned()) {
        // According to the SymbolTable rules, this doesn't make any sense.
        return emitOpError() << "public symbol must be owned";
    }

    if ((isPublic() || !isOwned()) && !llvm::isa<ABIReferenceType>(getType())) {
        // This variable does not have an ABI.
        return emitOpError()
            << "non-local variable requires ABI-compatible reference, found "
            << getType();
    }

    if (const auto initializer = getInitializerAttr()) {
        // If an initializer was provided, it must be valid.
        if (!isOwned())
            return emitOpError() << "can't initialize imported value";
        if (!isSubtype(initializer.getType(), getDataType())) {
            auto diag = emitOpError() << "initializer type mismatch: ";
            diag << initializer.getType() << " can't initialize value of type ";
            diag << getDataType();
            return diag;
        }

        return success();
    }

    // If no initializer was provided, it must not be needed.
    if (isPublic())
        return emitOpError() << "exported variable must be initialized";
    if (isOwned() && isReadable())
        return emitOpError() << "readable local variable must be initialized";

    return success();
}

LogicalResult StaticOp::canonicalize(StaticOp op, PatternRewriter &rewriter)
{
    // Remove the initializer attribute of a write-only local variable.
    if (!op.getInitializerAttr() || op.isPublic() || !op.isOwned()
        || op.isReadable())
        return failure();

    rewriter.updateRootInPlace(op, [&]() { op.removeInitializerAttr(); });
    return success();
}

AccessModifier StaticOp::getAccessModifier()
{
    // This must mirror the builder, assuming verification.
    if (isPublic()) return AccessModifier::Export;
    if (isOwned()) return AccessModifier::Local;
    return AccessModifier::Import;
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

void KernelOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    bool isPrivate)
{
    state.addRegion()->emplaceBlock();
    state.addAttribute(
        getSymNameAttrName(state.name),
        builder.getStringAttr(name));
    if (isPrivate)
        state.addAttribute(
            getSymVisibilityAttrName(state.name),
            builder.getStringAttr("private"));
}

LogicalResult KernelOp::verify()
{
    if (isPrivate()) return success();

    // For public kernels, the argument types must be ABI compatible.
    for (auto arg : getBody()->getArguments()) {
        const auto argTy = llvm::cast<Expression>(arg).getType().getTypeBound();
        if (llvm::isa<ABIType>(argTy)) continue;

        auto diag = emitOpError() << argTy << " is not ABI-compatible";
        diag.attachNote(arg.getLoc()) << "see argument #" << arg.getArgNumber();
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ReadOp implementation
//===----------------------------------------------------------------------===//

Speculation::Speculatability ReadOp::getSpeculatability()
{
    if (!isFullyTyped()) return Speculation::NotSpeculatable;
    // TODO: Implement more concrete reference kinds.
    return Speculation::Speculatable;
}

LogicalResult ReadOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The reference operand must be a readable reference.
    ReadableRefType refTy;
    if (auto contra =
            adaptor.require(getReference(), refTy, "readable reference"))
        return contra;

    // Result type is the referenced array type.
    return adaptor.refineBound(getResult(), refTy.getArrayType());
}

//===----------------------------------------------------------------------===//
// WriteOp implementation
//===----------------------------------------------------------------------===//

Speculation::Speculatability WriteOp::getSpeculatability()
{
    if (!isFullyTyped()) return Speculation::NotSpeculatable;
    // TODO: Implement more concrete reference kinds.
    return Speculation::NotSpeculatable;
}

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

static void buildIf(
    OpBuilder &builder,
    OperationState &state,
    Value condition,
    BranchBuilderRef thenBranch,
    BranchBuilderRef elseBranch,
    Type resultType = {})
{
    state.addOperands({condition});
    if (resultType) {
        assert(llvm::isa<ExpressionType>(resultType));
        state.addTypes({resultType});
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&state.addRegion()->emplaceBlock());
    if (thenBranch) thenBranch(builder, state.location);
    if (elseBranch || resultType) {
        builder.setInsertionPointToStart(&state.addRegion()->emplaceBlock());
        if (elseBranch) elseBranch(builder, state.location);
    }
}

void IfOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value condition,
    BranchBuilderRef thenBranch,
    BranchBuilderRef elseBranch)
{
    buildIf(builder, state, condition, thenBranch, elseBranch);
}

void IfOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value condition,
    Type resultBound,
    BranchBuilderRef thenBranch,
    BranchBuilderRef elseBranch)
{
    buildIf(
        builder,
        state,
        condition,
        thenBranch,
        elseBranch,
        ExpressionType::get(builder.getContext(), resultBound));
}

LogicalResult IfOp::verify()
{
    // The if branches may not have any arguments when they are present.
    if (getThenRegion().getNumArguments() != 0)
        return emitOpError() << "no arguments allowed in then branch";
    if (getElseRegion().getNumArguments() != 0)
        return emitOpError() << "no arguments allowed in else branch";

    if (isStatement()) {
        const auto noTerminator = [&](Block *block) -> LogicalResult {
            if (!block || block->empty()) return success();
            auto op = &block->back();
            if (!op->hasTrait<mlir::OpTrait::IsTerminator>()) return success();

            auto diag = emitOpError() << "unexpected region terminator";
            diag.attachNote(op->getLoc()) << "see terminator";
            return diag;
        };

        // The statement if branches may not have terminators.
        if (failed(noTerminator(getThenBranch()))) return failure();
        if (failed(noTerminator(getElseBranch()))) return failure();
        return success();
    }

    const auto yesTerminator = [&](Block *block,
                                   const llvm::Twine &name) -> LogicalResult {
        if (!block) return emitOpError() << name << " is required";
        Operation *op = block->empty() ? nullptr : &block->back();
        if (llvm::isa_and_present<YieldOp>(op)) return success();
        // NOTE: HasFunctors has already barked at any non YieldOp terminator.
        return emitOpError() << name << " requires terminator";
    };

    // The expression if branches must have terminators.
    if (failed(yesTerminator(getThenBranch(), "then branch"))) return failure();
    if (failed(yesTerminator(getElseBranch(), "else branch"))) return failure();
    return success();
}

LogicalResult IfOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The condition must be a boolean.
    BoolType boolTy;
    if (auto contra = adaptor.require(getCondition(), boolTy, "bool"))
        return contra;

    if (isStatement()) return success();

    // Unify the types of both branch expressions.
    const auto trueVal  = getThenExpression();
    const auto falseVal = getElseExpression();
    Type unifiedTy;
    if (auto contra = adaptor.unify(ValueRange({trueVal, falseVal}), unifiedTy))
        return contra;

    return adaptor.refineBound(getExpression(), unifiedTy);
}

Expression IfOp::getThenExpression()
{
    if (!isExpression()) return {};
    return llvm::cast<YieldOp>(getThenBranch()->getTerminator())
        .getExpression();
}

Expression IfOp::getElseExpression()
{
    if (!isExpression()) return {};
    return llvm::cast<YieldOp>(getElseBranch()->getTerminator())
        .getExpression();
}

//===----------------------------------------------------------------------===//
// LiteralOp implementation
//===----------------------------------------------------------------------===//

LogicalResult LiteralOp::verify()
{
    if (!isSubtype(getValue().getType(), getType().getTypeBound()))
        return emitOpError()
            << getType().getTypeBound() << " is not a supertype of "
            << getValue().getType();

    return success();
}

OpFoldResult LiteralOp::fold(LiteralOp::FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// GetStaticOp implementation
//===----------------------------------------------------------------------===//

void GetStaticOp::build(
    OpBuilder &builder,
    OperationState &state,
    StaticOp target)
{
    state.addTypes(
        {ExpressionType::get(builder.getContext(), target.getType())});
    state.addAttribute(
        getTargetNameAttrName(state.name),
        FlatSymbolRefAttr::get(target.getNameAttr()));
}

LogicalResult GetStaticOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    auto targetOp =
        symbolTable.lookupNearestSymbolFrom(*this, getTargetNameAttr());
    auto staticOp = llvm::dyn_cast_if_present<StaticOp>(targetOp);
    if (!staticOp) {
        auto diag = emitOpError("'")
                 << getTargetName() << "' does not reference a static variable";
        if (targetOp)
            diag.attachNote(targetOp->getLoc()) << "references this symbol";
        return diag;
    }

    if (!isSubtype(getType().getTypeBound(), staticOp.getType())) {
        auto diag = emitOpError()
                 << getType().getTypeBound() << " is not a subtype of "
                 << staticOp.getType();
        diag.attachNote(targetOp->getLoc()) << "symbol declared here";
        return diag;
    }

    return success();
}

LogicalResult GetStaticOp::typeCheck(AbstractTypeChecker &)
{
    return success();
}

//===----------------------------------------------------------------------===//
// SubscriptOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult SubscriptOp::fold(SubscriptOp::FoldAdaptor adaptor)
{
    if (!isSpeculatable(*this)) return {};

    // Must have constant array.
    const auto array =
        llvm::dyn_cast_if_present<ekl::ArrayAttr>(adaptor.getArray());
    if (!array) return {};

    const auto bounds = array.getType().getExtents();
    if (adaptor.getSubscripts().size() > bounds.size()) return {};

    // Must be constant index values only.
    SmallVector<extent_t> indices;
    for (auto [attr, bound] :
         llvm::zip_first(adaptor.getSubscripts(), bounds)) {
        const auto indexAttr = llvm::dyn_cast_if_present<ekl::IndexAttr>(attr);
        if (!indexAttr || indexAttr.getValue() >= bound) return {};
        indices.push_back(indexAttr.getValue());
    }

    // Perform the subscript operation.
    return array.subscript(indices);
}

[[nodiscard]] static BlockArgument getInferrableIndex(Value value)
{
    // Must be a block argument.
    const auto argument = llvm::dyn_cast<BlockArgument>(value);
    if (!argument) return {};

    // Must be from the map region of an AssocOp.
    const auto owner = argument.getOwner()->getParentOp();
    if (!llvm::isa_and_present<AssocOp>(owner)) return {};
    return argument;
}

static FailureOr<ekl::IndexType> meetIndexBound(
    AbstractTypeChecker &typeChecker,
    BlockArgument index,
    uint64_t bound)
{
    // Update the bound on the index value, which will fail if there is already
    // a different bound.
    const auto type = ekl::IndexType::get(index.getContext(), bound);
    if (failed(typeChecker.meetBound(llvm::cast<Expression>(index), type)))
        return failure();

    // This invalidates the owner as well.
    typeChecker.invalidate(index.getParentRegion()->getParentOp());
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
            if (getInferrableIndex(subscript)) {
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

    return unbounded ? Contradiction::indeterminate() : Contradiction::none();
}

LogicalResult SubscriptOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Array operand must be an array type.
    ArrayType arrayTy;
    if (auto contra = adaptor.require(getArray(), arrayTy, "array"))
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
            const auto index = getInferrableIndex(value);
            assert(index);

            // The subscript type checker let this through because it can be
            // inferred from the array extents here.
            const auto meet = meetIndexBound(
                typeChecker,
                index,
                arrayTy.getExtent(sourceDim) - 1UL);
            if (failed(meet)) return failure();
            bound = *meet;
            assert(bound);
        }
        if (llvm::isa<ExtentType>(bound)) {
            // Insert a new unit dimension.
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
            // Insert the identity indexer until enough dimensions are bound.
            while (remaining < (arrayTy.getNumExtents() - sourceDim))
                extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        // For all other kinds of subscripts, bind 1 source dimension.
        if (sourceDim == arrayTy.getNumExtents()) {
            auto diag = emitError() << "exceeded number of array extents ("
                                    << arrayTy.getNumExtents() << ")";
            diag.attachNote(value.getLoc()) << "with this subscript";
            return diag;
        }
        if (llvm::isa<IdentityType>(bound)) {
            // Map this dimension using the identity.
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

    // For partial subscripting, append all the remaining dimensions.
    concat(extents, arrayTy.getExtents().drop_front(sourceDim));

    // The result is an array with the inferred extents, decaying to a scalar
    // if the extents are empty.
    Type resultTy = arrayTy.getScalarType();
    if (!extents.empty()) resultTy = ArrayType::get(resultTy, extents);
    return adaptor.refineBound(getResult(), resultTy);
}

//===----------------------------------------------------------------------===//
// StackOp implementation
//===----------------------------------------------------------------------===//

LogicalResult StackOp::verify()
{
    if (getOperands().empty())
        return emitOpError() << "requires at least 1 operand";

    return success();
}

OpFoldResult StackOp::fold(StackOp::FoldAdaptor adaptor)
{
    if (!isSpeculatable(*this)) return {};

    const auto arrayTy =
        llvm::dyn_cast_if_present<ArrayType>(getType().getTypeBound());
    if (!arrayTy) return {};
    if (llvm::count(adaptor.getOperands(), Attribute{}) > 0) return {};

    return ekl::ArrayAttr::get(arrayTy, adaptor.getOperands());
}

LogicalResult StackOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The operands must all broadcast and unify together.
    BroadcastType unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(getOperands(), unifiedTy))
        return contra;

    // The result extents are determined by the unified extents, prepended by
    // the number of stacked atoms.
    const auto resultExtents = concat(
        {static_cast<uint64_t>(getOperands().size())},
        unifiedTy.getExtents());
    return adaptor.refineBound(getResult(), unifiedTy.cloneWith(resultExtents));
}

//===----------------------------------------------------------------------===//
// YieldOp implementation
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    // When this op is invalidated, the parent should update as well.
    if (const auto parent = (*this)->getParentOp())
        typeChecker.invalidate(parent);

    return success();
}

//===----------------------------------------------------------------------===//
// AssocOp implementation
//===----------------------------------------------------------------------===//

void AssocOp::build(
    OpBuilder &builder,
    OperationState &state,
    unsigned numExtents,
    FunctorBuilderRef map)
{
    const auto unboundedTy = ExpressionType::get(builder.getContext());

    auto &functor = state.addRegion()->emplaceBlock();
    state.addTypes({unboundedTy});

    const auto loc = builder.getUnknownLoc();
    while (numExtents-- > 0U) functor.addArgument(unboundedTy, loc);

    if (map) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&functor);
        map(builder, state.location, functor.getArguments());
    }
}

void AssocOp::build(
    OpBuilder &builder,
    OperationState &state,
    ExtentRange extents,
    FunctorBuilderRef map)
{
    auto &functor = state.addRegion()->emplaceBlock();
    state.addTypes({ExpressionType::get(builder.getContext())});

    const auto loc = builder.getUnknownLoc();
    for (auto extent : extents)
        functor.addArgument(
            ExpressionType::get(
                builder.getContext(),
                ekl::IndexType::get(builder.getContext(), extent - 1UL)),
            loc);

    if (map) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&functor);
        map(builder, state.location, functor.getArguments());
    }
}

void AssocOp::build(
    OpBuilder &builder,
    OperationState &state,
    ArrayType resultBound,
    FunctorBuilderRef map)
{
    const auto extents = resultBound ? resultBound.getExtents() : ExtentRange{};
    build(builder, state, extents, map);
    state.types[0] = ExpressionType::get(builder.getContext(), resultBound);
}

OpFoldResult AssocOp::fold(AssocOp::FoldAdaptor)
{
    if (!isSpeculatable(*this)) return {};

    // If the yielded expression was folded to a scalar, a splat can be derived.
    const auto expr = getMapExpression();
    ScalarAttr value;
    if (!matchPattern(expr, m_Constant(&value))) return {};
    return ekl::ArrayAttr::get(
        llvm::cast<ArrayType>(getType().getTypeBound()),
        value);
}

static Contradiction typeCheckMap(
    TypeCheckingAdaptor &adaptor,
    Block *map,
    Type &yieldTy,
    SmallVectorImpl<extent_t> &extents)
{
    // Ensure that all the arguments are indices and infer the extents.
    auto validExtents = true;
    for (auto arg : map->getArguments()) {
        ekl::IndexType indexTy;
        auto contra =
            adaptor.require(llvm::cast<Expression>(arg), indexTy, "index");
        if (failed(contra)) return contra;

        if (!indexTy || indexTy.isUnbounded()) {
            // The premise is contradicted already, but find more errors.
            validExtents = false;
        } else
            extents.push_back(indexTy.getUpperBound() + 1UL);
    }

    // Obtain the yielded value type.
    auto yield = llvm::cast<YieldOp>(map->getTerminator());
    if (!(yieldTy = adaptor.getType(yield.getExpression())))
        return Contradiction::indeterminate();

    // If the extents were not inferred, abort.
    return validExtents ? Contradiction::none()
                        : Contradiction::indeterminate();
}

LogicalResult AssocOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Type-check the map functor.
    Type yieldTy;
    SmallVector<extent_t> extents;
    if (auto contra = typeCheckMap(adaptor, getMap(), yieldTy, extents))
        return contra;

    // Determine the effective extents by concatenating what was declared with
    // what was yielded by the functor.
    const auto yieldExtents = getExtents(yieldTy);
    if (failed(yieldExtents)) {
        auto diag = emitError() << "expected scalar or array";
        diag.attachNote(getMapExpression().getLoc()) << "for this value";
        return diag;
    }
    concat(extents, *yieldExtents);

    // The result is always an array type.
    return adaptor.refineBound(
        getResult(),
        ArrayType::get(getScalarType(yieldTy), extents));
}

void AssocOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setName)
{
    static constexpr StringRef indexLetters = "ijklmnopqrstuvwxyz";
    for (auto &&[arg, name] : llvm::zip(region.getArguments(), indexLetters))
        setName(arg, StringRef(&name, 1));
}

Expression AssocOp::getMapExpression()
{
    return llvm::cast<YieldOp>(getMap()->getTerminator()).getExpression();
}

//===----------------------------------------------------------------------===//
// ZipOp implementation
//===----------------------------------------------------------------------===//

void ZipOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange operands,
    FunctorBuilderRef combinator,
    Type resultBound)
{
    auto &functor = state.addRegion()->emplaceBlock();

    state.addOperands(operands);
    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});

    SmallVector<Type> types(
        operands.size(),
        ExpressionType::get(builder.getContext()));
    SmallVector<Location> locs(operands.size(), builder.getUnknownLoc());
    functor.addArguments(types, locs);

    if (combinator) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&functor);
        combinator(builder, state.location, functor.getArguments());
    }
}

LogicalResult ZipOp::verify()
{
    if (getCombinator()->getNumArguments() != getNumOperands())
        return emitOpError() << "expected " << getNumOperands()
                             << " arguments to combinator functor";

    return success();
}

LogicalResult ZipOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // The operands must broadcast together.
    SmallVector<Type> argTys;
    if (auto contra = adaptor.broadcast(getOperands(), argTys)) return contra;

    // Refine the bounds on the combinator.
    for (auto [idx, arg] : llvm::enumerate(getCombinator()->getArguments())) {
        const auto expr = llvm::cast<Expression>(arg);
        if (failed(adaptor.refineBound(expr, getScalarType(argTys[idx]))))
            return failure();
    }

    const auto yieldTy = adaptor.getType(getCombinatorExpression());
    if (!yieldTy) return success();
    if (argTys.empty()) return adaptor.refineBound(getResult(), yieldTy);

    // Determine the effective extents by concatenating what was inferred with
    // what was yielded by the functor.
    const auto yieldExtents = getExtents(yieldTy);
    if (failed(yieldExtents)) {
        auto diag = emitError() << "expected scalar or array";
        diag.attachNote(getCombinatorExpression().getLoc()) << "for this value";
        return diag;
    }

    // The result type copies extents from the arguments to the yielded scalar.
    const auto exemplar = llvm::cast<BroadcastType>(argTys.front());
    const auto extents  = concat(exemplar.getExtents(), *yieldExtents);
    return adaptor.refineBound(
        getResult(),
        exemplar.cloneWith(getScalarType(yieldTy)));
}

Expression ZipOp::getCombinatorExpression()
{
    return llvm::cast<YieldOp>(getCombinator()->getTerminator())
        .getExpression();
}

//===----------------------------------------------------------------------===//
// ReduceOp implementation
//===----------------------------------------------------------------------===//

void ReduceOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value array,
    FunctorBuilderRef reduction,
    Type resultBound,
    Value init)
{
    auto &functor = state.addRegion()->emplaceBlock();

    state.addOperands({array});
    if (init) state.addOperands({init});
    state.addTypes({ExpressionType::get(builder.getContext(), resultBound)});

    const auto loc    = builder.getUnknownLoc();
    const auto exprTy = ExpressionType::get(builder.getContext());
    functor.addArguments({exprTy, exprTy}, {loc, loc});

    if (reduction) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&functor);
        reduction(builder, state.location, functor.getArguments());
    }
}

LogicalResult ReduceOp::verify()
{
    if (getReduction()->getNumArguments() != 2)
        return emitOpError() << "expected 2 arguments to reduction block";

    return success();
}

LogicalResult ReduceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    ArrayType arrayTy;
    if (auto contra = adaptor.require(getArray(), arrayTy, "array"))
        return contra;

    // The element expression must accept the array elements.
    if (failed(adaptor.refineBound(
            getElementExpression(),
            arrayTy.getScalarType())))
        return failure();

    // Refine the bound on the accumulator expression.
    if (const auto init = getInitExpression()) {
        // The accumulator expression must accept the initializer.
        const auto initTy = adaptor.getType(init);
        if (!initTy) return success();
        if (failed(adaptor.refineBound(getAccumulatorExpression(), initTy)))
            return failure();
    } else {
        // The accumulator expression must accept array elements.
        if (failed(adaptor.refineBound(
                getAccumulatorExpression(),
                arrayTy.getScalarType())))
            return failure();
    }

    // The reduction expression must be accepted by the accumulator.
    const auto accuTy = adaptor.getType(getAccumulatorExpression());
    Type redTy;
    if (auto contra = adaptor.require(getReductionExpression(), accuTy, redTy))
        return contra;

    // The result is the accumulator.
    return adaptor.refineBound(getResult(), accuTy);
}

Expression ReduceOp::getReductionExpression()
{
    return llvm::cast<YieldOp>(getReduction()->getTerminator()).getExpression();
}

//===----------------------------------------------------------------------===//
// ConstexprOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult ConstexprOp::fold(ConstexprOp::FoldAdaptor)
{
    if (!isSpeculatable(*this)) return {};

    // Fold to the constant expression value.
    LiteralAttr literal;
    if (matchPattern(getExpression(), m_Constant(&literal))) return literal;
    return {};
}

LogicalResult ConstexprOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Refine the bound to the yielded expression type.
    const auto yieldTy = adaptor.getType(getExpression());
    if (!yieldTy) return success();
    return adaptor.refineBound(getResult(), yieldTy);
}

Expression ConstexprOp::getExpression()
{
    return llvm::cast<YieldOp>(getBody()->getTerminator()).getExpression();
}

//===----------------------------------------------------------------------===//
// UnifyOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult UnifyOp::fold(UnifyOp::FoldAdaptor adaptor)
{
    if (!isSpeculatable(*this)) return {};

    // Attributes are covariant in the IR, no unification happens. The
    // materializer will produce a LiteralOp with a different type.
    return adaptor.getOperand();
}

LogicalResult UnifyOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type resultTy;
    return adaptor.require(getOperand(), getType().getTypeBound(), resultTy);
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold(BroadcastOp::FoldAdaptor adaptor)
{
    if (!isSpeculatable(*this) || !adaptor.getOperand()) return {};

    const auto resultTy = llvm::cast<ArrayType>(getType().getTypeBound());

    // Fold scalar-to-array broadcasts.
    if (const auto scalar = llvm::dyn_cast<ScalarAttr>(adaptor.getOperand()))
        return ArrayAttr::get(resultTy, {scalar});
    // Fold array-to-array broadcasts.
    if (const auto array = llvm::dyn_cast<ekl::ArrayAttr>(adaptor.getOperand()))
        return array.broadcastTo(resultTy.getExtents());

    return {};
}

LogicalResult BroadcastOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    ArrayType resultTy;
    if (auto contra = adaptor.broadcast(getOperand(), getExtents(), resultTy))
        return contra;

    return adaptor.refineBound(getResult(), resultTy);
}

DenseI64ArrayAttr
BroadcastOp::getSignedExtentsAttr(MLIRContext *context, ExtentRange extents)
{
    return DenseI64ArrayAttr::get(
        context,
        ArrayRef<int64_t>(
            reinterpret_cast<const int64_t *>(extents.data()),
            extents.size()));
}

ExtentRange BroadcastOp::getExtents()
{
    return ExtentRange(
        std::bit_cast<const extent_t *>(getSignedExtents().data()),
        getSignedExtents().size());
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

LogicalResult CoerceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    return adaptor.coerce(getOperand(), getType().getTypeBound());
}

//===----------------------------------------------------------------------===//
// ChoiceOp implementation
//===----------------------------------------------------------------------===//

LogicalResult ChoiceOp::verify()
{
    if (getAlternatives().empty())
        return emitOpError() << "requires at least 1 alternative";

    return success();
}

LogicalResult ChoiceOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Check the selector type.
    const auto selectorTy = adaptor.getType(getSelector());
    if (!selectorTy) return success();
    const auto choiceTy = getScalarType(selectorTy);
    if (!choiceTy) {
        auto diag = emitError() << "selector must be scalar or array";
        diag.attachNote(getSelector().getLoc()) << "from this expression";
        return diag;
    }

    // Check the selector scalar type.
    unsigned minArity = 1;
    if (llvm::isa<BoolType>(choiceTy)) {
        minArity = 2;
    } else if (const auto indexTy = llvm::dyn_cast<ekl::IndexType>(choiceTy)) {
        if (!indexTy.isUnbounded()) minArity = indexTy.getUpperBound() + 1UL;
    } else {
        auto diag = emitError() << "selector must be bool or index";
        diag.attachNote(getSelector().getLoc()) << "from this expression";
        return diag;
    }

    // Check the arity.
    if (getAlternatives().size() < minArity) {
        auto diag = emitError()
                 << "too few alternatives (" << getAlternatives().size()
                 << " < " << minArity << ")";
        diag.attachNote(getSelector().getLoc()) << "for this selector";
        return diag;
    }

    // Broadcast and unify the alternatives.
    BroadcastType altTy;
    if (auto contra = adaptor.broadcastAndUnify(getAlternatives(), altTy))
        return contra;

    // Determine the result extents by extending and broadcasting.
    auto resultExtents = llvm::to_vector(*getExtents(selectorTy));
    if (resultExtents.size() < altTy.getExtents().size())
        resultExtents.append(
            altTy.getExtents().size() - resultExtents.size(),
            1UL);
    if (failed(ekl::broadcast(resultExtents, altTy.getExtents()))) {
        auto diag = emitError() << "can't broadcast [";
        llvm::interleaveComma(resultExtents, diag);
        diag << "] and " << altTy << "together";
        diag.attachNote(getSelector().getLoc()) << "for this selector";
        return diag;
    }

    // The result type has the alternative scalar type and the inferred extents.
    return adaptor.refineBound(getResult(), altTy.cloneWith(resultExtents));
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

    BroadcastType unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(getOperands(), unifiedTy))
        return contra;

    // Unified scalar type must be a number (or bool for eq/ne).
    if (!llvm::isa<NumericType>(unifiedTy.getScalarType())) {
        switch (getKind()) {
        case RelationKind::Equivalent:
        case RelationKind::Antivalent:
            if (llvm::isa<BoolType>(unifiedTy.getScalarType())) {
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
        unifiedTy.cloneWith(BoolType::get(getContext())));
}

//===----------------------------------------------------------------------===//
// Logical operator implementation
//===----------------------------------------------------------------------===//

static LogicalResult typeCheckLogicalOp(TypeCheckingAdaptor &adaptor)
{
    // The operands must all unify or broadcast together.
    LogicType unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(
            adaptor.getParent()->getOperands(),
            unifiedTy,
            "logic type"))
        return contra;

    // The result type is the unified type, decayed to a scalar.
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
    ArithmeticType unifiedTy;
    if (auto contra = adaptor.broadcastAndUnify(
            adaptor.getParent()->getOperands(),
            unifiedTy,
            "arithmetic type"))
        return contra;

    // Arithmetic operations need to properly update the upper bounds on the
    // types of index values they produce.
    if (llvm::isa<ekl::IndexType>(unifiedTy.getScalarType())) {
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
            unifiedTy.cloneWith(ekl::IndexType::get(
                unifiedTy.getContext(),
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
            if (llvm::count(bounds, ekl::IndexType::kUnbounded) > 0)
                return ekl::IndexType::kUnbounded;
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
            if (llvm::count(bounds, ekl::IndexType::kUnbounded) > 0)
                return ekl::IndexType::kUnbounded;
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
        [](ArrayRef<uint64_t> bounds) -> uint64_t {
            if (bounds[1] == ekl::IndexType::kUnbounded)
                return ekl::IndexType::kUnbounded;
            return bounds[1] - 1;
        });
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
    ekl::ArithmeticType lhsTy;
    if (auto contra = adaptor.require(getLhs(), lhsTy, "arithmetic type"))
        return contra;
    ekl::ArithmeticType rhsTy;
    if (auto contra = adaptor.require(getRhs(), rhsTy, "arithmetic type"))
        return contra;
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
