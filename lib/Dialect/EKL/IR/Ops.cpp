/// Implements the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

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
    // `(` [ ssa-id [ `:` type ] { `,` ssa-id [ `:` type ] } ] `)`
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
    // `(` [ ssa-id [ `:` type ] { `,` ssa-id [ `:` type ] } ] `)`
    printer << "(";
    llvm::interleaveComma(
        body.getArguments(),
        printer,
        [&](BlockArgument &arg) {
            printer << arg << ": "
                    << llvm::cast<ExpressionType>(arg.getType()).getTypeBound();
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
                return success();
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
// GlobalOp implementation
//===----------------------------------------------------------------------===//

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

LogicalResult UnifyOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    Type resultTy;
    return adaptor.require(getInput(), getType().getTypeBound(), resultTy);
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

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

static FailureOr<ekl::IndexType> meetIndexBound(
    AbstractTypeChecker &typeChecker,
    Expression expr,
    uint64_t bound)
{
    // Only applies to block arguments.
    const auto argument = llvm::dyn_cast<BlockArgument>(expr);
    if (!argument) return success(ekl::IndexType{});

    // Only applies to block arguments declared by a domain expression.
    const auto owner = argument.getOwner()->getParentOp();
    if (!llvm::isa_and_present<AssocOp, ReduceOp>(owner))
        return success(ekl::IndexType{});

    // Update the bound on the index value, which will fail if there is already
    // a different bound.
    const auto type = ekl::IndexType::get(expr.getContext(), bound);
    if (failed(typeChecker.meetBound(expr, type))) return failure();

    // This invalidates the owner as well.
    typeChecker.invalidate(owner);
    return success(type);
}

LogicalResult SubscriptOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Array operand must be an array or reference type.
    ContiguousType arrayTy;
    if (auto contra =
            adaptor.require(getArray(), arrayTy, "array or reference"))
        return contra;

    // Check all the subscript operands and attempt to infer the result extents.
    auto sourceDim    = 0U;
    auto extentsValid = true;
    SmallVector<uint64_t> extents;
    for (auto subscript : getSubscripts()) {
        auto bound = adaptor.getType(llvm::cast<Expression>(subscript));
        if (!bound) {
            // Let us try to bound this index.
            const auto meet = meetIndexBound(
                typeChecker,
                llvm::cast<Expression>(subscript),
                arrayTy.getExtent(sourceDim) - 1UL);
            if (failed(meet)) return failure();
            if (!(bound = *meet)) {
                // We really don't have a bound yet, so the extents will be
                // wrong.
                extentsValid = false;
                continue;
            }
        }
        if (llvm::isa<ExtentType>(bound)) {
            // We insert a new unit dimension.
            extents.push_back(1UL);
            continue;
        }

        // For all other kinds of subscripts, we must map to some source
        // dimension!
        if (sourceDim == arrayTy.getNumExtents()) {
            auto diag = emitError() << "exceeded number of array extents ("
                                    << arrayTy.getNumExtents() << ")";
            diag.attachNote(subscript.getLoc()) << "with this subscript";
            return diag;
        }
        if (llvm::isa<IdentityType>(bound)) {
            // We map this dimension using the identity.
            extents.push_back(arrayTy.getExtent(sourceDim++));
            continue;
        }

        // We expect something that resolves to an index.
        const auto indexTy =
            llvm::dyn_cast_if_present<ekl::IndexType>(getScalarType(bound));
        if (!indexTy) {
            auto diag = emitError() << "expected indexer, but got " << bound;
            diag.attachNote(subscript.getLoc()) << "for this subscript";
            return diag;
        }

        // Handle statically known index bounds.
        if (!indexTy.isUnbounded()
            && indexTy.getUpperBound() >= arrayTy.getExtent(sourceDim)) {
            auto diag = emitOpError()
                     << "index out of bounds (" << indexTy.getUpperBound()
                     << " >= " << arrayTy.getExtent(sourceDim) << ")";
            diag.attachNote(subscript.getLoc()) << "for this subscript";
            return diag;
        }

        // Insert the indexer's extents here, and skip this dimension in the
        // source.
        ++sourceDim;
        concat(extents, getExtents(bound).value_or(ExtentRange{}));
    }

    if (!extentsValid) {
        // We don't have all subscripts yet, so we can't deduce anything else.
        return success();
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
    ScalarType &scalarTy,
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

    // Check that a scalar is yielded.
    auto yield = llvm::cast<YieldOp>(&map.front().back());
    if (auto contra = adaptor.require(
            llvm::cast<Expression>(yield.getValue()),
            scalarTy,
            "scalar"))
        return contra;

    return success();
}

LogicalResult AssocOp::verify() { return verifyFunctor(*this, getMap()); }

LogicalResult AssocOp::typeCheck(AbstractTypeChecker &typeChecker)
{
    TypeCheckingAdaptor adaptor(typeChecker, *this);

    // Type-check the map functor.
    ScalarType scalarTy;
    SmallVector<extent_t> extents;
    bool validExtents;
    if (failed(
            typeCheckMap(adaptor, getMap(), scalarTy, extents, validExtents)))
        return failure();

    // Only iff all of the above is known can we deduce the result type.
    if (!validExtents || !scalarTy) return success();

    // Refine the bound on the result.
    if (extents.empty()) return adaptor.refineBound(getResult(), scalarTy);
    return adaptor.refineBound(getResult(), ArrayType::get(scalarTy, extents));
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
// ReduceOp implementation
//===----------------------------------------------------------------------===//

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
// CompareOp implementation
//===----------------------------------------------------------------------===//

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
