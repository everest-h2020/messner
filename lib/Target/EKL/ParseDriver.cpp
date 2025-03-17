/// Implements the ParseDriver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ParseDriver.h"

#include "messner/Dialect/EKL/Transforms/TypeCheck.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "ekl-parser"

using namespace mlir;
using namespace mlir::ekl;

static llvm::cl::opt<unsigned> errorLimit{
    "ekl-import-error-limit",
    llvm::cl::init(5U),
    llvm::cl::desc(
        "Number of lexer / parser errors to accumulate before aborting.")};

//===----------------------------------------------------------------------===//
// ParseDriver implementation
//===----------------------------------------------------------------------===//

ParseDriver::ParseDriver(
    MLIRContext *context,
    std::shared_ptr<llvm::SourceMgr> sourceMgr)
        : m_sourceMgr(std::move(sourceMgr)),
          m_filename(),
          m_source(),
          m_hasWarnings(),
          m_numErrors(0U),
          m_builder(context),
          m_result(),
          m_scopes(),
          m_constexprPatterns()
{
    assert(m_sourceMgr);

    // Initialize the context variables.
    const auto buffer =
        m_sourceMgr->getMemoryBuffer(m_sourceMgr->getMainFileID());
    m_filename     = m_builder.getStringAttr(buffer->getBufferIdentifier());
    m_source       = buffer->getBuffer();
    const auto loc = FileLineColLoc::get(m_filename, 1U, 1U);

    // Create the ProgramOp root and enter it.
    m_result = m_builder.create<ProgramOp>(loc);
    m_builder.setInsertionPointToStart(m_result->getBody());

    // Open the file-level scope and add the builtins.
    m_scopes.push(llvm::SMLoc::getFromPointer(m_source.data()));

    // Initialize the constexpr evaluation patterns.
    RewritePatternSet constexprPatterns(getContext());
    for (auto *dialect : getContext()->getLoadedDialects())
        dialect->getCanonicalizationPatterns(constexprPatterns);
    for (auto &op : getContext()->getRegisteredOperations())
        op.getCanonicalizationPatterns(constexprPatterns, getContext());
    populateHomogenizePatterns(constexprPatterns);
    populateLowerPatterns(constexprPatterns);
    m_constexprPatterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(constexprPatterns));
}

OwningOpRef<ProgramOp> ParseDriver::takeResult()
{
    if (m_scopes.size() > 1) {
        // Ensure we don't accept programs that still have open scopes besides
        // the file scope we opened in the constructor.
        emitError(m_scopes.peek().getStartLoc(), "unfinished scope");
        ++m_numErrors;
    }

    if (getNumErrors() > 0) {
        emitError("failed to parse input");

        LLVM_DEBUG(llvm::dbgs() << "[Parser] dumping IR before delete:\n";
                   m_result->print(
                       llvm::dbgs(),
                       OpPrintingFlags{}.printGenericOpForm());
                   llvm::dbgs() << "\n");

        m_result.release();
    } else if (hasWarnings())
        emitWarning("parsing completed with warnings");

    m_scopes.clear();
    return std::move(m_result);
}

Location ParseDriver::getLocation(ImportLocation loc) const
{
    if (!loc.begin.isValid() || !loc.end.isValid()) return getUnknownLoc();

    const auto [startLine, startColumn] = getSourceMgr().getLineAndColumn(
        loc.begin,
        getSourceMgr().getMainFileID());
    const auto [endLine, endColumn] = getSourceMgr().getLineAndColumn(
        loc.end,
        getSourceMgr().getMainFileID());

    return SourceLocationAttr::get(
               m_filename,
               startLine,
               startColumn,
               endLine,
               endColumn)
        .toLocation();
}

template<llvm::SourceMgr::DiagKind Kind>
static void
emit(llvm::SourceMgr &sourceMgr, ImportLocation where, const llvm::Twine &msg)
{
    // Do not indicate a source range if just a single character is referenced.
    // NOTE: May not actually be needed.
    ArrayRef<llvm::SMRange> ranges(where);
    if (where.begin == where.end) ranges = {};

    // Print a bold error message with the red "error:" label, prefixed by the
    // location, and followed by an excerpt from the source file.
    sourceMgr.PrintMessage(where.begin, Kind, msg, ranges);
}

void ParseDriver::emitWarning(const llvm::Twine &msg)
{
    m_hasWarnings = true;

    llvm::WithColor(llvm::WithColor::warning(), raw_ostream::SAVEDCOLOR, true)
        << msg << "\n";
}

void ParseDriver::emitWarning(ImportLocation where, const llvm::Twine &msg)
{
    m_hasWarnings = true;

    emit<llvm::SourceMgr::DiagKind::DK_Warning>(getSourceMgr(), where, msg);
}

void ParseDriver::emitError(const llvm::Twine &msg)
{
    // Print a bold error message with the red "error:" label.
    llvm::WithColor(llvm::WithColor::error(), raw_ostream::SAVEDCOLOR, true)
        << msg << "\n";
}

void ParseDriver::emitError(ImportLocation where, const llvm::Twine &msg)
{
    emit<llvm::SourceMgr::DiagKind::DK_Error>(getSourceMgr(), where, msg);
}

LogicalResult ParseDriver::recover()
{
    const auto limit = errorLimit.getValue();
    if (m_numErrors++ == limit) emitError("too many errors, aborting");

    return success(m_numErrors <= limit);
}

std::optional<LogicalResult>
ParseDriver::parseIndex(Token token, extent_t &value)
{
    auto [loc, text] = token;

    // Parse the index literal.
    text = text.drop_front(1);
    if (!text.consumeInteger(10, value) && text.empty()) {
        // False indicates success, and we must consume the whole token.
        return success();
    }

    if (failed(recover(loc, "index literal too large"))) return failure();
    return std::nullopt;
}

[[nodiscard]] static bool isInexact(APFloat::opStatus status)
{
    using int_t         = std::underlying_type_t<APFloat::opStatus>;
    constexpr auto mask = static_cast<int_t>(APFloat::opStatus::opInexact);
    return (static_cast<int_t>(status) & mask) == mask;
}

std::optional<LogicalResult>
ParseDriver::parseDecimal(Token token, Number &value)
{
    auto [loc, text] = token;

    // Parse the decimal literal.
    llvm::APFloat binary64(llvm::APFloat::IEEEdouble());
    auto maybeFloat = binary64.convertFromString(
        text,
        APFloat::roundingMode::NearestTiesToEven);
    auto error = maybeFloat.takeError();
    if (!error) {
        if (isInexact(maybeFloat.get()))
            emitWarning(token.first, "inexact float literal");

        value = Number(binary64.convertToDouble());
        return success();
    }

    // Who knows why this fails, we can't fix it.
    llvm::consumeError(std::move(error));
    if (failed(recover(loc, "invalid float literal"))) return failure();
    return std::nullopt;
}

std::optional<LogicalResult>
ParseDriver::parseRational(Token token, Number &value)
{
    auto [loc, text] = token;

    // Consume the sign.
    text.consume_front("+");
    auto negate = text.consume_front("-");

    // Parse the binary rational literal.
    Number::mantissa_t mantissa;
    Number::exponent_t exponent = 0UL;
    if (!text.consumeInteger(10, mantissa)
        && (text.empty()
            || (text.consume_front_insensitive("p")
                && !text.consumeInteger(10, exponent) && text.empty()))) {
        // Ensure that the sign is preserved correctly.
        if (mantissa.isNegative())
            mantissa = mantissa.zext(mantissa.getBitWidth() + 1U);
        if (negate) mantissa.negate();

        value = Number(std::move(mantissa), exponent);
        return success();
    }

    // I guess the exponent is stupendously large?
    if (failed(recover(loc, "invalid rational literal"))) return failure();
    return std::nullopt;
}

void ParseDriver::typeDef(ImportLocation nameLoc, StringRef name, TypeExpr type)
{
    if (!type) type = {getErrorType(), nameLoc};

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), type));
    assert(succeeded(ok));
}

void ParseDriver::constDef(
    ImportLocation nameLoc,
    StringRef name,
    ConstExpr value)
{
    if (!value) value = {getErrorLiteral(), nameLoc};

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), value));
    assert(succeeded(ok));
}

void ParseDriver::exprDef(ImportLocation nameLoc, StringRef name, Expr value)
{
    if (!value) value = expr<LiteralOp>(nameLoc, getErrorLiteral());

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), value));
    assert(succeeded(ok));
}

LogicalResult
ParseDriver::argDecl(ImportLocation loc, StringRef name, TypeExpr type)
{
    // NOTE: type can be nullptr and it will still work, no error recovery
    //       necessary.
    type             = {getExpressionType(type), loc};
    const auto opLoc = getLocation(loc, name);
    const auto arg   = getBlock()->addArgument(type, opLoc);
    return define(
        Shadow::Outer,
        Definition(opLoc, llvm::cast<Expression>(arg)));
}

FailureOr<TypeExpr> ParseDriver::resolveType(ImportLocation loc, StringRef name)
{
    const auto sym = resolve(loc, name);
    if (failed(sym)) return failure();
    if (!*sym) return TypeExpr{getErrorType(), loc};
    if (const auto result = (*sym)->dyn_cast<Type>())
        return TypeExpr{result, loc};

    // Definition has the wrong kind.
    auto diag = emitError(getLocation(loc, name))
             << "expected type, but found " << (*sym)->getKind() << " '" << name
             << "'";
    diag.attachNote((*sym)->getLoc()) << "defined here";
    if (failed(recover())) return failure();
    return TypeExpr{getErrorType(), loc};
}

FailureOr<Expr> ParseDriver::resolveExpr(ImportLocation loc, StringRef name)
{
    const auto sym = resolve(loc, name);
    if (failed(sym)) return failure();
    if (!*sym) return expr<LiteralOp>(loc, getErrorLiteral());

    if (const auto result = (*sym)->dyn_cast<Expression>())
        return Expr{result, loc};
    if (const auto result = (*sym)->dyn_cast<LiteralAttr>()) {
        // Materialize the constant here.
        return expr<LiteralOp>(loc, result);
    }
    if (const auto result = (*sym)->dyn_cast<SymbolOpInterface>()) {
        if (const auto staticOp = llvm::dyn_cast<StaticOp>(result)) {
            // Read from the static reference here.
            return expr<ReadOp>(loc, expr<GetStaticOp>(loc, staticOp));
        }
    }

    // Definition has the wrong kind.
    auto diag = emitError(getLocation(loc, name))
             << "expected expression, but found " << (*sym)->getKind() << " '"
             << name << "'";
    diag.attachNote((*sym)->getLoc()) << "defined here";
    if (failed(recover())) return failure();
    return expr<LiteralOp>(loc, getErrorLiteral());
}

LogicalResult ParseDriver::staticDecl(
    ImportLocation nameLoc,
    AccessModifier access,
    StringRef name,
    TypeExpr type,
    ekl::ArrayAttr init)
{
    type = ensure(type);

    // Create a GlobalOp.
    const auto loc = getLocation(nameLoc, name);
    auto refTy     = llvm::dyn_cast<ReferenceType>(type.getValue());
    if (!refTy) {
        if (failed(recover(type, "expected reference type"))) return failure();

        // Recover from this error by supplying an 'in& u8' reference type.
        refTy = llvm::cast<ReferenceType>(
            ReferenceType::get(ArrayType::get(getIntegerType(8U, false))));
    }

    auto staticOp = m_builder.create<StaticOp>(loc, name, refTy, access, init);

    // Define the global symbol.
    return define(
        Shadow::None,
        llvm::cast<SymbolOpInterface>(staticOp.getOperation()));
}

LogicalResult ParseDriver::outStmt(
    ImportLocation loc,
    ImportLocation nameLoc,
    StringRef name,
    Expr value)
{
    const auto resolveRef = [&](ImportLocation nameLoc,
                                StringRef name) -> FailureOr<Expr> {
        const auto sym = resolve(nameLoc, name);
        if (failed(sym)) return failure();
        if (!*sym) return expr<LiteralOp>(nameLoc, getErrorLiteral());

        if (const auto result = (*sym)->dyn_cast<Expression>())
            return Expr{result, nameLoc};
        if (const auto result = (*sym)->dyn_cast<SymbolOpInterface>()) {
            if (const auto staticOp = llvm::dyn_cast<StaticOp>(result)) {
                // Read from the static reference here.
                return expr<GetStaticOp>(nameLoc, staticOp);
            }
        }

        // Definition has the wrong kind.
        auto diag = emitError(getLocation(nameLoc, name))
                 << "expected expression, but found " << (*sym)->getKind()
                 << " '" << name << "'";
        diag.attachNote((*sym)->getLoc()) << "defined here";
        if (failed(recover())) return failure();
        return expr<LiteralOp>(loc, getErrorLiteral());
    };

    // Resolve the name of the target reference.
    const auto ref = resolveRef(nameLoc, name);
    if (failed(ref)) return ref;

    // Write to that reference.
    create<WriteOp>(loc, *ref, value);
    return success();
}

LogicalResult ParseDriver::beginKernel(ImportLocation nameLoc, StringRef name)
{
    // Create a KernelOp and enter it.
    const auto loc = getLocation(nameLoc, name);
    auto kernelOp  = m_builder.create<KernelOp>(loc, name);
    m_builder.setInsertionPointToStart(kernelOp.getBody());
    LLVM_DEBUG(
        llvm::dbgs()
        << "[Parser] begin " << KernelOp::getOperationName() << "\n");

    // Define the kernel symbol.
    return define(
        Shadow::None,
        llvm::cast<SymbolOpInterface>(kernelOp.getOperation()));
}

void ParseDriver::beginAssoc()
{
    // Create the AssocOp and enter its map body.
    auto assocOp = m_builder.create<AssocOp>(getUnknownLoc());
    m_builder.setInsertionPointToStart(assocOp.getMap());
    LLVM_DEBUG(
        llvm::dbgs()
        << "[Parser] begin " << AssocOp::getOperationName() << "\n");
}

void ParseDriver::beginReduce(Expr array, Expr init)
{
    auto reduceOp = m_builder.create<ReduceOp>(
        getUnknownLoc(),
        array,
        FunctorBuilderRef{},
        Type{},
        init);
    m_builder.setInsertionPointToStart(reduceOp.getReduction());
    LLVM_DEBUG(
        llvm::dbgs()
        << "[Parser] begin " << ReduceOp::getOperationName() << "\n");
}

void ParseDriver::beginZip(ArrayRef<Expr> operands)
{
    // Create the ZipOp and enter its combinator body.
    auto zipOp = create<ZipOp>(ImportLocation{}, operands);
    m_builder.setInsertionPointToStart(zipOp.getCombinator());
    LLVM_DEBUG(
        llvm::dbgs() << "[Parser] begin " << ZipOp::getOperationName() << "\n");
}

void ParseDriver::beginIf(Expr condition)
{
    condition = ensure(condition);

    // Create the IfOp and enter its then block.
    auto ifOp = m_builder.create<IfOp>(getUnknownLoc(), condition, Type{});
    m_builder.setInsertionPointToStart(ifOp.getThenBranch());
    LLVM_DEBUG(
        llvm::dbgs() << "[Parser] begin " << IfOp::getOperationName() << "\n");
}

FailureOr<Expr> ParseDriver::call(
    ImportLocation loc,
    ImportLocation nameLoc,
    StringRef name,
    ArrayRef<Expr> arguments)
{
    const auto functionStyleCast = [&](Expr input, Type output) -> Expr {
        beginZip({input});
        return {
            yieldAndEnd<ZipOp>(
                {expr<CoerceOp>(
                     loc,
                     getOp<ZipOp>().getCombinator()->getArgument(0),
                     output),
                 loc}),
            loc};
    };

    // Allow function-style casting.
    if (const auto sym = lookup(name)) {
        if (const auto type = sym->dyn_cast<Type>())
            return functionStyleCast(arguments.front(), type);

        // Other symbols are not callable.
        auto diag = emitError(nameLoc) << "can't call " << sym->getKind();
        diag.attachNote(sym->getLoc()) << "defined here";
        return recover();
    }

    if (name == "log") {
        // TODO: Implement.
        return arguments[0];
    } else if (name == "sum") {
        // TODO: Implement.
        return arguments[0];
    }

    if (failed(recover(nameLoc, "unknown function"))) return failure();
    return expr<LiteralOp>(loc, getErrorLiteral());
}

FailureOr<TypeExpr>
ParseDriver::refType(ImportLocation loc, ReferenceKind kind, TypeExpr pointee)
{
    pointee = ensure(pointee);

    auto arrayTy = llvm::dyn_cast<ArrayType>(pointee.getValue());
    if (!arrayTy) {
        if (failed(recover(pointee, "expected array type"))) return failure();

        // Recover from this error by supplying a 'Number[]' array type.
        arrayTy = ArrayType::get(getNumberType());
    }

    return TypeExpr{ReferenceType::get(arrayTy, kind), loc};
}

FailureOr<TypeExpr> ParseDriver::typeCtor(
    ImportLocation loc,
    ImportLocation nameLoc,
    StringRef name,
    ArrayRef<ConstExpr> params)
{
    if (name == "index") {
        if (params.size() != 1) {
            if (failed(recover(loc, "expected 1 type parameter")))
                return failure();
            return TypeExpr{getErrorType(), loc};
        }

        const auto boundAttr =
            llvm::dyn_cast<ekl::IndexAttr>(params.front().getValue());
        if (!boundAttr) {
            auto diag = emitError(params.front().getLoc())
                     << "expected index value";
            diag.attachNote(getLocation(loc))
                << "while constructing index type";
            if (failed(recover())) return failure();
            return TypeExpr{getErrorType(), loc};
        }

        return TypeExpr{getIndexType(boundAttr.getValue()), loc};
    }

    if (failed(recover(nameLoc, "unknown type constructor"))) return failure();
    return TypeExpr{getErrorType(), loc};
}

FailureOr<TypeExpr>
ParseDriver::arrayType(ImportLocation loc, TypeExpr scalar, Extents extents)
{
    scalar = ensure(scalar);

    auto scalarTy = llvm::dyn_cast<ScalarType>(scalar.getValue());
    if (!scalarTy) {
        if (failed(recover(scalar, "expected scalar type"))) return failure();

        // Recover from this error by supplying the number type.
        scalarTy = getNumberType();
    }

    if (hasNoElements(extents.getValue())) {
        if (failed(recover(extents, "array is empty"))) return failure();

        // Recover from this error by making those extents 1.
        for (auto &extent : extents.getValue())
            if (extent == 0) extent = 1UL;
    }

    return TypeExpr{ArrayType::get(scalarTy, extents.getValue()), loc};
}

void ParseDriver::beginConstexpr()
{
    // Start a constexpr scope by beginning an AssocOp.
    auto cexprOp = m_builder.create<ConstexprOp>(getUnknownLoc());
    m_builder.setInsertionPointToStart(cexprOp.getBody());
}

FailureOr<ConstExpr> ParseDriver::endConstexpr(Expr expr)
{
    // Finish the constexpr scope by closing the ConstexprOp.
    create<YieldOp>(expr.getLoc(), expr);
    auto cexprOp   = end<ConstexprOp>();
    const auto loc = getLocation(expr.getLoc());
    cexprOp->setLoc(loc);

    // Install a temporary diagnostic handler so that the user is not confused
    // as to where the type-checking and verification errors come from.
    ScopedDiagnosticHandler diagHandler(getContext(), [&](Diagnostic &diag) {
        diag.attachNote(loc) << "while evaluating this constant expression";
        return failure();
    });

    const auto result = [&]() -> FailureOr<LiteralAttr> {
        // The ConstexprOp and its descendants must verify and type check.
        if (failed(verify(cexprOp)) || failed(typeCheck(cexprOp)))
            return failure();

        // Apply all of our known constant evaluation patterns.
        if (failed(applyPatternsGreedily(
                cexprOp.getBodyRegion(),
                *m_constexprPatterns)))
            return failure();

        // Fold the ConstexprOp itself and try to get the resulting literal.
        const auto attr = cexprOp.fold(ConstexprOp::FoldAdaptor({}, cexprOp))
                              .dyn_cast<Attribute>();
        if (const auto literal = llvm::dyn_cast_if_present<LiteralAttr>(attr)) {
            // The ConstexprOp is not needed anymore.
            cexprOp.erase();
            return literal;
        }

        return failure();
    }();
    if (succeeded(result)) return ConstExpr{*result, expr.getLoc()};

    LLVM_DEBUG(
        llvm::dbgs() << "[Parser] failed to fold constant expression:\n";
        cexprOp.print(llvm::dbgs(), OpPrintingFlags{}.printGenericOpForm());
        llvm::dbgs() << "\n";);

    cexprOp.erase();
    if (failed(recover(
            expr.getLoc(),
            "expression did not evaluate to a constant")))
        return failure();
    return ConstExpr{LiteralAttr(getErrorLiteral()), expr.getLoc()};
}

FailureOr<Extents>
ParseDriver::extents(ImportLocation loc, ArrayRef<ConstExpr> exprs)
{
    // Make the result extents, accumulating all errors in the process.
    SmallVector<ImportLocation> errors;
    SmallVector<extent_t> result;
    for (auto expr : exprs) {
        auto extent =
            llvm::TypeSwitch<LiteralAttr, std::optional<extent_t>>(expr)
                .Case([](ekl::IndexAttr indexAttr) -> std::optional<extent_t> {
                    return indexAttr.getValue();
                })
                .Case([](ekl::IntegerAttr intAttr) -> std::optional<extent_t> {
                    const auto value = intAttr.getValue();
                    if (intAttr.getType().isSigned() && value.isNegative())
                        return std::nullopt;
                    return value.tryZExtValue();
                })
                .Case([](NumberAttr numAttr) -> std::optional<extent_t> {
                    return numAttr.getValue().tryGetUInt();
                })
                .Default(std::optional<extent_t>{});
        if (!extent) {
            extent.emplace(1UL);
            errors.push_back(expr.getLoc());
        }

        result.push_back(*extent);
    }

    // Report all the errors at once.
    for (auto errorLoc : errors)
        if (failed(recover(errorLoc, "value is not a valid array extent")))
            return failure();

    return Extents{std::move(result), loc};
}

Operation *ParseDriver::endImpl()
{
    const auto result = getOp();
    LLVM_DEBUG(llvm::dbgs() << "[Parser] ending " << result->getName() << "\n");
    m_builder.setInsertionPointAfter(result);
    return result;
}

LogicalResult ParseDriver::define(Shadow shadow, Definition def)
{
    assert(!def.isNull());

    // Attempt to define the name.
    std::pair<Definition *, bool> result;
    switch (shadow) {
    case Shadow::None:  result = m_scopes.define(def); break;
    case Shadow::Outer: result = m_scopes.shadowOuter(def); break;
    case Shadow::Inner: m_scopes.shadow(def); return success();
    }
    if (result.second) return success();

    // We found an existing definition that we are not allowed to shadow.
    auto diag = emitError(def.getLoc())
             << def.getKind() << " with name '" << def.getName()
             << "' has already been defined";
    diag.attachNote(result.first->getLoc()) << "previous definition is here";
    return recover();
}

const Definition *ParseDriver::lookup(StringRef name)
{
    if (const auto sym = m_scopes.lookup(name)) return sym;

    if (const auto builtin = lookupBuiltin(name)) {
        // Cache the definition of this builtin.
        const auto [sym, ok] = getFileScope().insert(*builtin);
        assert(ok);
        return sym;
    }

    return nullptr;
}

std::optional<Definition> ParseDriver::lookupBuiltin(StringRef name)
{
    const auto loc =
        NameLoc::get(m_builder.getStringAttr(name), m_builder.getUnknownLoc());

    if (name == "bool") return Definition(loc, getBoolType());
    if (name == "number") return Definition(loc, getNumberType());
    if (name == "index") return Definition(loc, getIndexType());
    if (name == "bf16") return Definition(loc, m_builder.getBF16Type());
    if (name == "f16") return Definition(loc, m_builder.getF16Type());
    if (name == "f32") return Definition(loc, m_builder.getF32Type());
    if (name == "f64") return Definition(loc, m_builder.getF64Type());
    if (name == "f80") return Definition(loc, m_builder.getF80Type());
    if (name == "f128") return Definition(loc, m_builder.getF128Type());
    if (name == "string") return Definition(loc, getStringType());

    if (name.starts_with("si") || name.starts_with("ui")) {
        auto window = name.drop_front(2);
        unsigned bitWidth;
        if (!window.consumeInteger(10U, bitWidth) && window.empty()
            && bitWidth <= mlir::IntegerType::kMaxWidth) {
            const auto isSigned = name.front() == 's';
            return Definition(loc, getIntegerType(bitWidth, isSigned));
        }
    }

    return std::nullopt;
}

FailureOr<const Definition *>
ParseDriver::resolve(ImportLocation nameLoc, StringRef name)
{
    if (const auto sym = lookup(name)) return sym;

    if (failed(recover(
            nameLoc,
            llvm::Twine("unknown symbol '").concat(name).concat("'"))))
        return failure();
    return static_cast<const Definition *>(nullptr);
}
