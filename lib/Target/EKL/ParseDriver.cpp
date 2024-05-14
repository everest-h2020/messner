/// Implements the ParseDriver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "ParseDriver.h"

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
    m_builder.setInsertionPointToStart(&m_result->getBody().front());

    // Open the file-level scope and add the builtins.
    m_scopes.push(llvm::SMLoc::getFromPointer(m_source.data()));

    // Initialize the constexpr evaluation patterns.
    RewritePatternSet constexprPatterns(getContext());
    for (auto *dialect : getContext()->getLoadedDialects())
        dialect->getCanonicalizationPatterns(constexprPatterns);
    for (auto &op : getContext()->getRegisteredOperations())
        op.getCanonicalizationPatterns(constexprPatterns, getContext());
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
    }

    if (hasWarnings()) emitWarning("parsing completed with warnings");

    m_scopes.clear();
    return std::move(m_result);
}

Location ParseDriver::getLocation(ImportLocation loc) const
{
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
    ArrayRef<llvm::SMRange> ranges = {where};
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

LogicalResult ParseDriver::recoverFromError()
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

    if (failed(recoverFromError(loc, "index literal too large")))
        return failure();
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
    if (failed(recoverFromError(loc, "invalid float literal")))
        return failure();
    return std::nullopt;
}

std::optional<LogicalResult>
ParseDriver::parseRational(Token token, Number &value)
{
    auto [loc, text] = token;

    // Parse the binary rational literal.
    Number::mantissa_t mantissa;
    Number::exponent_t exponent = 0UL;
    if (!text.consumeInteger(10, mantissa)
        && (text.empty()
            || (text.consume_front_insensitive("p")
                && !text.consumeInteger(10, exponent) && text.empty()))) {
        // Ensure that the sign is preserved correctly.
        if (!token.second.starts_with("-") && mantissa.isNegative())
            mantissa = mantissa.zext(mantissa.getBitWidth() + 1U);

        value = Number(std::move(mantissa), exponent);
        return success();
    }

    // I guess the exponent is stupendously large?
    if (failed(recoverFromError(loc, "invalid rational literal")))
        return failure();
    return std::nullopt;
}

void ParseDriver::defineType(ImportLocation nameLoc, StringRef name, Type type)
{
    if (!type) type = getErrorType();

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), type));
    assert(succeeded(ok));
}

void ParseDriver::defineConst(
    ImportLocation nameLoc,
    StringRef name,
    LiteralAttr value)
{
    if (!value) value = getErrorLiteral();

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), value));
    assert(succeeded(ok));
}

void ParseDriver::defineExpr(
    ImportLocation nameLoc,
    StringRef name,
    Expression value)
{
    if (!value) value = expr<LiteralOp>(nameLoc, getErrorLiteral());

    const auto ok =
        define(Shadow::Inner, Definition(getLocation(nameLoc, name), value));
    assert(succeeded(ok));
}

LogicalResult
ParseDriver::defineArg(ImportLocation nameLoc, StringRef name, Type type)
{
    // NOTE: type can be nullptr and it will still work, no error recovery
    //       necessary.
    type           = getExpressionType(type);
    const auto loc = getLocation(nameLoc, name);
    const auto arg = getBlock()->addArgument(type, loc);
    return define(Shadow::Outer, Definition(loc, llvm::cast<Expression>(arg)));
}

FailureOr<Type> ParseDriver::resolveType(ImportLocation nameLoc, StringRef name)
{
    const auto sym = resolve(nameLoc, name);
    if (failed(sym)) return failure();
    if (!*sym) return getErrorType();
    if (const auto result = (*sym)->dyn_cast<Type>()) return result;

    // Definition has the wrong kind.
    auto diag = emitError(getLocation(nameLoc, name))
             << "expected type, but found " << (*sym)->getKind() << " '" << name
             << "'";
    diag.attachNote((*sym)->getLoc()) << "defined here";
    if (failed(recoverFromError())) return failure();
    return getErrorType();
}

FailureOr<LiteralAttr>
ParseDriver::resolveConst(ImportLocation nameLoc, StringRef name)
{
    const auto sym = resolve(nameLoc, name);
    if (failed(sym)) return failure();
    if (!*sym) return LiteralAttr(getErrorLiteral());
    if (const auto result = (*sym)->dyn_cast<LiteralAttr>()) return result;

    // Definition has the wrong kind.
    auto diag = emitError(getLocation(nameLoc, name))
             << "expected constant, but found " << (*sym)->getKind() << " '"
             << name << "'";
    diag.attachNote((*sym)->getLoc()) << "defined here";
    if (failed(recoverFromError())) return failure();
    return LiteralAttr(getErrorLiteral());
}

FailureOr<Expression>
ParseDriver::resolveExpr(ImportLocation nameLoc, StringRef name)
{
    const auto sym = resolve(nameLoc, name);
    if (failed(sym)) return failure();
    if (!*sym) return expr<LiteralOp>(nameLoc, getErrorLiteral());

    if (const auto result = (*sym)->dyn_cast<Expression>()) return result;
    if (const auto result = (*sym)->dyn_cast<LiteralAttr>()) {
        // Materialize the constant here.
        return expr<LiteralOp>(nameLoc, result);
    }
    if (const auto result = (*sym)->dyn_cast<SymbolOpInterface>()) {
        if (const auto global = llvm::dyn_cast<GlobalOp>(result)) {
            // Materialize the global reference here.
            return expr<GetGlobalOp>(nameLoc, global);
        }
    }

    // Definition has the wrong kind.
    auto diag = emitError(getLocation(nameLoc, name))
             << "expected expression, but found " << (*sym)->getKind() << " '"
             << name << "'";
    diag.attachNote((*sym)->getLoc()) << "defined here";
    if (failed(recoverFromError())) return failure();
    return expr<LiteralOp>(nameLoc, getErrorLiteral());
}

LogicalResult ParseDriver::global(
    ImportLocation nameLoc,
    StringRef name,
    TypeExpr type,
    InitializerAttr init)
{
    type = ensure(type);

    // Create a GlobalOp.
    const auto loc = getLocation(nameLoc, name);
    auto refTy     = llvm::dyn_cast<ABIReferenceType>(type.getValue());
    if (!refTy) {
        if (failed(recoverFromError(type, "expected ABI reference type")))
            return failure();

        // Recover from this error by supplying an 'in& u8' reference type.
        refTy = llvm::cast<ABIReferenceType>(
            ReferenceType::get(ArrayType::get(getIntegerType(8U, false))));
    }

    auto globalOp = m_builder.create<GlobalOp>(
        loc,
        name,
        refTy,
        init,
        SymbolTable::Visibility::Private);

    // Define the global symbol.
    return define(
        Shadow::None,
        llvm::cast<SymbolOpInterface>(globalOp.getOperation()));
}

LogicalResult ParseDriver::beginKernel(ImportLocation nameLoc, StringRef name)
{
    // Create a KernelOp and enter it.
    const auto loc = getLocation(nameLoc, name);
    auto kernelOp  = m_builder.create<KernelOp>(loc, loc.getName());
    m_builder.setInsertionPointToStart(&kernelOp.getBody().front());

    // Define the kernel symbol.
    return define(
        Shadow::None,
        llvm::cast<SymbolOpInterface>(kernelOp.getOperation()));
}

void ParseDriver::beginIf(ImportLocation introLoc, Expr cond, bool withResult)
{
    cond = ensure(cond);

    // Create the IfOp and enter its then block.
    const auto loc = getLocation(introLoc);
    auto ifOp      = withResult ? m_builder.create<IfOp>(loc, cond, nullptr)
                                : m_builder.create<IfOp>(loc, cond);
    m_builder.setInsertionPointToStart(&ifOp.getThenBranch().front());
}

void ParseDriver::beginElse()
{
    // Create the else branch block and enter it.
    auto ifOp = llvm::cast<IfOp>(getOp());
    m_builder.setInsertionPointToStart(&ifOp.getElseBranch().emplaceBlock());
}

void ParseDriver::beginAssoc(ImportLocation introLoc)
{
    // Create the AssocOp and enter its map body.
    auto assocOp = m_builder.create<AssocOp>(getLocation(introLoc));
    m_builder.setInsertionPointToStart(&assocOp.getMap().front());
}

void ParseDriver::beginZip(ImportLocation introLoc, ArrayRef<Expr> exprs)
{
    // Create the ZipOp and enter its combinator body.
    auto zipOp = create<ZipOp>(introLoc, exprs);
    m_builder.setInsertionPointToStart(&zipOp.getCombinator().front());
}

void ParseDriver::beginReduce(
    ImportLocation introLoc,
    ImportLocation reductionLoc,
    OperationName reductionOp)
{
    // Create the ReduceOp, which will populate its reduction block, then enter
    // its map body.
    auto reduceOp = m_builder.create<ReduceOp>(
        getLocation(introLoc),
        getLocation(reductionLoc),
        reductionOp);
    m_builder.setInsertionPointToStart(&reduceOp.getMap().front());
}

LogicalResult ParseDriver::write(
    ImportLocation opLoc,
    ImportLocation nameLoc,
    StringRef name,
    Expr value)
{
    // Resolve the name of the target reference.
    const auto ref = resolveExpr(nameLoc, name);
    if (failed(ref)) return ref;

    // Write to that reference.
    write(opLoc, {*ref, nameLoc}, value);
    return success();
}

FailureOr<Expression> ParseDriver::call(
    ImportLocation nameLoc,
    StringRef name,
    ArrayRef<Expr> arguments)
{
    const auto functionStyleCast = [&](Expr input, Type output) -> Expression {
        beginZip(input, {input});
        return yieldAndEnd<ZipOp>(
            {expr<CoerceOp>(
                 input,
                 getOp<ZipOp>().getCombinator().getArgument(0),
                 output),
             input.getLoc()});
    };

    // Allow function-style casting.
    if (const auto sym = m_scopes.lookup(name)) {
        if (const auto type = sym->dyn_cast<Type>())
            return functionStyleCast(arguments.front(), type);

        // Other symbols are not callable.
        auto diag = emitError(nameLoc) << "can't call " << sym->getKind();
        diag.attachNote(sym->getLoc()) << "defined here";
        return recoverFromError();
    }

    if (failed(recoverFromError(nameLoc, "unknown function"))) return failure();
    return expr<LiteralOp>(nameLoc, getErrorLiteral());
}

void ParseDriver::pushConstexpr(ImportLocation introLoc)
{
    // Start a constexpr scope by beginning an AssocOp.
    auto cexprOp = create<ConstexprOp>(introLoc);
    m_builder.setInsertionPointToStart(&cexprOp.getExpression().front());
    pushScope(introLoc);
}

FailureOr<LiteralAttr> ParseDriver::popConstexpr(Expr expr)
{
    // Finish the constexpr scope by closing the AssocOp.
    popScope();
    create<YieldOp>(expr.getLoc(), expr);
    auto cexprOp = end<ConstexprOp>();

    // Install a temporary diagnostic handler so that the user is not confused
    // as to where the type-checking and verification errors come from.
    const auto loc = getLocation(expr);
    ScopedDiagnosticHandler diagHandler(getContext(), [&](Diagnostic &diag) {
        diag.attachNote(loc) << "while evaluating this constant expression";
        return failure();
    });

    const auto result = [&]() -> FailureOr<LiteralAttr> {
        // The ConstexprOp and its descendants must verify and type check.
        if (failed(verify(cexprOp)) || failed(typeCheck(cexprOp)))
            return failure();

        // Apply all of our known constant evaluation patterns.
        if (failed(applyPatternsAndFoldGreedily(
                cexprOp.getExpression(),
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
    if (succeeded(result)) return *result;

    LLVM_DEBUG(
        llvm::dbgs() << "[Parser] failed to fold constant expression:\n";
        cexprOp.print(llvm::dbgs(), OpPrintingFlags{}.printGenericOpForm());
        llvm::dbgs() << "\n";);

    cexprOp.erase();
    if (failed(recoverFromError(
            expr.getLoc(),
            "expression did not evaluate to a constant")))
        return failure();
    return LiteralAttr(getErrorLiteral());
}

FailureOr<SmallVector<extent_t>> ParseDriver::extents(ArrayRef<ConstExpr> exprs)
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
        if (failed(recoverFromError(
                errorLoc,
                "value is not a valid array extent")))
            return failure();

    return std::move(result);
}

FailureOr<ReferenceType>
ParseDriver::referenceType(ReferenceKind kind, TypeExpr pointee)
{
    pointee = ensure(pointee);

    auto arrayTy = llvm::dyn_cast<ArrayType>(pointee.getValue());
    if (!arrayTy) {
        if (failed(recoverFromError(pointee, "expected array type")))
            return failure();

        // Recover from this error by supplying a 'Number[]' array type.
        arrayTy = ArrayType::get(getNumberType());
    }

    return ReferenceType::get(arrayTy, kind);
}

FailureOr<ArrayType> ParseDriver::arrayType(TypeExpr scalar, Extents extents)
{
    scalar = ensure(scalar);

    auto scalarTy = llvm::dyn_cast<ScalarType>(scalar.getValue());
    if (!scalarTy) {
        if (failed(recoverFromError(scalar, "expected scalar type")))
            return failure();

        // Recover from this error by supplying the number type.
        scalarTy = getNumberType();
    }

    if (hasNoElements(extents.getValue())) {
        if (failed(recoverFromError(extents, "array is empty")))
            return failure();

        // Recover from this error by making those extents 1.
        for (auto &extent : extents.getValue())
            if (extent == 0) extent = 1UL;
    }

    return ArrayType::get(scalarTy, extents.getValue());
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
    return recoverFromError();
}

FailureOr<const Definition *>
ParseDriver::resolve(ImportLocation nameLoc, StringRef name)
{
    if (const auto sym = m_scopes.lookup(name)) return sym;

    if (const auto builtin = resolveBuiltin(name)) {
        // Cache the definition of this builtin.
        const auto [sym, ok] = getFileScope().insert(*builtin);
        assert(ok);
        return sym;
    }

    if (failed(recoverFromError(
            nameLoc,
            llvm::Twine("unknown symbol '").concat(name).concat("'"))))
        return failure();
    return static_cast<const Definition *>(nullptr);
}

std::optional<Definition> ParseDriver::resolveBuiltin(StringRef name)
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
