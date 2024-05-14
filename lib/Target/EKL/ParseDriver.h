/// Declares the ParseDriver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "Definitions.h"
#include "ImportLocation.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

#include "llvm/Support/SourceMgr.h"

#include <memory>
#include <utility>

namespace mlir::ekl {

using Token = std::pair<ImportLocation, StringRef>;

//===----------------------------------------------------------------------===//
// Semantic types
//===----------------------------------------------------------------------===//

template<class T>
struct Semantic {
    /*implicit*/ Semantic() = default;
    /*implicit*/ Semantic(T value, ImportLocation loc)
            : m_value(value),
              m_loc(loc)
    {}

    [[nodiscard]] const T &getValue() const { return m_value; }
    [[nodiscard]] T &getValue() { return m_value; }
    [[nodiscard]] ImportLocation getLoc() const { return m_loc; }

    /*implicit*/ operator bool() const { return !!getValue(); }
    /*implicit*/ operator const T &() const { return getValue(); }
    /*implicit*/ operator ImportLocation() const { return getLoc(); }

private:
    T m_value;
    ImportLocation m_loc;
};

using Expr          = Semantic<Expression>;
using ExprList      = SmallVector<Expr>;
using TypeExpr      = Semantic<Type>;
using ConstExpr     = Semantic<LiteralAttr>;
using ConstExprList = SmallVector<ConstExpr>;
using Extents       = Semantic<SmallVector<extent_t>>;

template<class T>
[[nodiscard]] static SmallVector<T> append(SmallVector<T> head, T tail)
{
    head.emplace_back(std::move(tail));
    return head;
}

//===----------------------------------------------------------------------===//
// ParseDriver
//===----------------------------------------------------------------------===//

enum class Shadow { None, Outer, Inner };

struct ParseDriver {
    explicit ParseDriver(
        MLIRContext *context,
        std::shared_ptr<llvm::SourceMgr> sourceMgr);

    [[nodiscard]] llvm::SourceMgr &getSourceMgr() const { return *m_sourceMgr; }
    [[nodiscard]] StringAttr getFilename() const { return m_filename; }
    [[nodiscard]] StringRef getSource() const { return m_source; }
    [[nodiscard]] OwningOpRef<ProgramOp> takeResult();

public:
    //===------------------------------------------------------------------===//
    // Infallible MLIR factories
    //===------------------------------------------------------------------===//

    [[nodiscard]] MLIRContext *getContext() const
    {
        return m_builder.getContext();
    }

    [[nodiscard]] ExpressionType getExpressionType(Type bound = {}) const
    {
        return ExpressionType::get(getContext(), bound);
    }
    [[nodiscard]] NumberType getNumberType() const
    {
        return NumberType::get(getContext());
    }
    [[nodiscard]] ekl::IndexType
    getIndexType(extent_t upperBound = ekl::IndexType::kUnbounded) const
    {
        return ekl::IndexType::get(getContext(), upperBound);
    }
    [[nodiscard]] ekl::IntegerType
    getIntegerType(unsigned bitWidth, bool isSigned = true) const
    {
        return ekl::IntegerType::get(getContext(), bitWidth, isSigned);
    }
    [[nodiscard]] BoolType getBoolType() const
    {
        return BoolType::get(getContext());
    }
    [[nodiscard]] StringType getStringType() const
    {
        return StringType::get(getContext());
    }
    [[nodiscard]] ErrorType getErrorType() const
    {
        return ErrorType::get(getContext());
    }

    [[nodiscard]] BoolAttr getLiteral(bool value) const
    {
        return BoolAttr::get(getContext(), value);
    }
    [[nodiscard]] ekl::IndexAttr getLiteral(extent_t value) const
    {
        return ekl::IndexAttr::get(getContext(), value);
    }
    [[nodiscard]] NumberAttr getLiteral(Number value) const
    {
        return NumberAttr::get(getContext(), value);
    }
    [[nodiscard]] StringAttr getLiteral(StringRef value) const
    {
        return StringAttr::get(getContext(), value);
    }
    [[nodiscard]] IdentityAttr getIdentityLiteral() const
    {
        return IdentityAttr::get(getContext());
    }
    [[nodiscard]] ExtentAttr getExtentLiteral() const
    {
        return ExtentAttr::get(getContext());
    }
    [[nodiscard]] EllipsisAttr getEllipsisLiteral() const
    {
        return EllipsisAttr::get(getContext());
    }
    [[nodiscard]] ErrorAttr getErrorLiteral() const
    {
        return ErrorAttr::get(getContext());
    }

    [[nodiscard]] ImportLocation
    getLocation(std::size_t offset, std::size_t length) const
    {
        assert(offset + length <= m_source.size());

        const auto begin = m_source.begin() + offset;
        return ImportLocation(
            llvm::SMLoc::getFromPointer(begin),
            llvm::SMLoc::getFromPointer(begin + length));
    }
    [[nodiscard]] Location getLocation(ImportLocation loc) const;
    [[nodiscard]] NameLoc getLocation(ImportLocation loc, StringRef name) const
    {
        return NameLoc::get(getLiteral(name), getLocation(loc));
    }

public:
    //===------------------------------------------------------------------===//
    // Error handling
    //===------------------------------------------------------------------===//

    [[nodiscard]] unsigned getNumErrors() const { return m_numErrors; }

    void emitError(const llvm::Twine &msg);
    void emitError(ImportLocation where, const llvm::Twine &msg);
    InFlightDiagnostic emitError(ImportLocation where)
    {
        return emitError(getLocation(where));
    }
    InFlightDiagnostic emitError(Location where)
    {
        return mlir::emitError(where);
    }

    LogicalResult recoverFromError();
    LogicalResult recoverFromError(ImportLocation where, const llvm::Twine &msg)
    {
        emitError(where, msg);
        return recoverFromError();
    }

    [[nodiscard]] TypeExpr ensure(TypeExpr expr)
    {
        if (!expr) expr = {getErrorType(), expr.getLoc()};
        return expr;
    }
    [[nodiscard]] ConstExpr ensure(ConstExpr expr)
    {
        if (!expr) expr = {getErrorLiteral(), expr.getLoc()};
        return expr;
    }
    [[nodiscard]] Expr ensure(Expr expr)
    {
        if (!expr)
            expr = {
                this->expr<LiteralOp>(expr.getLoc(), getErrorLiteral()),
                expr.getLoc()};
        return expr;
    }

public:
    //===------------------------------------------------------------------===//
    // Lexer support methods
    //===------------------------------------------------------------------===//

    std::optional<LogicalResult> parseIndex(Token token, extent_t &value);
    std::optional<LogicalResult> parseDecimal(Token token, Number &value);
    std::optional<LogicalResult> parseRational(Token token, Number &value);

public:
    //===------------------------------------------------------------------===//
    // Lexical scopes
    //===------------------------------------------------------------------===//

    void pushScope(ImportLocation start) { m_scopes.push(start.end); }
    void popScope() { m_scopes.pop(); }

    void defineType(ImportLocation nameLoc, StringRef name, Type type);
    void defineConst(ImportLocation nameLoc, StringRef name, LiteralAttr value);
    void defineExpr(ImportLocation nameLoc, StringRef name, Expression value);
    LogicalResult defineArg(ImportLocation nameLoc, StringRef name, Type type);

    FailureOr<Type> resolveType(ImportLocation nameLoc, StringRef name);
    FailureOr<LiteralAttr> resolveConst(ImportLocation nameLoc, StringRef name);
    FailureOr<Expression> resolveExpr(ImportLocation nameLoc, StringRef name);

public:
    //===------------------------------------------------------------------===//
    // Operation factories
    //===------------------------------------------------------------------===//

    LogicalResult global(
        ImportLocation nameLoc,
        StringRef name,
        TypeExpr type,
        InitializerAttr init = {});

    LogicalResult beginKernel(ImportLocation nameLoc, StringRef name);
    void beginIf(ImportLocation introLoc, Expr cond, bool withResult = false);
    void beginElse();
    void beginElse(Expr trueValue)
    {
        create<YieldOp>(trueValue.getLoc(), trueValue);
        beginElse();
    }
    void beginAssoc(ImportLocation introLoc);
    void beginZip(ImportLocation introLoc, ArrayRef<Expr> exprs);
    void beginReduce(
        ImportLocation introLoc,
        ImportLocation reductionLoc,
        OperationName reductionOp);

    template<class T = Operation *>
    T end()
    {
        const auto result = getOp();
        m_builder.setInsertionPointAfter(result);
        return llvm::cast<T>(result);
    }
    template<class T = Operation *>
    Expression yieldAndEnd(Expr expr)
    {
        create<YieldOp>(expr.getLoc(), expr);
        return llvm::cast<Expression>(end<T>()->getResult(0));
    }

    void write(ImportLocation loc, Expr reference, Expr value)
    {
        create<WriteOp>(loc, reference, value);
    }
    LogicalResult write(
        ImportLocation opLoc,
        ImportLocation nameLoc,
        StringRef name,
        Expr value);

    template<class Op>
    Expression expr(ImportLocation opLoc, auto &&...args)
    {
        return llvm::cast<Expression>(
            create<Op>(opLoc, std::forward<decltype(args)>(args)...)
                ->getResult(0));
    }

    FailureOr<Expression>
    call(ImportLocation nameLoc, StringRef name, ArrayRef<Expr> arguments);

public:
    //===------------------------------------------------------------------===//
    // Constant expressions
    //===------------------------------------------------------------------===//

    void pushConstexpr(ImportLocation introLoc);
    FailureOr<LiteralAttr> popConstexpr(Expr expr);

    FailureOr<SmallVector<extent_t>> extents(ArrayRef<ConstExpr> exprs);

    FailureOr<ReferenceType>
    referenceType(ReferenceKind kind, TypeExpr pointee);
    FailureOr<ArrayType> arrayType(TypeExpr scalar, Extents extents);

private:
    template<class Op>
    Op create(ImportLocation opLoc, auto &&...args)
    {
        const auto mapArg = [&](auto &&arg) -> decltype(auto) {
            using arg_t = std::decay_t<decltype(arg)>;

            if constexpr (
                std::same_as<arg_t, TypeExpr> || std::same_as<arg_t, ConstExpr>
                || std::same_as<arg_t, Expr>)
                return ensure(std::forward<decltype(arg)>(arg));
            else if constexpr (std::convertible_to<
                                   decltype(arg),
                                   ArrayRef<Expr>>) {
                SmallVector<Value> result;
                for (auto &expr : static_cast<ArrayRef<Expr>>(
                         std::forward<decltype(arg)>(arg)))
                    result.push_back(ensure(expr));
                return result;
            } else
                return std::forward<decltype(arg)>(arg);
        };

        return m_builder.create<Op>(
            getLocation(opLoc),
            mapArg(std::forward<decltype(args)>(args))...);
    }

    Scope &getFileScope() { return m_scopes.back(); }
    Block *getBlock() { return m_builder.getInsertionBlock(); }
    template<class Op = Operation *>
    Op getOp()
    {
        return llvm::cast<Op>(getBlock()->getParentOp());
    }

    SmallVector<Value> unpack(ArrayRef<Expr> exprs);

    LogicalResult define(Shadow shadow, Definition def);
    FailureOr<const Definition *>
    resolve(ImportLocation nameLoc, StringRef name);
    std::optional<Definition> resolveBuiltin(StringRef name);

    std::shared_ptr<llvm::SourceMgr> m_sourceMgr;
    StringAttr m_filename;
    StringRef m_source;
    unsigned m_numErrors;
    OpBuilder m_builder;
    OwningOpRef<ProgramOp> m_result;
    Scopes m_scopes;
    std::unique_ptr<FrozenRewritePatternSet> m_constexprPatterns;
};

} // namespace mlir::ekl
