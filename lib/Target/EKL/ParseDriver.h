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
    [[nodiscard]] UnknownLoc getUnknownLoc() const
    {
        return UnknownLoc::get(getContext());
    }

public:
    //===------------------------------------------------------------------===//
    // Error handling
    //===------------------------------------------------------------------===//

    [[nodiscard]] bool hasWarnings() const { return m_hasWarnings; }

    void emitWarning(const llvm::Twine &msg);
    void emitWarning(ImportLocation where, const llvm::Twine &msg);

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

    LogicalResult recover();
    LogicalResult recover(ImportLocation where, const llvm::Twine &msg)
    {
        emitError(where, msg);
        return recover();
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

    void typeDef(ImportLocation nameLoc, StringRef name, TypeExpr type);
    void constDef(ImportLocation nameLoc, StringRef name, ConstExpr value);
    void exprDef(ImportLocation nameLoc, StringRef name, Expr value);
    LogicalResult argDecl(ImportLocation loc, StringRef name, TypeExpr type);

    FailureOr<TypeExpr> resolveType(ImportLocation loc, StringRef name);
    FailureOr<Expr> resolveExpr(ImportLocation loc, StringRef name);

public:
    //===------------------------------------------------------------------===//
    // Operation factories
    //===------------------------------------------------------------------===//

    LogicalResult staticDecl(
        ImportLocation loc,
        AccessModifier access,
        StringRef name,
        TypeExpr type,
        ekl::ArrayAttr init = {});

    LogicalResult outStmt(
        ImportLocation loc,
        ImportLocation nameLoc,
        StringRef name,
        Expr value);

    LogicalResult beginKernel(ImportLocation nameLoc, StringRef name);
    void endKernel() { end<KernelOp>(); }

    void beginAssoc();
    Expr endAssoc(ImportLocation loc, Expr yield)
    {
        getOp()->setLoc(getLocation(loc));
        return {yieldAndEnd<AssocOp>(yield), loc};
    }

    Expr reduce(ImportLocation loc, StringRef op, Expr array)
    {
        return expr<ReduceOp>(
            loc,
            array,
            getFunctorBuilder(
                OperationName(op, getContext()),
                getLocation(loc)));
    }
    void beginReduce(Expr array, Expr init);
    Expr endReduce(ImportLocation loc, Expr yield)
    {
        getOp()->setLoc(getLocation(loc));
        return {yieldAndEnd<ReduceOp>(yield), loc};
    }

    void beginZip(ArrayRef<Expr> operands);
    Expr endZip(ImportLocation loc, Expr yield)
    {
        getOp()->setLoc(getLocation(loc));
        return {yieldAndEnd<ZipOp>(yield), loc};
    }

    void beginIf(Expr condition);
    void beginElse(Expr trueValue)
    {
        create<YieldOp>(trueValue.getLoc(), trueValue);
        m_builder.setInsertionPointToStart(getOp<IfOp>().getElseBranch());
    }
    Expr endElse(ImportLocation loc, Expr falseValue)
    {
        getOp()->setLoc(getLocation(loc));
        return {yieldAndEnd<IfOp>(falseValue), loc};
    }

    template<class Op>
    Expr expr(ImportLocation loc, auto &&...args)
    {
        return {
            llvm::cast<Expression>(
                create<Op>(loc, std::forward<decltype(args)>(args)...)
                    ->getResult(0)),
            loc};
    }

    FailureOr<Expr> call(
        ImportLocation loc,
        ImportLocation nameLoc,
        StringRef name,
        ArrayRef<Expr> arguments);

public:
    //===------------------------------------------------------------------===//
    // Type expressions
    //===------------------------------------------------------------------===//

    FailureOr<TypeExpr>
    refType(ImportLocation loc, ReferenceKind kind, TypeExpr pointee);
    FailureOr<TypeExpr>
    arrayType(ImportLocation loc, TypeExpr scalar, Extents extents);

public:
    //===------------------------------------------------------------------===//
    // Constant expressions
    //===------------------------------------------------------------------===//

    void beginConstexpr();
    FailureOr<ConstExpr> endConstexpr(Expr expr);

    FailureOr<Extents> extents(ImportLocation loc, ArrayRef<ConstExpr> exprs);

private:
    Scope &getFileScope() { return m_scopes.back(); }
    Block *getBlock() { return m_builder.getInsertionBlock(); }
    template<class Op = Operation *>
    Op getOp()
    {
        return llvm::cast<Op>(getBlock()->getParentOp());
    }

    Operation *endImpl();
    template<class Op = Operation *>
    Op end()
    {
        return llvm::cast<Op>(endImpl());
    }
    template<class Op = Operation *>
    Expression yieldAndEnd(Expr expr)
    {
        create<YieldOp>(expr.getLoc(), expr);
        auto op = end<Op>();
        return llvm::cast<Expression>(op->getResult(0));
    }

    template<class Op>
    Op create(ImportLocation loc, auto &&...args)
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
            getLocation(loc),
            mapArg(std::forward<decltype(args)>(args))...);
    }

    SmallVector<Value> unpack(ArrayRef<Expr> exprs);

    LogicalResult define(Shadow shadow, Definition def);
    const Definition *lookup(StringRef name);
    std::optional<Definition> lookupBuiltin(StringRef name);
    FailureOr<const Definition *>
    resolve(ImportLocation nameLoc, StringRef name);

    std::shared_ptr<llvm::SourceMgr> m_sourceMgr;
    StringAttr m_filename;
    StringRef m_source;
    bool m_hasWarnings;
    unsigned m_numErrors;
    OpBuilder m_builder;
    OwningOpRef<ProgramOp> m_result;
    Scopes m_scopes;
    std::unique_ptr<FrozenRewritePatternSet> m_constexprPatterns;
};

} // namespace mlir::ekl
