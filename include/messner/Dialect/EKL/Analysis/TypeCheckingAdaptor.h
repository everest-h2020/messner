/// Declares the TypeCheckingAdaptor.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Casting.h"
#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h"

#include <optional>

namespace mlir::ekl {

/// Return type of an operation that checks the premise of an implication.
///
/// This type supports an assignment-if pattern, where contextual conversion to
/// @c true indicates a contradiction that invalidates the implication. However,
/// a contradiction that is not classified as an error might still not contain
/// a diagnostic, so converting the contained value to LogicalResult might yield
/// a success anyway.
struct [[nodiscard]] Contradiction : std::optional<InFlightDiagnostic> {
    using base = std::optional<InFlightDiagnostic>;

    using base::base;

    /// Obtains the no contradiction result.
    static Contradiction none() { return std::nullopt; }
    /// Obtains a Contradiction without an error.
    static Contradiction indeterminate()
    {
        return Contradiction(std::in_place);
    }
    /// Obtains a Contradiction with an error.
    static Contradiction error(InFlightDiagnostic error)
    {
        return Contradiction(std::in_place, std::move(error));
    }

    /// Ignores the contained error, if any.
    void ignore()
    {
        if (this->has_value()) this->value().abandon();
    }

    /// Attaches a note to the contained error, if any.
    ///
    /// @param              location    Optional location for the diagnostic.
    /// @param              fn          Function to populate the note.
    ///
    /// @return Contradiction.
    Contradiction explain(
        std::optional<Location> location,
        function_ref<void(Diagnostic &)> fn) &&
    {
        return std::move(*this).explainImpl(location, fn);
    }
    /// Attaches a note to the contained error, if any.
    ///
    /// @param              location    Optional location for the diagnostic.
    /// @param              what        Message to attach.
    ///
    /// @return Contradiction.
    Contradiction
    explain(std::optional<Location> location, const llvm::Twine &what) &&
    {
        return std::move(*this).explainImpl(location, [&](Diagnostic &diag) {
            diag << what;
        });
    }
    /// Attaches a note to the contained error, if any.
    ///
    /// @param              type       Type to mention.
    ///
    /// @return Contradiction.
    Contradiction explain(Type type) &&
    {
        return std::move(*this).explainImpl(
            std::nullopt,
            [&](Diagnostic &diag) { diag << "found type: " << type; });
    }
    /// Attaches a note to the contained error, if any.
    ///
    /// @param              types       Types to mention.
    ///
    /// @return Contradiction.
    Contradiction explain(ArrayRef<Type> types) &&
    {
        return std::move(*this).explainImpl(
            std::nullopt,
            [&](Diagnostic &diag) {
                diag << "found types: ";
                llvm::interleaveComma(types, diag);
            });
    }
    /// Attaches a note to the contained error, if any.
    ///
    /// @param              expr        Expression to mention.
    ///
    /// @return Contradiction.
    Contradiction explain(Expression expr) &&
    {
        return std::move(*this).explain(expr.getLoc(), "from this expression");
    }
    /// Attaches a note to the contained error, if any.
    ///
    /// @pre    @p exprs only contains Expression elements.
    ///
    /// @param              exprs       Expressions to mention.
    ///
    /// @return Contradiction.
    Contradiction explain(ValueRange exprs) &&
    {
        explainImpl([&](Diagnostic &diag) {
            for (auto expr : exprs)
                diag.attachNote(expr.getLoc()) << "from this expression";
        });
        return std::move(*this);
    }

    /// Obtains a LogicalResult that indicates whether an error is contained.
    /*implicit*/ operator LogicalResult() const
    {
        return success(!this->has_value() || succeeded(this->value()));
    }
    /// Obtains the contained error, if any.
    /*implicit*/ operator InFlightDiagnostic()
    {
        return std::move(*this).value_or(InFlightDiagnostic{});
    }

private:
    void explainImpl(auto fn)
    {
        if (this->has_value() && failed(value()))
            fn(*(this->value().getUnderlyingDiagnostic()));
    }
    Contradiction explainImpl(std::optional<Location> location, auto fn) &&
    {
        explainImpl([&](Diagnostic &diag) { fn(diag.attachNote(location)); });
        return std::move(*this);
    }
};

/// Provides an adaptor around an AbstractTypeChecker bound to some
/// TypeCheckOpInterface.
///
/// The adaptor defines some convenience methods to perform common type checking
/// tasks on an operation, which automatically generate error diagnostics when
/// necessary.
struct TypeCheckingAdaptor : AbstractTypeChecker {
    /// Initializes a TypeCheckingAdaptor using @p impl for @p parent .
    explicit TypeCheckingAdaptor(
        AbstractTypeChecker &impl,
        TypeCheckOpInterface parent)
            : m_impl(impl),
              m_parent(parent)
    {}

    /// Gets the AbstractTypeChecker.
    [[nodiscard]] AbstractTypeChecker &getImpl() { return m_impl; }
    /// Gets the AbstractTypeChecker.
    [[nodiscard]] const AbstractTypeChecker &getImpl() const { return m_impl; }
    /// Gets the parent operation
    [[nodiscard]] TypeCheckOpInterface getParent() const { return m_parent; }

    /// @copydoc AbstractTypeChecker::getType(Expression)
    [[nodiscard]] virtual Type getType(Expression expr) const override
    {
        return getImpl().getType(expr);
    }
    /// @copydoc AbstractTypeChecker::refineBound(Expression, Type)
    virtual LogicalResult refineBound(Expression expr, Type incoming) override
    {
        return getImpl().refineBound(expr, incoming);
    }
    /// @copydoc AbstractTypeChecker::meetBound(Expression)
    virtual LogicalResult meetBound(Expression expr, Type incoming) override
    {
        return getImpl().meetBound(expr, incoming);
    }
    /// @copydoc AbstractTypeChecker::invalidate(Operation *)
    virtual void invalidate(Operation *op) override
    {
        return getImpl().invalidate(op);
    }

    // TODO: Document all these.

    template<type_constraint ResultType>
    Contradiction
    require(Type type, ResultType &result, const llvm::Twine &what) const
    {
        if (!type) {
            result = ResultType{};
            return Contradiction::indeterminate();
        }
        if ((result = llvm::dyn_cast<ResultType>(type)))
            return Contradiction::none();
        return Contradiction(emitError() << "expected " << what).explain(type);
    }

    template<type_constraint ResultType>
    Contradiction
    require(Expression expr, ResultType &result, const llvm::Twine &what) const
    {
        return require(getType(expr), result, what).explain(expr);
    }

    Contradiction require(Type result, Type supertype) const
    {
        if (!result) return Contradiction::indeterminate();
        if (isSubtype(result, supertype)) return Contradiction::none();
        return Contradiction(
            emitError() << result << " is not a subtype of " << supertype);
    }

    Contradiction require(Expression expr, Type supertype, Type &result) const
    {
        result = getType(expr);
        return require(result, supertype).explain(expr);
    }

    Contradiction unify(ArrayRef<Type> types, Type &result) const;

    Contradiction unify(ValueRange exprs, Type &result) const;

    template<type_constraint ResultType>
    Contradiction
    unify(ArrayRef<Type> types, ResultType &result, const llvm::Twine &what)
        const
    {
        Type unified;
        if (auto contra = unify(types, unified)) return contra;
        return require(unified, result, what)
            .explain([&](Diagnostic &diag) {
                diag << "after unifying to " << unified;
            })
            .explain(types);
    }

    template<type_constraint ResultType>
    Contradiction
    unify(ValueRange exprs, ResultType &result, const llvm::Twine &what) const
    {
        const auto types = getTypes(exprs);
        return unify<ResultType>(types, result, what).explain(exprs);
    }

    Contradiction
    broadcast(ArrayRef<Type> types, SmallVectorImpl<extent_t> &extents) const;

    Contradiction
    broadcast(ValueRange exprs, SmallVectorImpl<extent_t> &extents) const;

    Contradiction
    broadcast(Type type, ExtentRange extents, ArrayType &result) const;

    Contradiction
    broadcast(Expression expr, ExtentRange extents, ArrayType &result) const
    {
        return broadcast(getType(expr), extents, result).explain(expr);
    }

    Contradiction broadcast(MutableArrayRef<Type> types) const;

    Contradiction
    broadcast(ValueRange exprs, SmallVectorImpl<Type> &result) const;

    Contradiction coerce(Type type, Type to) const
    {
        if (!type) return Contradiction::indeterminate();
        if (ekl::canCoerce(type, to)) return Contradiction::none();
        return emitError() << "can't coerce " << type << " to " << to;
    }

    Contradiction coerce(Expression expr, Type to) const
    {
        return coerce(getType(expr), to).explain(expr);
    }

    Contradiction broadcastAndUnify(ValueRange exprs, Type &result) const;

    InFlightDiagnostic emitError() const
    {
        return mlir::emitError(getParent().getLoc());
    }

private:
    Contradiction unifyImpl(SmallVectorImpl<Type> &types, Type &result) const;

    AbstractTypeChecker &m_impl;
    TypeCheckOpInterface m_parent;
};

} // namespace mlir::ekl
