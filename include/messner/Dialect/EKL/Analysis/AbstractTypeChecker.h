/// Declares the AbstractTypeChecker.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/TypeUtils.h"

#include <utility>

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Subtype relation
//===----------------------------------------------------------------------===//
//
// EKL's type checker relies on a default assumption of which types can be
// substituted without changing the semantics of the program. This subtype
// relation is fixed and can be queried by the following methods.

/// Determines whether @p subtype is a supertype of @p supertype .
///
/// The subtype relation in EKL is the transitive closure of the following set
/// of relations:
///
///     T       :> T
///     nullptr :> T
///
///     Number  :> Float
///     Number  :> Integer
///     Number  :> Index
///
/// For all subtypes T, U of Number, the following shall hold:
///
///     u : U -> u in T -> T :> U
///
/// In particular, this means that any type T that accurately represents all
/// objects from type U is a supertype of U.
///
///     Array(T, []) :> T
///
/// For all subtypes of Array, the following shall hold:
///
///     T :> U -> Array(T, x) :> Array(U, x)
///
/// In particular, this means that arrays are covariant.
///
///     Ref(inout, T) :> Ref(out, T)
///     Ref(inout, T) :> Ref(in, T)
///
/// For all subtypes of Reference, the following shall hold:
///
///     T :> U -> Ref(x, T) :> Ref(x, U)
///
/// In particular, this means that references are covariant.
///
/// @param              subtype     The subtype.
/// @param              supertype   The supertype.
///
/// @return @p subtype <: @p supertype
[[nodiscard]] bool isSubtype(Type subtype, Type supertype);

/// Determines whether @p subtype is a proper subtype of @p supertype .
///
/// @param              subtype     The subtype.
/// @param              supertype   The supertype.
///
/// @return @p subtype != @p supertype AND @p subtype <: @p supertype
[[nodiscard]] inline bool isProperSubtype(Type subtype, Type supertype)
{
    return subtype != supertype && isSubtype(subtype, supertype);
}

/// @copydoc isSubtype
[[nodiscard]] inline bool isSupertype(Type supertype, Type subtype)
{
    return isSubtype(subtype, supertype);
}

/// @copydoc isProperSubtype
[[nodiscard]] inline bool isProperSupertype(Type supertype, Type subtype)
{
    return isProperSubtype(subtype, supertype);
}

/// Determines whether @p subtype is a subtype of @p supertype .
///
/// Checks that every value in @p subtype can be represented in @p supertype .
///
/// @pre    `subtype && supertype`
///
/// @param              subtype     The subtype.
/// @param              supertype   The supertype.
///
/// @return @p subtype <: @p supertype
[[nodiscard]] bool
isSubtype(ekl::IntegerType subtype, ekl::IntegerType supertype);

/// @copydoc isSubtype(ekl::IntegerType, ekl::IntegerType)
[[nodiscard]] bool isSubtype(FloatType subtype, FloatType supertype);

/// @copydoc isSubtype(ekl::IntegerType, ekl::IntegerType)
[[nodiscard]] bool isSubtype(ekl::IntegerType subtype, FloatType supertype);

//===----------------------------------------------------------------------===//
// AbstractTypeChecker
//===----------------------------------------------------------------------===//
//
// The base class for implementing concrete type checkers, which is passed to
// ExpressionOp consumers via the typeCheck method.

/// Type of a value that supports type checking.
using Expression = TypedValue<ExpressionType>;

/// Distinguishes the possible outcomes of type bound refinement.
enum class RefinementResult {
    /// No change was made.
    NoChange   = 0b00,
    /// The bound was updated to a more restrictive type.
    Restricted = 0b01,
    /// Attempted to refine the bound to an illegal type.
    Illegal    = 0b10
};

/// Checks whether @p incoming is a valid refinement of @p present .
///
/// The @p incoming bound must be a subtype of the present bound.
///
/// @param              present     The presently deduced bound.
/// @param              incoming    The incoming bound.
///
/// @return RefinementResult.
[[nodiscard]] inline RefinementResult refineBound(Type present, Type incoming)
{
    if (incoming == present) return RefinementResult::NoChange;
    return isSubtype(incoming, present) ? RefinementResult::Restricted
                                        : RefinementResult::Illegal;
}

/// Checks whether @p incoming meets with @p present .
///
/// The @p incoming bound must be equal to @p present , or @p present must be
/// @c nullptr .
///
/// @param              present     The presently deduced bound.
/// @param              incoming    The incoming bound.
///
/// @return RefinementResult.
[[nodiscard]] inline RefinementResult meetBound(Type present, Type incoming)
{
    // If no bound is yet present, use the IR one.
    if (incoming == present || !incoming) return RefinementResult::NoChange;
    if (!present) return RefinementResult::Restricted;
    return RefinementResult::Illegal;
}

/// Gets the owner of @p expr .
///
/// @param              expr    Expression.
///
/// @return Defining op of @p expr , or immediate parent.
[[nodiscard]] inline Operation *getOwner(Expression expr)
{
    if (const auto result = llvm::dyn_cast<OpResult>(expr))
        return result.getOwner();
    const auto argument = llvm::cast<BlockArgument>(expr);
    return argument.getOwner()->getParentOp();
}

/// Abstract base class for the EKL type checker.
///
/// The EKL type checker tries to annotate every ExpressionOp in an EKL kernel
/// program with concrete types. It does this by computing the most restrictive
/// bounds on all expression values, without modifying the IR.
struct AbstractTypeChecker {
    virtual ~AbstractTypeChecker();

    /// Gets the most restrictive bound on @p expr .
    ///
    /// @pre    `expr`
    ///
    /// @param              expr    Expression.
    ///
    /// @return The most restrictive bound on @p expr .
    [[nodiscard]] virtual Type getType(Expression expr) const
    {
        std::ignore = expr;
        return nullptr;
    }

    /// Gets the most restrictive bounds for the @p exprs .
    ///
    /// @pre    @p exprs only contains Expression elements.
    ///
    /// @param              exprs   Expression range.
    ///
    /// @return The most restrictive bounds on @p exprs .
    [[nodiscard]] SmallVector<Type> getTypes(ValueRange exprs) const
    {
        return llvm::to_vector(llvm::map_range(exprs, [&](Value value) {
            return getType(llvm::cast<Expression>(value));
        }));
    }

    /// Attempts to refine the bound on @p expr to @p incoming .
    ///
    /// The type checker will accept @p incoming as the new type for @p expr if
    /// it is a subtype of the currently deduced bound.
    ///
    /// @pre    `expr && incoming`
    ///
    /// @param              expr        Expression.
    /// @param              incoming    The incoming bound for @p expr .
    ///
    /// @return Whether the bound was accepted.
    virtual LogicalResult refineBound(Expression expr, Type incoming)
    {
        std::ignore = expr;
        std::ignore = incoming;
        return success();
    }

    /// Attempts to meet the bound on @p expr with @p incoming .
    ///
    /// The type checker will accept @p incoming as the new type for @p expr if
    /// it is currently unbounded or has the same bound.
    ///
    /// @param              expr        Expression.
    /// @param              incoming    The incoming bound for @p expr .
    ///
    /// @return Whether the bound was accepted.
    virtual LogicalResult meetBound(Expression expr, Type incoming)
    {
        std::ignore = expr;
        std::ignore = incoming;
        return success();
    }

    /// Marks @p op as invalid.
    ///
    /// This causes the type checker to (re-)visit @p op .
    ///
    /// @pre    `op`
    ///
    /// @param              op  Operation.
    virtual void invalidate(Operation *op) { std::ignore = op; }

    /// Invalidates the owner of @p expr .
    ///
    /// If @p expr is the result of an operation, invalidates that operation.
    /// Otherwise, @p expr is a block argument, and its immediately surrounding
    /// parent operation is invalidated.
    ///
    /// @pre    `expr`
    void invalidate(Expression expr) { invalidate(getOwner(expr)); }

    /// Invalidates all users of @p expr .
    ///
    /// @pre    `expr`
    void invalidateUsers(Expression expr)
    {
        for (auto user : expr.getUsers()) invalidate(user);
    }

    /// Recursively invalidates all operations nested under (& including) @p op
    /// .
    ///
    /// @pre    `op`
    ///
    /// @param              op  Operation.
    void recursivelyInvalidate(Operation *op)
    {
        op->walk([&](Operation *child) { this->invalidate(child); });
    }
};

} // namespace mlir::ekl
