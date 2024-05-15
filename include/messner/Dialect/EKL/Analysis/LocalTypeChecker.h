/// Declares the LocalTypeChecker.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h"

namespace mlir::ekl {

/// Implements an AbstractTypeChecker that locally invokes TypeCheckOpInterface.
///
/// Instead of accumulating any type bounds, the LocalTypeChecker only verifies
/// that the bounds deduced by the operation are subtypes of the result types
/// specified in the IR.
struct LocalTypeChecker final : AbstractTypeChecker {
    /// Initializes a LocalTypeChecker for @p parent .
    ///
    /// @pre    `parent`
    explicit LocalTypeChecker(TypeCheckOpInterface parent) : m_parent(parent)
    {
        assert(parent);
    }

    /// Gets the parent TypeCheckOpInterface.
    ///
    /// @post   `result`
    [[nodiscard]] TypeCheckOpInterface getParent() const { return m_parent; }

    /// Gets the type bound on @p expr that's in the IR.
    virtual Type getType(Expression expr) const override
    {
        return getTypeBound(expr);
    }

    /// Verifies that @p incoming is a legal refinement of @p expr .
    virtual LogicalResult refineBound(Expression expr, Type incoming) override;

    /// Verifies that @p incoming legally meets with @p expr .
    virtual LogicalResult meetBound(Expression expr, Type incoming) override;

private:
    /// Generates a diagnostic for @p expr .
    ///
    /// @param              expr    Expression.
    ///
    /// @return (diagnostic, isError)
    std::pair<InFlightDiagnostic, bool> report(Expression expr) const;

    TypeCheckOpInterface m_parent;
};

} // namespace mlir::ekl
