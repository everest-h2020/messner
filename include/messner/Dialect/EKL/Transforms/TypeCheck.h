#pragma once

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"
#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::ekl {

struct TypeChecker : AbstractTypeChecker {
    [[nodiscard]] virtual Type getType(Expression expr) const override
    {
        if (const auto deduced = m_context.lookup(expr)) return deduced;
        return getTypeBound(expr);
    }

    virtual LogicalResult refineBound(Expression expr, Type incoming) override;

    virtual LogicalResult meetBound(Expression expr, Type incoming) override;

    virtual void invalidate(Operation *op) override;

    LogicalResult check(RewriterBase::Listener *listener = nullptr);

private:
    TypeCheckOpInterface popInvalid();

    void applyToIR(RewriterBase::Listener *listener = nullptr);

    llvm::DenseMap<Expression, Type> m_context;
    llvm::DenseSet<TypeCheckOpInterface> m_invalid;
};

/// Perform type checking for @p root and all of its descendants.
///
/// @pre    `root`
///
/// @param              root    Operation.
///
/// @return Whether type checking suceeded.
LogicalResult typeCheck(Operation *root);

} // namespace mlir::ekl
