/// Declaration of the EKL dialect type system.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Types.h"

#include <algorithm>
#include <mlir/IR/TypeSystem.h>

namespace mlir::ekl {

/// Provides the DialectTypeSystemInterface implementation for the EKL dialect.
class TypeSystem : public DialectTypeSystemInterface {
public:
    using DialectTypeSystemInterface::DialectTypeSystemInterface;

    /// @copydoc isSubtype(Type, Type)
    auto isSubtype(FloatType sub, FloatType super) const -> bool;
    /// @copydoc isSubtype(Type, Type)
    auto isSubtype(IntegerType sub, FloatType super) const -> bool;
    /// @copydoc isSubtype(Type, Type)
    auto isSubtype(IntegerType sub, IntegerType super) const -> bool;
    /// @copydoc isSubtype(Type, Type)
    auto isSubtype(IndexType sub, IndexType super) const -> bool;
    /// @copydoc isSubtype(Type, Type)
    auto isSubtype(ArrayType sub, ArrayType super) const -> bool;
    /// @copydoc AbstractTypeSystem::isSubtype(Type, Type)
    auto isSubtype(Type sub, Type super) const -> bool override;

    using AbstractTypeSystem::promote;
    /// @copydoc promote(Type, Type)
    auto promote(FloatType lhs, FloatType rhs) const -> FloatType;
    /// @copydoc promote(Type, Type)
    auto promote(FloatType lhs, IntegerType rhs) const -> FloatType;
    /// @copydoc promote(Type, Type)
    auto promote(IntegerType lhs, IntegerType rhs) const -> IntegerType;
    /// @copydoc promote(Type, Type)
    auto promote(IndexType lhs, IndexType rhs) const -> IndexType;
    /// @copydoc promote(Type, Type)
    auto promote(ArrayType lhs, ArrayType rhs) const -> ArrayType;
    /// @copydoc AbstractTypeSystem::promote(Type, Type)
    auto promote(Type lhs, Type rhs) const -> Type override;

    /// @copydoc AbstractTypeSystem::promote(OpBuilder &, Location, Value, Type)
    auto
    promote(OpBuilder &builder, Location loc, Value input, Type superTy) const
        -> Operation * override;
};

} // namespace mlir::ekl

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// TypeSystem implementation
//===----------------------------------------------------------------------===//

inline auto TypeSystem::isSubtype(IndexType sub, IndexType super) const -> bool
{
    assert(sub && super);

    // N <= M -> Index(N, x) <: Index(M, x)
    if (sub.isFromEnd() ^ super.isFromEnd()) return false;
    if (!super.getBound().isBounded()) return true;
    return sub.getBound() <= super.getBound();
}

inline auto TypeSystem::isSubtype(ArrayType sub, ArrayType super) const -> bool
{
    assert(sub && super);

    // U <: T -> Array(U, x) <: Array(T, x)
    return sub.getShape() == super.getShape()
        && isSubtype(sub.getScalarType(), super.getScalarType());
}

inline auto TypeSystem::promote(IndexType lhs, IndexType rhs) const -> IndexType
{
    assert(lhs && rhs);

    // Index(N, x) |_| Index(M, x) = Index(max(N, M), x)
    if (lhs.isFromEnd() ^ rhs.isFromEnd()) return {};
    if (!lhs.getBound().isBounded()) return lhs;
    if (!rhs.getBound().isBounded()) return rhs;
    return IndexType::get(
        lhs.getContext(),
        std::max(lhs.getBound(), rhs.getBound()),
        lhs.isFromEnd());
}

inline auto TypeSystem::promote(ArrayType lhs, ArrayType rhs) const -> ArrayType
{
    assert(lhs && rhs);

    // Array(T, x) |_| Array(U, x) = Array(T |_| U, x)
    if (lhs.getShape() != rhs.getShape()) return {};
    if (const auto scalarTy = llvm::cast_if_present<ScalarType>(
            promote(lhs.getScalarType(), rhs.getScalarType()));
        scalarTy)
        return lhs.cloneWith(scalarTy);
    return {};
}

} // namespace mlir::ekl
