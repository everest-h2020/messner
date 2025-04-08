/// Declaration of the EKL dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// RelationKind utilities
//===----------------------------------------------------------------------===//

/// Obtains the logically negated RelationKind.
///
/// @param  [in]        kind    RelationKind.
///
/// @return The RelationKind that is the logically negated predicate.
auto negate(RelationKind kind) -> RelationKind;

/// Flips the operands of a RelationKind.
///
/// @param  [in]        kind    RelationKind.
///
/// @return The RelationKind that is the same predicate for flipped operands.
auto flip(RelationKind kind) -> RelationKind;

} // namespace mlir::ekl

namespace mlir::ekl {

inline auto negate(RelationKind kind) -> RelationKind
{
    switch (kind) {
    case RelationKind::Equivalent:     return RelationKind::Antivalent;
    case RelationKind::Antivalent:     return RelationKind::Equivalent;
    case RelationKind::GreaterOrEqual: return RelationKind::LessThan;
    case RelationKind::GreaterThan:    return RelationKind::LessOrEqual;
    case RelationKind::LessOrEqual:    return RelationKind::GreaterThan;
    case RelationKind::LessThan:       return RelationKind::GreaterOrEqual;
    }
}

inline auto flip(RelationKind kind) -> RelationKind
{
    switch (kind) {
    case RelationKind::Equivalent:     return RelationKind::Equivalent;
    case RelationKind::Antivalent:     return RelationKind::Antivalent;
    case RelationKind::GreaterOrEqual: return RelationKind::LessOrEqual;
    case RelationKind::GreaterThan:    return RelationKind::LessThan;
    case RelationKind::LessOrEqual:    return RelationKind::GreaterOrEqual;
    case RelationKind::LessThan:       return RelationKind::GreaterThan;
    }
}

} // namespace mlir::ekl
