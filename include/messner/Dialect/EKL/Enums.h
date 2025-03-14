/// Declaration of the EKL dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ekl {

/// Distinguishes access restrictions on static variables.
enum class AccessModifier {
    // The variable is defined locally and not accessible externally.
    Local,
    // The variable is defined and accessible externally.
    Import,
    // The variable is defined locally, but accessible externally.
    Export
};

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ReferenceKind utilities
//===----------------------------------------------------------------------===//

/// Determines whether @p kind indicates a readable reference kind.
///
/// @param              kind    ReferenceKind.
///
/// @return Whether @p kind contains the ReferenceKind::In flag.
[[nodiscard]] inline bool isReadable(ReferenceKind kind)
{
    return bitEnumContainsAll(kind, ReferenceKind::In);
}

/// Determines whether @p kind indicates a readable reference kind.
///
/// @param              kind    ReferenceKind.
///
/// @return Whether @p kind contains the ReferenceKind::Out flag.
[[nodiscard]] inline bool isWritable(ReferenceKind kind)
{
    return bitEnumContainsAll(kind, ReferenceKind::Out);
}

//===----------------------------------------------------------------------===//
// RelationKind utilities
//===----------------------------------------------------------------------===//

/// Obtains the logically negated RelationKind.
///
/// @param  [in]        kind    RelationKind.
///
/// @return The RelationKind that is the logically negated predicate.
[[nodiscard]] inline RelationKind negate(RelationKind kind)
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

/// Flips the operands of a RelationKind.
///
/// @param  [in]        kind    RelationKind.
///
/// @return The RelationKind that is the same predicate for flipped operands.
[[nodiscard]] inline RelationKind flip(RelationKind kind)
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
