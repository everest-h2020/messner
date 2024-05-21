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

} // namespace mlir::ekl
