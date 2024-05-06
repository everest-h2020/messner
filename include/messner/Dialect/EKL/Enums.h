/// Declaration of the EKL dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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
