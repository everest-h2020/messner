/// Declaration of the REF dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Support/int.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/Ref/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// ReferenceKind utilities
//===----------------------------------------------------------------------===//

/// Determines whether @p kind is a readable reference kind.
[[nodiscard]]
constexpr bool isReadable(ReferenceKind kind);

/// Determines whether @p kind is a writable reference kind.
[[nodiscard]]
constexpr bool isWritable(ReferenceKind kind);

} // namespace mlir::ref

namespace mlir::ref {

[[nodiscard]]
constexpr bool isReadable(ReferenceKind kind)
{
    return messner::test_all(kind, ReferenceKind::Read);
}

[[nodiscard]]
constexpr bool isWritable(ReferenceKind kind)
{
    return messner::test_all(kind, ReferenceKind::Write);
}

} // namespace mlir::ref
