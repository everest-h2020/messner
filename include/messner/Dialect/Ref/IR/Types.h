/// Declaration of the Ref dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/IR/Dialect.h" // IWYU pragma: keep
#include "messner/Dialect/Ref/Interfaces/ReferenceTypeInterface.h" // IWYU pragma: keep

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/Ref/IR/Types.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// ReferenceType implementation
//===----------------------------------------------------------------------===//

inline auto ReferenceType::cloneWith(Type cellType) const -> ReferenceType
{
    assert(cellType);

    return get(getContext(), cellType, getKind());
}

inline auto ReferenceType::cloneWith(ReferenceKind kind) const -> ReferenceType
{
    return get(getContext(), getCellType(), kind);
}

} // namespace mlir::ref
