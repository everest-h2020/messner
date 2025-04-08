/// Declaration of the Ref dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/IR/Attributes.h" // IWYU pragma: keep
#include "messner/Support/int.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/Ref/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// ReadOp implementation
//===----------------------------------------------------------------------===//

inline auto ReadOp::isVolatile() -> bool
{
    return getIsVolatile().value_or(false)
        || !messner::test_all(
               getReference().getType().getKind(),
               ReferenceKind::Exclusive);
}

inline auto ReadOp::isImpure() -> bool
{
    return getIsImpure().value_or(false)
        || !messner::test_all(
               getReference().getType().getKind(),
               ReferenceKind::Pure);
}

inline auto ReadOp::getSpeculatability() -> Speculation::Speculatability
{
    return isImpure() ? Speculation::NotSpeculatable
                      : Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// WriteOp implementation
//===----------------------------------------------------------------------===//

inline auto WriteOp::isVolatile() -> bool
{
    return getIsVolatile().value_or(false)
        || !messner::test_all(
               getReference().getType().getKind(),
               ReferenceKind::Idempotent);
}

inline auto WriteOp::getSpeculatability() -> Speculation::Speculatability
{
    return isVolatile() ? Speculation::NotSpeculatable
                        : Speculation::Speculatable;
}

} // namespace mlir::ref
