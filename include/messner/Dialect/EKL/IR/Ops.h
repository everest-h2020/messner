/// Declaration of the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/Traits.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::ekl {

/// Reference to a function that populates a branch body.
using BranchBuilderRef = llvm::function_ref<void(OpBuilder &, Location)>;

/// Reference to a function that populates a functor body.
using FunctorBuilderRef =
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)>;

/// Obtains an callable that builds a simple functor body.
///
/// The callable will create an operation with the name @p op that accepts all
/// the block arguments and is expected to produce one expression-typed result.
/// The result type may be bounded using @p resultBound . A location can be
/// specified using @p opLoc , which is substituted with the parent loc if it
/// is absent. Optional @p attributes will be attached to the op as well.
///
/// The callable will pass the first result of the @p op to a `YieldOp`.
///
/// @param              op          OperationName.
/// @param              opLoc       Optional location.
/// @param              resultBound Optional result type bound.
/// @param              attributes  Optional attributes.
///
/// @return An owning FunctorBuilderRef-like type.
[[nodiscard]] std::function<void(OpBuilder &, Location, ValueRange)>
getFunctorBuilder(
    OperationName op,
    std::optional<Location> opLoc = {},
    Type resultBound              = {},
    DictionaryAttr attributes     = {});

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/EKL/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//
