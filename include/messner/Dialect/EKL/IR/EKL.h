/// Convenience include for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/TypeCheckingAdaptor.h"
#include "messner/Dialect/EKL/IR/DiagHandler.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "messner/Dialect/EKL/Interfaces/Interfaces.h"
#include "messner/Dialect/EKL/Transforms/Passes.h"

namespace mlir::ekl {

/// Makes the "zeros" value for @p type using @p builder .
///
/// @pre    `type`
[[nodiscard]]
Value makeZero(OpBuilder &builder, Location loc, BroadcastType type);

/// Makes the "ones" value for @p type using @p builder .
///
/// @pre    `type`
[[nodiscard]]
Value makeOne(OpBuilder &builder, Location loc, BroadcastType type);

} // namespace mlir::ekl
