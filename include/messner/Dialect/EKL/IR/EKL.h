/// Convenience include for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/TypeCheckingAdaptor.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/DiagHandler.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "messner/Dialect/EKL/Interfaces/Interfaces.h"
#include "messner/Dialect/EKL/Transforms/Passes.h"

#include <mlir/IR/MLIRContext.h>

namespace mlir::ekl {

/// Makes a literal @p value splat for @p type .
///
/// @pre    `type`
///
/// @retval LiteralAttr Literal representing @p value .
/// @retval nullptr     @p value does not fit into the scalar of @p type .
[[nodiscard]] LiteralAttr makeLiteral(BroadcastType type, int value);

//===----------------------------------------------------------------------===//
// coerce
//===----------------------------------------------------------------------===//
// TODO: Document.

[[nodiscard]] ekl::IntegerAttr
coerce(ScalarAttr input, ekl::IntegerType output);

[[nodiscard]] FloatAttr coerce(ScalarAttr input, FloatType output);

[[nodiscard]] ekl::IndexAttr coerce(ScalarAttr input, ekl::IndexType output);

[[nodiscard]] ScalarAttr coerce(ScalarAttr input, ScalarType output);

[[nodiscard]] ekl::ArrayAttr coerce(ekl::ArrayAttr input, ScalarType output);

} // namespace mlir::ekl
