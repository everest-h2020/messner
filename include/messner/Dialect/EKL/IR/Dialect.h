/// Declaration of the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Enums.h" // IWYU pragma: export

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//
//
// We don't need to re-implement everything, a copule of core MLIR entities can
// be reused. We assign them a unique, non-conflicting name here.

using mlir::FloatType;
using mlir::FloatAttr;
using mlir::BoolAttr;
using mlir::StringAttr;
using TupleAttr = mlir::ArrayAttr;

//===----------------------------------------------------------------------===//
// TypeSystem
//===----------------------------------------------------------------------===//
//
// Fowrard declaration allows users to query the type system from the dialect.

class TypeSystem;

/// Gets the TypeSystem instance of the EKL dialect in @p context .
auto getTypeSystem(MLIRContext *context) -> const TypeSystem &;

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/EKL/IR/Dialect.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

inline auto getTypeSystem(MLIRContext *context) -> const TypeSystem &
{
    assert(context);
    auto dialect = context->getOrLoadDialect<EKLDialect>();
    return dialect->getTypeSystem();
}

} // namespace mlir::ekl
