/// Declares custom diagnostics and debugging helpers.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/Analysis/Shape.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>

namespace mlir::ekl {

auto operator<<(llvm::raw_ostream &os, ShapeRef shape) -> llvm::raw_ostream &;

auto operator<<(Diagnostic &os, const Extent &extent) -> Diagnostic &;
auto operator<<(Diagnostic &os, ShapeRef shape) -> Diagnostic &;

} // namespace mlir::ekl

namespace mlir::ekl {

inline auto operator<<(Diagnostic &os, const Extent &extent) -> Diagnostic &
{
    if (!extent.isBounded()) return os << "?";
    return os << extent.getValue();
}

inline void append(auto &os, ShapeRef shape)
{
    os << "[";
    llvm::interleaveComma(shape, os);
    os << "]";
}

inline auto operator<<(llvm::raw_ostream &os, ShapeRef shape)
    -> llvm::raw_ostream &
{
    append(os, shape);
    return os;
}

inline auto operator<<(Diagnostic &os, ShapeRef shape) -> Diagnostic &
{
    append(os, shape);
    return os;
}

} // namespace mlir::ekl
