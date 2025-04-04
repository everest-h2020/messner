/// Implements the Ref dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Base.h"

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/Ref/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RefDialect implementation
//===----------------------------------------------------------------------===//

void RefDialect::initialize()
{
    registerOps();
    registerTypes();
}
