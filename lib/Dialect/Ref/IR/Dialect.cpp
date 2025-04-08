/// Implementation of the Ref dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Dialect.h"

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/Ref/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RefDialect implementation
//===----------------------------------------------------------------------===//

void RefDialect::initialize()
{
    registerTypes();
    registerAttributes();
    registerOps();
}
