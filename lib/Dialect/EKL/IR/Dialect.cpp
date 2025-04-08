/// Implementation of the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Dialect.h"

#include "messner/Dialect/EKL/IR/TypeSystem.h"
#include "messner/Dialect/Ref/IR/Dialect.h" // IWYU pragma: keep

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/EKL/IR/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EKLDialect implementation
//===----------------------------------------------------------------------===//

auto EKLDialect::materializeConstant(OpBuilder &, Attribute, Type, Location)
    -> Operation *
{
    // TODO: Implement LiteralOp.
    return nullptr;
}

void EKLDialect::initialize()
{
    _typeSystem = &addInterface<TypeSystem>();

    registerTypes();
    registerAttributes();
    registerOps();
}
