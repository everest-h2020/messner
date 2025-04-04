/// Declaration of the Ref dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/IR/Types.h"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/Ref/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//
