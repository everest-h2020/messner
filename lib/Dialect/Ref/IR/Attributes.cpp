/// Implementation of the Ref dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Attributes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "messner/Dialect/Ref/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RefDialect implementation
//===----------------------------------------------------------------------===//

void RefDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "messner/Dialect/Ref/IR/Attributes.cpp.inc"
        >();
}
