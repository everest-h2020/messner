/// Implements the Ref dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Ops.h"

#include "messner/Support/int.h"

#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/Ref/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReadOp implementation
//===----------------------------------------------------------------------===//

Speculation::Speculatability ReadOp::getSpeculatability()
{
    return messner::test_all(
               getReference().getType().getKind(),
               ReferenceKind::Pure)
             ? Speculation::Speculatable
             : Speculation::NotSpeculatable;
}

LogicalResult ReadOp::inferReturnTypes(
    MLIRContext *,
    std::optional<Location>,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{
    ReadOp::GenericAdaptor<ValueRange> adaptor(
        operands,
        attributes,
        properties,
        regions);

    inferredReturnTypes.push_back(
        llvm::cast<ReferenceType>(adaptor.getReference().getType())
            .getCellType());
    return success();
}

//===----------------------------------------------------------------------===//
// WriteOp implementation
//===----------------------------------------------------------------------===//

Speculation::Speculatability WriteOp::getSpeculatability()
{
    return messner::test_all(
               getReference().getType().getKind(),
               ReferenceKind::Exclusive)
             ? Speculation::Speculatable
             : Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// RefDialect implementation
//===----------------------------------------------------------------------===//

void RefDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "messner/Dialect/Ref/IR/Ops.cpp.inc"
        >();
}
