/// Implementation of the Ref dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/Ref/IR/Ops.h"

#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ref;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/Ref/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReadOp implementation
//===----------------------------------------------------------------------===//

void ReadOp::build(
    OpBuilder &odsBuilder,
    OperationState &odsState,
    ReadableRef reference,
    bool isImpure,
    bool isVolatile)
{
    assert(reference);

    odsState.addOperands(reference);
    odsState.addTypes(reference.getType().getCellType());

    auto &props         = odsState.getOrAddProperties<Properties>();
    const auto unitAttr = odsBuilder.getUnitAttr();
    props.setIsImpure(isImpure ? unitAttr : UnitAttr{});
    props.setIsVolatile(isVolatile ? unitAttr : UnitAttr{});
}

void ReadOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects)
{
    const auto resource = getReference().getType().getResource();

    if (isVolatile()) {
        effects.emplace_back(
            MemoryEffects::Write::get(),
            &getReferenceMutable(),
            VolatileAttr::get(getContext()),
            0,
            false,
            resource);
    }

    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getReferenceMutable(),
        1,
        false,
        resource);

    if (isImpure()) {
        effects.emplace_back(
            MemoryEffects::Write::get(),
            &getReferenceMutable(),
            VolatileAttr::get(getContext()),
            2,
            true,
            resource);
    }
}

auto ReadOp::inferReturnTypes(
    MLIRContext *,
    std::optional<Location>,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) -> LogicalResult
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

void WriteOp::build(
    OpBuilder &odsBuilder,
    OperationState &odsState,
    Value value,
    WritableRef reference,
    bool isVolatile)
{
    assert(value && reference);
    assert(value.getType() == reference.getType().getCellType());

    odsState.addOperands(value);
    odsState.addOperands(reference);

    auto &props         = odsState.getOrAddProperties<Properties>();
    const auto unitAttr = odsBuilder.getUnitAttr();
    props.setIsVolatile(isVolatile ? unitAttr : UnitAttr{});
}

void WriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects)
{
    const auto resource = getReference().getType().getResource();

    if (isVolatile()) {
        effects.emplace_back(
            MemoryEffects::Read::get(),
            &getReferenceMutable(),
            VolatileAttr::get(getContext()),
            0,
            true,
            resource);
    }

    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getReferenceMutable(),
        1,
        false,
        resource);
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
