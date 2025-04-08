/// Declaration of the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Attributes.h" // IWYU pragma: keep
#include "messner/Dialect/EKL/IR/Traits.h"     // IWYU pragma: keep
#include "messner/Dialect/EKL/IR/TypeUtils.h"  // IWYU pragma: keep
#include "messner/Support/concepts.h"          // IWYU pragma: keep

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/EKL/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

inline void FuncOp::build(
    OpBuilder &odsBuilder,
    OperationState &odsState,
    StringRef name,
    FunctionType type)
{
    build(odsBuilder, odsState, odsBuilder.getStringAttr(name), type);
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

inline void
KernelOp::build(OpBuilder &odsBuilder, OperationState &odsState, StringRef name)
{
    build(odsBuilder, odsState, odsBuilder.getStringAttr(name));
}

inline auto KernelOp::getAttrs(BlockArgument arg) -> DictionaryAttr::ValueType
{
    assert(arg && arg.getOwner() == getBody());

    if (!getArgAttrs()) return {};
    return llvm::cast<DictionaryAttr>(
               getArgAttrs()->operator[](arg.getArgNumber()))
        .getValue();
}

inline auto KernelOp::getAttr(BlockArgument arg, StringRef name) -> Attribute
{
    assert(arg && arg.getOwner() == getBody());

    const auto attrs = getAttrs(arg);
    const auto [it, found] =
        impl::findAttrSorted(attrs.begin(), attrs.end(), name);
    return found ? it->getValue() : Attribute{};
}

//===----------------------------------------------------------------------===//
// LiteralOp implementation
//===----------------------------------------------------------------------===//

inline void
LiteralOp::build(OpBuilder &, OperationState &odsState, LiteralAttr value)
{
    assert(value);

    odsState.addAttribute(getValueAttrName(odsState.name), value);
    odsState.addTypes(value.getType());
}

inline auto LiteralOp::fold(FoldAdaptor) -> OpFoldResult { return getValue(); }

//===----------------------------------------------------------------------===//
// YieldOp implementation
//===----------------------------------------------------------------------===//

inline void YieldOp::build(OpBuilder &, OperationState &odsState, Value operand)
{
    assert(operand);

    odsState.addOperands(operand);
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

inline void BroadcastOp::build(
    OpBuilder &odsBuilder,
    OperationState &odsState,
    ArrayType resultType,
    Value operand)
{
    assert(resultType);
    assert(operand);

    odsState.addOperands(operand);
    odsState.addAttribute(
        getSignedExtentsAttrName(odsState.name),
        getSignedExtentsAttr(odsBuilder.getContext(), resultType.getShape()));
    odsState.addTypes(resultType);
}

inline void BroadcastOp::build(
    OpBuilder &odsBuilder,
    OperationState &odsState,
    ShapeRef resultShape,
    Value operand)
{
    assert(operand);

    odsState.addOperands(operand);
    odsState.addAttribute(
        getSignedExtentsAttrName(odsState.name),
        getSignedExtentsAttr(odsBuilder.getContext(), resultShape));

    if (const auto bcastTy = llvm::dyn_cast<BroadcastType>(operand.getType());
        bcastTy)
        odsState.addTypes(bcastTy.cloneWith(resultShape));
    else
        odsState.addTypes(ExpressionType::get(odsBuilder.getContext()));
}

inline auto
BroadcastOp::getSignedExtentsAttr(MLIRContext *context, ShapeRef shape)
    -> DenseI64ArrayAttr
{
    return DenseI64ArrayAttr::get(
        context,
        ArrayRef<int64_t>(
            reinterpret_cast<const int64_t *>(shape.data()),
            shape.size()));
}

inline auto BroadcastOp::getResultShape() -> ShapeRef
{
    return ShapeRef(
        std::bit_cast<const Extent *>(getSignedExtents().data()),
        getSignedExtents().size());
}

} // namespace mlir::ekl
