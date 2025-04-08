/// Implementation of the EKL dialect ops.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Ops.h"

#include "messner/Dialect/EKL/IR/Dialect.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ProgramOp implementation
//===----------------------------------------------------------------------===//

auto ProgramOp::verifyRegions() -> LogicalResult
{
    // All descendants must be declarations.
    for (auto &op : this->getOps())
        if (!op.hasTrait<ekl::OpTrait::Declaration>()) {
            auto diag = emitError("invalid program");
            diag.attachNote(op.getLoc()) << "expected declaration";
            return diag;
        }

    return success();
}

//===----------------------------------------------------------------------===//
// FuncOp implementation
//===----------------------------------------------------------------------===//

void FuncOp::build(
    OpBuilder &,
    OperationState &state,
    StringAttr name,
    FunctionType type)
{
    assert(name && type);

    state.addAttribute(getSymNameAttrName(state.name), name);
    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(type));
    state.addRegion();
}

auto FuncOp::verify() -> LogicalResult
{
    // Can't export functions.
    if (isPublic()) return emitOpError("visibility can't be public");

    return success();
}

auto FuncOp::verifyRegions() -> LogicalResult
{
    if (isExternal()) return success();

    // There must be a YieldOp terminator if the definition is not empty.
    if (getBody()->empty() || !llvm::isa<YieldOp>(&getBody()->back())) {
        auto diag = emitOpError("requires `ekl.yield` terminator");
        if (!getBody()->empty())
            diag.attachNote(getBody()->back().getLoc())
                << "found `" << getBody()->back().getName() << "` instead";
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// KernelOp implementation
//===----------------------------------------------------------------------===//

static auto getAllArgAttrs(KernelOp op) -> SmallVector<Attribute>
{
    auto allAttrs = llvm::to_vector(
        op.getArgAttrs() ? op.getArgAttrs()->getValue()
                         : ArrayRef<Attribute>{});
    allAttrs.resize(
        op.getBody()->getNumArguments(),
        DictionaryAttr::get(op.getContext()));
    return allAttrs;
}

void KernelOp::build(OpBuilder &, OperationState &state, StringAttr name)
{
    assert(name);

    state.addAttribute(getSymNameAttrName(state.name), name);
    state.addRegion()->emplaceBlock();
}

auto KernelOp::insertArgument(
    unsigned pos,
    ABIType type,
    Location loc,
    DictionaryAttr attrs) -> BlockArgument
{
    assert(pos <= getBody()->getNumArguments());
    assert(type);

    const auto result = getBody()->insertArgument(pos, type, loc);
    if (attrs) setAttrs(result, attrs);
    return result;
}

void KernelOp::eraseArgument(BlockArgument arg)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs = getAllArgAttrs(*this);
    allAttrs.erase(std::next(allAttrs.begin(), arg.getArgNumber()));
    setArgAttrsAttr(TupleAttr::get(getContext(), allAttrs));
}

void KernelOp::setAttrs(BlockArgument arg, DictionaryAttr attrs)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs                = getAllArgAttrs(*this);
    allAttrs[arg.getArgNumber()] = attrs;
    setArgAttrsAttr(TupleAttr::get(getContext(), allAttrs));
}

void KernelOp::setAttr(BlockArgument arg, NamedAttribute attr)
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs  = getAllArgAttrs(*this);
    auto &argAttrs = allAttrs[arg.getArgNumber()];
    NamedAttrList dict(llvm::cast<DictionaryAttr>(argAttrs));
    dict.set(attr.getName(), attr.getValue());
    argAttrs = DictionaryAttr::get(getContext(), dict);
}

auto KernelOp::removeAttr(BlockArgument arg, StringRef name) -> Attribute
{
    assert(arg && arg.getOwner() == getBody());

    auto allAttrs  = getAllArgAttrs(*this);
    auto &argAttrs = allAttrs[arg.getArgNumber()];
    NamedAttrList dict(llvm::cast<DictionaryAttr>(argAttrs));
    const auto result = dict.erase(name);
    argAttrs          = DictionaryAttr::get(getContext(), dict);
    return result;
}

auto KernelOp::verify() -> LogicalResult
{
    // If argument attributes are specified, their number must match.
    if (const auto allArgAttrs = getArgAttrs(); allArgAttrs) {
        if (allArgAttrs->size() != getBody()->getNumArguments())
            return emitOpError()
                << "expected " << getBody()->getNumArguments()
                << " argument attributes, but got " << allArgAttrs->size();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// GetStaticOp implementation
//===----------------------------------------------------------------------===//

auto GetStaticOp::verifySymbolUses(SymbolTableCollection &symbolTable)
    -> LogicalResult
{
    auto targetOp =
        symbolTable.lookupNearestSymbolFrom(*this, getTargetNameAttr());
    auto staticOp = llvm::dyn_cast_if_present<StaticOp>(targetOp);
    if (!staticOp) {
        auto diag = emitOpError("'")
                 << getTargetName() << "' does not reference a static variable";
        if (targetOp)
            diag.attachNote(targetOp->getLoc()) << "references this symbol";
        return diag;
    }

    if (getType() != staticOp.getType()) {
        auto diag = emitOpError("expected ")
                 << staticOp.getType() << ", but got " << getType();
        diag.attachNote(targetOp->getLoc()) << "symbol declared here";
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// PromoteOp implementation
//===----------------------------------------------------------------------===//

auto PromoteOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

auto BroadcastOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

auto CoerceOp::fold(FoldAdaptor) -> OpFoldResult
{
    // TODO: Implement.
    return {};
}

//===----------------------------------------------------------------------===//
// EKLDialect implementation
//===----------------------------------------------------------------------===//

void EKLDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "messner/Dialect/EKL/IR/Ops.cpp.inc"
        >();
}
