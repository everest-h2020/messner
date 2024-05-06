/// Implements the TypeCheckingAdaptor.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/TypeCheckingAdaptor.h"

#include <numeric>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// TypeCheckingAdaptor implementation
//===----------------------------------------------------------------------===//

Contradiction
TypeCheckingAdaptor::unify(ArrayRef<Type> types, Type &result) const
{
    auto temp = llvm::to_vector(types);
    return unifyImpl(temp, result);
}

Contradiction TypeCheckingAdaptor::unify(ValueRange exprs, Type &result) const
{
    auto types = getTypes(exprs);
    return unifyImpl(types, result).explain(exprs);
}

Contradiction TypeCheckingAdaptor::broadcast(
    ArrayRef<Type> types,
    SmallVectorImpl<extent_t> &extents) const
{
    switch (ekl::broadcast(types, extents)) {
    case BroadcastResult::Success:   return Contradiction::none();
    case BroadcastResult::Unbounded: return Contradiction::indeterminate();
    case BroadcastResult::Failure:
        auto diag = emitError() << "can't broadcast ";
        llvm::interleaveComma(types, diag);
        diag << " together";
        return diag;
    }
}

Contradiction TypeCheckingAdaptor::broadcast(
    ValueRange exprs,
    SmallVectorImpl<extent_t> &extents) const
{
    const auto types = getTypes(exprs);
    return broadcast(types, extents).explain(exprs);
}

Contradiction TypeCheckingAdaptor::broadcast(
    Type type,
    ExtentRange extents,
    ArrayType &result) const
{
    const auto maybe = ekl::broadcast(type, extents);
    if (failed(maybe)) {
        auto diag = emitError() << "can't broadcast " << type << " to [";
        llvm::interleaveComma(extents, diag);
        diag << "]";
        return diag;
    }
    result = *maybe;
    return result ? Contradiction::none() : Contradiction::indeterminate();
}

Contradiction TypeCheckingAdaptor::broadcast(MutableArrayRef<Type> types) const
{
    switch (ekl::broadcast(types)) {
    case BroadcastResult::Success:   return Contradiction::none();
    case BroadcastResult::Unbounded: return Contradiction::indeterminate();
    case BroadcastResult::Failure:
        auto diag = emitError() << "can't broadcast ";
        llvm::interleaveComma(types, diag);
        diag << " together";
        return diag;
    }
}

Contradiction TypeCheckingAdaptor::broadcast(
    ValueRange exprs,
    SmallVectorImpl<Type> &result) const
{
    result = getTypes(exprs);
    return broadcast(result).explain(exprs);
}

Contradiction
TypeCheckingAdaptor::broadcastAndUnify(ValueRange exprs, Type &result) const
{
    SmallVector<Type> broadcasted;
    if (auto contra = broadcast(exprs, broadcasted)) return contra;
    return unify(broadcasted, result).explain(exprs);
}

Contradiction
TypeCheckingAdaptor::unifyImpl(SmallVectorImpl<Type> &types, Type &result) const
{
    const auto unified = ekl::unify(types);
    if (succeeded(unified)) {
        result = *unified;
        return *unified ? Contradiction::none()
                        : Contradiction::indeterminate();
    }
    auto diag = emitError() << "can't unify ";
    llvm::interleaveComma(types, diag);
    return diag;
}
