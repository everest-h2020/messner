/// Declaration of the Ref ReferenceTypeInterface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/Enums.h" // IWYU pragma: keep

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;
using namespace mlir::ref;

//===- Generated includes -------------------------------------------------===//

#include "messner/Dialect/Ref/Interfaces/ReferenceTypeInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

class ReadableRefType : public ReferenceTypeInterface {
public:
    static auto classof(ReferenceTypeInterface type) -> bool;
    static auto classof(Type type) -> bool;

    using ReferenceTypeInterface::ReferenceTypeInterface;

    auto isReadable() const -> bool;
};

using ReadableRef = TypedValue<ReadableRefType>;

class WritableRefType : public ReferenceTypeInterface {
public:
    static auto classof(ReferenceTypeInterface type) -> bool;
    static auto classof(Type type) -> bool;

    using ReferenceTypeInterface::ReferenceTypeInterface;

    auto isWritable() const -> bool;
};

using WritableRef = TypedValue<WritableRefType>;

} // namespace mlir::ref

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// ReadableRefType implementation
//===----------------------------------------------------------------------===//

inline auto ReadableRefType::classof(ReferenceTypeInterface type) -> bool
{
    return type.isReadable();
}

inline auto ReadableRefType::classof(Type type) -> bool
{
    if (const auto refTy = llvm::dyn_cast<ReferenceTypeInterface>(type); refTy)
        return classof(refTy);
    return false;
}

inline auto ReadableRefType::isReadable() const -> bool { return true; }

//===----------------------------------------------------------------------===//
// WritableRefType implementation
//===----------------------------------------------------------------------===//

inline auto WritableRefType::classof(ReferenceTypeInterface type) -> bool
{
    return type.isWritable();
}

inline auto WritableRefType::classof(Type type) -> bool
{
    if (const auto refTy = llvm::dyn_cast<ReferenceTypeInterface>(type); refTy)
        return classof(refTy);
    return false;
}

inline auto WritableRefType::isWritable() const -> bool { return true; }

} // namespace mlir::ref
