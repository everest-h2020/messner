/// Declaration of the Ref dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/Ref/IR/Base.h"

#include <mlir/IR/Types.h>

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/Ref/IR/Types.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

class ReadableRefType : public ReferenceType {
public:
    [[nodiscard]] static bool classof(ReferenceType type);
    [[nodiscard]] static bool classof(Type type);

    using ReferenceType::ReferenceType;

    [[nodiscard]] bool isReadable() const;
};

class WritableRefType : public ReferenceType {
public:
    [[nodiscard]] static bool classof(ReferenceType type);
    [[nodiscard]] static bool classof(Type type);

    using ReferenceType::ReferenceType;

    [[nodiscard]] bool isWritable() const;
};

} // namespace mlir::ref

namespace mlir::ref {

//===----------------------------------------------------------------------===//
// ReferenceType implementation
//===----------------------------------------------------------------------===//

inline bool ReferenceType::isReadable() const
{
    return ref::isReadable(getKind());
}

inline bool ReferenceType::isWritable() const
{
    return ref::isWritable(getKind());
}

inline ReferenceType ReferenceType::cloneWith(Type cellType) const
{
    assert(cellType);

    return get(getContext(), cellType, getKind());
}

inline ReferenceType ReferenceType::cloneWith(ReferenceKind kind) const
{
    return get(getContext(), getCellType(), kind);
}

//===----------------------------------------------------------------------===//
// ReadableRefType implementation
//===----------------------------------------------------------------------===//

inline bool ReadableRefType::classof(ReferenceType type)
{
    return type.isReadable();
}

inline bool ReadableRefType::classof(Type type)
{
    if (const auto refTy = llvm::dyn_cast<ReferenceType>(type); refTy)
        return classof(refTy);
    return false;
}

inline bool ReadableRefType::isReadable() const { return true; }

//===----------------------------------------------------------------------===//
// WritableRefType implementation
//===----------------------------------------------------------------------===//

inline bool WritableRefType::classof(ReferenceType type)
{
    return type.isWritable();
}

inline bool WritableRefType::classof(Type type)
{
    if (const auto refTy = llvm::dyn_cast<ReferenceType>(type); refTy)
        return classof(refTy);
    return false;
}

inline bool WritableRefType::isWritable() const { return true; }

} // namespace mlir::ref
