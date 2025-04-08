/// Declaration of the EKL dialect type utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Types.h"

namespace mlir::ekl {

/// Determines whether @p type is a concrete type.
///
/// A concrete type is any type that is not ExpressionType.
///
/// @pre    `type`
auto hasConcreteType(Type type) -> bool;
/// Determines whether @p types are concrete types.
///
/// See hasConcreteType(Type) for more information.
auto hasConcreteType(TypeRange types) -> bool;
/// Determines whether @p value has a concrete type.
///
/// See hasConcreteType(Type) for more information.
///
/// @pre    `value`
auto hasConcreteType(Value value) -> bool;
/// Determines whether @p values have concrete types.
///
/// See hasConcreteType(Value) for more information.
auto hasConcreteType(ValueRange values) -> bool;
/// Determines whether @p op has concrete types.
///
/// An operation has a concrete type if all its operands, results and block
/// arguments of its regions have concrete types. See hasConcreteType(Value)
/// for more information.
///
/// @pre    `op`
auto hasConcreteType(Operation *op) -> bool;

//===----------------------------------------------------------------------===//
// ABI type constraints
//===----------------------------------------------------------------------===//

struct ABIScalarType : ScalarType {
    static auto classof(IntegerType) -> bool { return true; }
    static auto classof(FloatType type) -> bool;
    static auto classof(BoolType) -> bool { return true; }
    static auto classof(Type type) -> bool;

    /*implicit*/ ABIScalarType() = default;
    /*implicit*/ ABIScalarType(const ImplType *impl);
    /*implicit*/ ABIScalarType(IntegerType type);
    /*implicit*/ ABIScalarType(BoolType type);
};

struct ABIReferenceType : ReferenceType {
    static auto classof(ReferenceType type) -> bool;
    static auto classof(Type type) -> bool;

    /// Obtains an ABIReferenceType for @p scalarTy with @p shape .
    ///
    /// @pre    `scalarTy`
    /// @pre    `isValid(shape)`
    static auto
    get(ref::ReferenceKind kind, ABIScalarType scalarTy, ShapeRef shape = {})
        -> ABIReferenceType;
    /// Obtains an ABIReferenceType for @p arrayTy that @p isScalable .
    ///
    /// @pre    `arrayTy`
    /// @pre    `llvm::isa<ABIScalarType>(arrayTy.getScalarType())`
    static auto
    get(ref::ReferenceKind kind, ArrayType arrayTy, bool isScalable = false)
        -> ABIReferenceType;

    using ReferenceType::ReferenceType;
};

struct ABIType : Type {
    static auto classof(ABIScalarType) -> bool { return true; }
    static auto classof(ABIReferenceType) -> bool { return true; }
    static auto classof(Type type) -> bool;

    using Type::Type;
    /*implicit*/ ABIType(ABIScalarType type);
    /*implicit*/ ABIType(ABIReferenceType type);
};

} // namespace mlir::ekl

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// hasConcreteType
//===----------------------------------------------------------------------===//

inline auto hasConcreteType(Type type) -> bool
{
    assert(type);
    return !llvm::isa<ExpressionType>(type);
}

inline auto hasConcreteType(TypeRange types) -> bool
{
    return llvm::all_of(types, [](Type type) { return hasConcreteType(type); });
}

inline auto hasConcreteType(ValueRange values) -> bool
{
    return hasConcreteType(values.getTypes());
}

//===----------------------------------------------------------------------===//
// ABIScalarType implementation
//===----------------------------------------------------------------------===//

inline auto ABIScalarType::classof(FloatType type) -> bool
{
    // NOTE: We support a subset of what the LLVM dialect considers compatible.
    //       In particular, we only allow IEEE-754 float types.
    return llvm::isa<
        BFloat16Type,
        Float16Type,
        Float32Type,
        Float64Type,
        Float80Type,
        Float128Type>(type);
}

inline auto ABIScalarType::classof(Type type) -> bool
{
    if (const auto floatTy = llvm::dyn_cast<FloatType>(type); floatTy)
        return classof(floatTy);
    return llvm::isa<IntegerType, BoolType>(type);
}

inline ABIScalarType::ABIScalarType(const ImplType *impl) : ScalarType(impl) {}

inline ABIScalarType::ABIScalarType(IntegerType type)
        : ABIScalarType(static_cast<Type>(type).getImpl())
{}

inline ABIScalarType::ABIScalarType(BoolType type)
        : ABIScalarType(static_cast<Type>(type).getImpl())
{}

//===----------------------------------------------------------------------===//
// ABIReferenceType implementation
//===----------------------------------------------------------------------===//

inline auto ABIReferenceType::classof(ReferenceType type) -> bool
{
    return llvm::isa<ABIScalarType>(type.getScalarType());
}

inline auto ABIReferenceType::classof(Type type) -> bool
{
    if (const auto refTy = llvm::dyn_cast<ReferenceType>(type); refTy)
        return classof(refTy);
    return false;
}

inline auto ABIReferenceType::get(
    ref::ReferenceKind kind,
    ABIScalarType scalarTy,
    ShapeRef shape) -> ABIReferenceType
{
    assert(scalarTy);

    return llvm::cast<ABIReferenceType>(
        ReferenceType::get(kind, scalarTy, shape));
}

inline auto ABIReferenceType::get(
    ref::ReferenceKind kind,
    ArrayType arrayTy,
    bool isScalable) -> ABIReferenceType
{
    assert(arrayTy);
    assert(llvm::isa<ABIScalarType>(arrayTy.getScalarType()));

    return llvm::cast<ABIReferenceType>(
        ReferenceType::get(kind, arrayTy, isScalable));
}

//===----------------------------------------------------------------------===//
// ABIType implementation
//===----------------------------------------------------------------------===//

inline auto ABIType::classof(Type type) -> bool
{
    return llvm::isa<ABIScalarType, ABIReferenceType>(type);
}

inline ABIType::ABIType(ABIScalarType type) : Type(type.getImpl()) {}

inline ABIType::ABIType(ABIReferenceType type)
        : Type(static_cast<Type>(type).getImpl())
{}

} // namespace mlir::ekl
