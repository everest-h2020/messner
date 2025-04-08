/// Declaration of the EKL dialect traits.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Dialect.h"                 // IWYU pragma: keep
#include "messner/Dialect/EKL/IR/TypeUtils.h"               // IWYU pragma: keep
#include "messner/Dialect/EKL/Interfaces/SemaOpInterface.h" // IWYU pragma: keep

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Typing/TypeCheckOpInterface.h>

namespace mlir::ekl::OpTrait {

template<class Concrete>
struct Declaration : mlir::OpTrait::TraitBase<Concrete, Declaration> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

template<class Concrete>
struct FunctorRegions : mlir::OpTrait::TraitBase<Concrete, FunctorRegions> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

template<class Concrete>
struct SemaOpImpl : mlir::OpTrait::TraitBase<Concrete, SemaOpImpl> {
    //===------------------------------------------------------------------===//
    // ConditionallySpeculatable
    //===------------------------------------------------------------------===//

    auto getSpeculatability() -> Speculation::Speculatability;
};

template<class Concrete>
struct Statement : mlir::OpTrait::TraitBase<Concrete, Statement> {};

template<class Concrete>
struct Expression : mlir::OpTrait::TraitBase<Concrete, Expression> {};

template<class Concrete>
struct CastOpImpl : mlir::OpTrait::TraitBase<Concrete, CastOpImpl> {
    //===------------------------------------------------------------------===//
    // CastOpInterface
    //===------------------------------------------------------------------===//

    static auto areCastCompatible(TypeRange, TypeRange) -> bool { return true; }
};

template<class Concrete>
struct Generator : mlir::OpTrait::TraitBase<Concrete, Generator> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

template<class Concrete>
struct Relational : mlir::OpTrait::TraitBase<Concrete, Relational> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

template<class Concrete>
struct Logical : mlir::OpTrait::TraitBase<Concrete, Logical> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

template<class Concrete>
struct Arithmetic : mlir::OpTrait::TraitBase<Concrete, Arithmetic> {
    static auto verifyTrait(Operation *) -> LogicalResult;
};

} // namespace mlir::ekl::OpTrait

namespace mlir::ekl::OpTrait {

//===----------------------------------------------------------------------===//
// Declaration implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto Declaration<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        T::template hasTrait<mlir::OpTrait::ZeroRegions>()
            || T::template hasTrait<mlir::OpTrait::IsIsolatedFromAbove>(),
        "`Declaration` trait is only applicable to `IsIsolatedFromAbove` ops.");

    return success();
}

//===----------------------------------------------------------------------===//
// FunctorRegions implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto FunctorRegions<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        !T::template hasTrait<mlir::OpTrait::ZeroRegions>(),
        "`FunctorRegions` trait is not applicable to `ZeroRegions` ops.");
    static_assert(
        T::template hasTrait<mlir::OpTrait::SingleBlock>(),
        "`FunctorRegions` trait is only applicable to `SingleBlock` ops.");

    return success();
}

//===----------------------------------------------------------------------===//
// SemaOpImpl implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto SemaOpImpl<T>::getSpeculatability() -> Speculation::Speculatability
{
    return hasConcreteType(this->getOperation())
             ? Speculation::Speculatability::Speculatable
             : Speculation::Speculatability::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// Generator implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto Generator<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        T::template hasTrait<Expression>(),
        "`Generator` trait is only applicable to `Expression` ops.");
    static_assert(
        T::template hasTrait<FunctorRegions>(),
        "`Generator` trait is only applicable to `FunctorRegions` ops.");

    return success();
}

//===----------------------------------------------------------------------===//
// Relational implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto Relational<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        T::template hasTrait<Expression>(),
        "`Relational` trait is only applicable to `Expression` ops.");

    return success();
}

//===----------------------------------------------------------------------===//
// Logical implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto Logical<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        T::template hasTrait<Expression>(),
        "`Logical` trait is only applicable to `Expression` ops.");

    return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic implementation
//===----------------------------------------------------------------------===//

template<class T>
inline auto Arithmetic<T>::verifyTrait(Operation *) -> LogicalResult
{
    static_assert(
        T::template hasTrait<Expression>(),
        "`Arithmetic` trait is only applicable to `Expression` ops.");

    return success();
}

} // namespace mlir::ekl::OpTrait
