/// Implements the EKL dialect traits.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Interfaces/TypeCheckOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Op traits
//===----------------------------------------------------------------------===//

namespace mlir::ekl::impl {

/// Verifies the HasFunctors trait.
LogicalResult verifyHasFunctors(Operation *op);

} // namespace mlir::ekl::impl

namespace mlir::ekl::OpTrait {

using mlir::OpTrait::TraitBase;

/// Implements the SymbolOp trait.
template<typename ConcreteType>
struct IsSymbol : TraitBase<ConcreteType, IsSymbol> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<SymbolOpInterface::Trait>(),
            "`IsSymbol` trait requires the op to implement the "
            "`SymbolOpInterface`");
        static_assert(
            ConcreteType::template hasTrait<mlir::OpTrait::ZeroRegions>()
                || ConcreteType::template hasTrait<
                    mlir::OpTrait::IsIsolatedFromAbove>(),
            "`IsSymbol` trait is only applicable to `IsIsolatedFromAbove` "
            "ops.");

        return success();
    }
};

/// Implements the HasFunctors trait.
template<typename ConcreteType>
struct HasFunctors : TraitBase<ConcreteType, HasFunctors> {
    static LogicalResult verifyRegionTrait(Operation *op)
    {
        static_assert(
            ConcreteType::template hasTrait<mlir::OpTrait::ZeroRegions>()
                || ConcreteType::template hasTrait<
                    mlir::OpTrait::SingleBlock>(),
            "`HasFunctors` trait is only applicable to `SingleBlock`");

        if constexpr (ConcreteType::template hasTrait<
                          mlir::OpTrait::ZeroRegions>())
            return success();
        else
            return impl::verifyHasFunctors(op);
    }

    //===------------------------------------------------------------------===//
    // ConditionallySpeculatable
    //===------------------------------------------------------------------===//

    Speculation::Speculatability getSpeculatability()
    {
        return impl::isFullyTyped(this->getOperation())
                 ? Speculation::RecursivelySpeculatable
                 : Speculation::NotSpeculatable;
    }
};

/// Implements the StatementOp trait.
template<typename ConcreteType>
struct IsStatement : TraitBase<ConcreteType, IsStatement> {};

/// Implements the ExpressionOp trait.
template<typename ConcreteType>
struct IsExpression : TraitBase<ConcreteType, IsExpression> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<TypeCheckOpInterface::Trait>(),
            "`IsExpression` trait requires the op to implement the "
            "`TypeCheckOpInterface`");

        return success();
    }
};

/// Implements the GeneratorOp trait.
template<typename ConcreteType>
struct IsGenerator : TraitBase<ConcreteType, IsGenerator> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<IsExpression>(),
            "`IsGenerator` trait is only applicable to `IsExpression` ops.");

        return success();
    }
};

/// Implements the RelationalOp trait.
template<typename ConcreteType>
struct IsRelational : TraitBase<ConcreteType, IsRelational> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<IsExpression>(),
            "`IsRelational` trait is only applicable to `IsExpression` ops.");

        return success();
    }
};

/// Implements the LogicalOp trait.
template<typename ConcreteType>
struct IsLogical : TraitBase<ConcreteType, IsLogical> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<IsExpression>(),
            "`IsLogical` trait is only applicable to `IsExpression` ops.");

        return success();
    }
};

/// Implements the ArithmeticOp trait.
template<typename ConcreteType>
struct IsArithmetic : TraitBase<ConcreteType, IsArithmetic> {
    static LogicalResult verifyTrait(Operation *)
    {
        static_assert(
            ConcreteType::template hasTrait<IsExpression>(),
            "`IsArithmetic` trait is only applicable to `IsExpression` ops.");

        return success();
    }
};

} // namespace mlir::ekl::OpTrait
