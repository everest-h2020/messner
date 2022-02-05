/** Declaration of the CFDlang dialect interfaces.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/Concepts/Atom.h"
#include "mlir/Dialect/CFDlang/IR/Types.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::cfdlang::interface_defaults {

/** Verifies a DefinitionOp. */
LogicalResult verifyDefinitionOp(Operation *op);

/** Infers the shape of the result Atom assuming SameOperandsAndResultType. */
LogicalResult inferAtomShape(
    ValueRange operands,
    teil::ShapeBuilder &atomShape
);

/** Reifies the size of the result Atom assuming SameOperandsAndResultType. */
FailureOr<teil::AtomSize> reifyAtomSize(
    OpBuilder &builder,
    Operation* op
);

/** Infers the type of the result Atom using the other interface methods. */
template<class ConcreteOp>
inline AtomType inferAtomType(
    MLIRContext *context,
    Optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    RegionRange regions
)
{
    teil::ShapeStorage atomShape;
    if (
        failed(
            ConcreteOp::inferAtomShape(
                context,
                location,
                operands,
                attributes,
                regions,
                atomShape
            )
        )
    ) {
        return nullptr;
    }

    return AtomType::get(context, atomShape);
}

/** Implements the ReifyRankedShapedTypeOpInterface for an AtomOp. */
LogicalResult reifyResultShapes(
    teil::AtomSize atomSize,
    OpBuilder &builder,
    Location loc,
    ReifiedRankedShapedTypeDims &reifiedReturnShapes
);

} // namespace mlir::cfdlang::interface_defaults

//===- Generated includes -------------------------------------------------===//

//#include "mlir/Dialect/CFDlang/IR/AttrInterfaces.h.inc"
#include "mlir/Dialect/CFDlang/IR/OpInterfaces.h.inc"
//#include "mlir/Dialect/CFDlang/IR/TypeInterfaces.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::OpTrait {

/** Native trait that implements the InferTypeOpInterface,
 *  InferShapedTypeOpInterface and ReifyRankedShapedTypeOpInterface interfaces
 *  using the AtomOp interface.
 */
template <typename ConcreteType>
class CFDlang_AtomInference : public TraitBase<ConcreteType, CFDlang_AtomInference> {
    using AtomOpModel = cfdlang::detail::AtomOpInterfaceTraits::Model<ConcreteType>;

    static cfdlang::AtomType modelInferAtomType(
        MLIRContext *context,
        Optional<Location> location,
        ValueRange operands,
        DictionaryAttr attributes,
        RegionRange regions
    )
    {
        return AtomOpModel::inferAtomType(
            context,
            location,
            operands,
            attributes,
            regions
        );
    }

public:
    // InferTypeOpInterface
    static LogicalResult inferReturnTypes(
        MLIRContext *context,
        Optional<Location> location,
        ValueRange operands,
        DictionaryAttr attributes,
        RegionRange regions,
        SmallVectorImpl<Type> &inferredReturnTypes
    )
    {
        if (
            const auto atomType = modelInferAtomType(
                context,
                location,
                operands,
                attributes,
                regions
            )
        ) {
            inferredReturnTypes.push_back(atomType);
            return success();
        }

        return failure();
    }

    // InferShapedTypeOpInterface
    // NOTE: Takes precedence over ReifyRankedShapedTypeOpInterface, but is
    //       worse for our purposes. Additionally, methods have default impl.,
    //       so this delegate-to-trait mechanism does not work, as the call will
    //       be ambiguous. Don't see a reason to support this at all, but this
    //       is how:
    // static LogicalResult inferReturnTypeComponents(
    //     MLIRContext *context,
    //     Optional<Location> location,
    //     ValueRange operands,
    //     DictionaryAttr attributes,
    //     RegionRange regions,
    //     SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes
    // )
    // {
    //     if (
    //         const auto atomType = modelInferAtomType(
    //             context,
    //             location,
    //             operands,
    //             attributes,
    //             regions
    //         )
    //     ) {
    //         inferredReturnShapes.emplace_back(
    //             atomType.getShape(),
    //             atomType.getElementType()
    //         );
    //         return success();
    //     }

    //     return failure();
    // }
    // LogicalResult reifyReturnTypeShapes(
    //     OpBuilder &builder,
    //     ValueRange operands,
    //     SmallVectorImpl<Value> &reifiedReturnShapes
    // )
    // {
    //     ReifiedRankedShapedTypeDims resultShapes;
    //     if (failed(reifyResultShapes(builder, resultShapes))) return failure();

    //     for (auto resultShape : resultShapes) {
    //         reifiedReturnShapes.push_back(
    //             builder.create<tensor::FromElementsOp>(
    //                 this->getOperation()->getLoc(),
    //                 resultShape
    //             ).getResult()
    //         );
    //     }

    //     return success();
    // }

    // ReifyRankedShapedTypeOpInterface
    LogicalResult reifyResultShapes(
        OpBuilder &builder,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes
    )
    {
        auto concreteOp = cast<ConcreteType>(this->getOperation());

        auto atomSize = concreteOp.reifyAtomSize(builder);
        if (failed(atomSize)) return failure();

        return cfdlang::interface_defaults::reifyResultShapes(
            *atomSize,
            builder,
            concreteOp.getLoc(),
            reifiedReturnShapes
        );
    }
};

} // namespace mlir::OpTrait
