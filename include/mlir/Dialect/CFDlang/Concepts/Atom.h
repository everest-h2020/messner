/** Declares the CFDlang atom concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Fluent.h"
#include "mlir/Dialect/CFDlang/IR/Types.h"
#include "mlir/Dialect/TeIL/Concepts/AtomSize.h"
#include "mlir/Dialect/TeIL/Concepts/Shape.h"

namespace mlir::cfdlang {

/** Atom type concept. */
struct AtomType : ConstrainedType<RankedTensorType, AtomType> {
    /** Checks whether @p rankedTensorType is an atom type. */
    static inline bool matches(RankedTensorType rankedTensorType)
    {
        return rankedTensorType.getElementType().isa<ScalarType>()
            && !teil::isTriviallyEmpty(rankedTensorType.getShape());
    }

    /** Obtains the AtomType for @p shape . */
    static inline AtomType get(
        MLIRContext *context,
        shape_t shape = teil::scalar_shape
    )
    {
        return RankedTensorType::get(shape, ScalarType::get(context))
            .cast<AtomType>();
    }

    using ConstrainedType<RankedTensorType, AtomType>::ConstrainedType;

    /** Gets the rank. */
    inline rank_t getRank() const
    {
        return static_cast<rank_t>(RankedTensorType::getRank());
    }
    /** Gets the shape. */
    inline shape_t getShape() const
    {
        return RankedTensorType::getShape();
    }
};

/** Atom value concept. */
struct Atom : ConstrainedValue<AtomType> {
    using ConstrainedValue<AtomType>::ConstrainedValue;

    /** @copydoc AtomType::getRank(). */
    inline rank_t getRank() const
    {
        return getType().getRank();
    }
    /** @copydoc AtomType::getShape(). */
    inline shape_t getShape() const
    {
        return getType().getShape();
    }
};

struct AtomTypeAttr : Concept<TypeAttr> {
    using ValueType = AtomType;

    static inline bool classof(TypeAttr attr)
    {
        return attr.getValue().isa<AtomType>();
    }
    static inline bool classof(Attribute attr)
    {
        if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
            return classof(typeAttr);
        }

        return false;
    }

    static AtomTypeAttr get(AtomType atomType)
    {
        return TypeAttr::get(atomType).cast<AtomTypeAttr>();
    }
    static AtomTypeAttr get(MLIRContext *context, shape_t atomShape)
    {
        return get(AtomType::get(context, atomShape));
    }

    using Concept<TypeAttr>::Concept;

    inline AtomType getValue() { return TypeAttr::getValue().cast<AtomType>(); }
};

struct AtomTypeArrayAttr : ConstrainedArrayAttribute<AtomTypeAttr> {
    template<class ShapesRange>
    static AtomTypeArrayAttr get(MLIRContext *context, ShapesRange &&shapes)
    {
        return ArrayAttr::get(
            context,
            to_vector(
                llvm::map_range(
                    shapes,
                    [=](shape_t x) -> Attribute {
                        return AtomTypeAttr::get(context, x);
                    }
                )
            )
        ).template cast<AtomTypeArrayAttr>();
    }

    using ConstrainedArrayAttribute<AtomTypeAttr>::ConstrainedArrayAttribute;
};

} // namespace mlir::cfdlang
