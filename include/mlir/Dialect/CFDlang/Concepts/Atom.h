/** Declares the CFDlang atom concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Concepts.h"
#include "mlir/Dialect/TeIL/Concepts/Shape.h"
#include "mlir/Dialect/TeIL/IR/Types.h"

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
        shape_t shape = scalar_shape
    )
    {
        return RankedTensorType::get(ScalarType::get(context), shape)
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

} // namespace mlir::cfdlang
