/** Declares the TeIL atom concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/TeIL/Concepts/Scalar.h"
#include "mlir/Dialect/TeIL/Concepts/Shape.h"

namespace mlir::teil {

/** Atom type concept. */
struct AtomType : ConstrainedType<RankedTensorType, AtomType> {
    /** Checks whether @p rankedTensorType is an atom type. */
    static inline bool matches(RankedTensorType rankedTensorType)
    {
        return rankedTensorType.getElementType().isa<ScalarType>()
            && !isTriviallyEmpty(rankedTensorType.getShape());
    }

    /** Obtains the AtomType for @p scalarType and @p shape . */
    static inline AtomType get(
        ScalarType scalarType,
        shape_t shape = scalar_shape
    )
    {
        return RankedTensorType::get(shape, scalarType).cast<AtomType>();
    }

    using ConstrainedType<RankedTensorType, AtomType>::ConstrainedType;

    /** Gets the scalar type. */
    inline ScalarType getScalarType() const
    {
        return RankedTensorType::getElementType().cast<ScalarType>();
    }
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

    /** @copydoc AtomType::getScalarType(). */
    inline ScalarType getScalarType() const
    {
        return getType().getScalarType();
    }
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

} // namespace mlir::teil
