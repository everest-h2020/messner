/** Declares the TeIL natural number concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Concepts.h"
#include "mlir/Concepts/Fluent.h"
#include "mlir/Dialect/TeIL/IR/Base.h"

namespace mlir::teil {

/** Natural number type concept. */
struct NatType : ConstrainedType<IndexType> {
    /** Obtains the NatType instance. */
    static inline NatType get(MLIRContext* context)
    {
        return IndexType::get(context).cast<NatType>();
    }

    using ConstrainedType<IndexType>::ConstrainedType;
};

/** Natural number value concept. */
struct Nat : ConstrainedValue<NatType> {
    using ConstrainedValue<NatType>::ConstrainedValue;
};

/** Natural number attribute concept. */
struct NatAttr : ConstrainedAttribute<IntegerAttr, NatType> {
    /** The underlying value type. */
    using ValueType = natural_t;

    /** Obtains a NatAttr for @p value . */
    static inline NatAttr get(MLIRContext* context, natural_t value)
    {
        return IntegerAttr::get(NatType::get(context), APInt(64, value, false))
            .cast<NatAttr>();
    }

    using ConstrainedAttribute<IntegerAttr, NatType>::ConstrainedAttribute;

    /** Gets the underlying value. */
    inline ValueType getValue() const
    {
        return IntegerAttr::getValue().getZExtValue();
    }
};

/** Natural number array attribute concept. */
struct NatArrayAttr : ConstrainedArrayAttribute<NatAttr> {
    /** Obtains a NatArrayAttr for @p values . */
    static inline NatArrayAttr get(
        MLIRContext* context,
        ArrayRef<natural_t> values
    )
    {
        return ArrayAttr::get(
            context,
            to_vector(
                llvm::map_range(
                    values,
                    [=](natural_t x) -> Attribute {
                        return NatAttr::get(context, x);
                    }
                )
            )
        ).cast<NatArrayAttr>();
    }

    using ConstrainedArrayAttribute<NatAttr>::ConstrainedArrayAttribute;
};

} // namespace mlir::teil
