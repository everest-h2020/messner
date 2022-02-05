/** Declares the TeIL scalar concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/TeIL/IR/Types.h"

namespace mlir::teil {

/** Scalar type concept. */
struct ScalarType : ConstrainedType<Type, ScalarType> {
    /** Determines whether @p type is a scalar type. */
    static inline bool matches(Type type)
    {
        return type.isa<FloatType, NumberType>();
    }

    using ConstrainedType<Type, ScalarType>::ConstrainedType;
};

/** Scalar value concept. */
struct Scalar : ConstrainedValue<ScalarType> {
    using ConstrainedValue<ScalarType>::ConstrainedValue;
};

} // namespace mlir::teil
