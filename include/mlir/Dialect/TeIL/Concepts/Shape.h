/** Declares the TeIL shape concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/iterator.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Concepts/Fluent.h"
#include "mlir/Dialect/TeIL/IR/Base.h"

#include <algorithm>
#include <variant>

namespace mlir::teil {

/** Provides mutable access to build a shape_t. */
using ShapeBuilder = SmallVectorImpl<dim_size_t>;
/** Stores a shape_t. */
using ShapeStorage = SmallVector<dim_size_t>;

//===----------------------------------------------------------------------===//
// Shape algorithms
//===----------------------------------------------------------------------===//

/** Determines whether @p shape has dimensions of size 0. */
template<class ShapeRange>
inline auto isTriviallyEmpty(ShapeRange &&shape)
{
    return llvm::any_of(
        std::forward<ShapeRange>(shape),
        [](auto sz){ return sz == 0; }
    );
}

/** Count the number of dynamically sized dimensions in @p shape . */
template<class ShapeRange>
inline std::size_t countDynamicDims(ShapeRange &&shape)
{
    return static_cast<std::size_t>(
        llvm::count(std::forward<ShapeRange>(shape), dynamic_size)
    );
}

/** Attempt to calculate the extent of @p shape in elements.
 *
 * If @p shape is trivially empty, the result is guaranteed to be 0, regardless
 * of whether dynamic dimensions are involved. Otherwise, dynamic dimensions or
 * integer overflows result in @c None .
 *
 * @param   [in]        shape   The shape.
 *
 * @retval  0           @p shape is trivially empty.
 * @retval  nullopt     @p shape has a too large or dynamic extent.
 * @retval  size_t      The static extent of @p shape .
 */
template<class ShapeRange>
inline std::optional<std::size_t> calculateSmallExtent(ShapeRange &&shape)
{
    std::size_t result{0};
    for (auto it = shape.begin(); it != shape.end(); ++it) {
        if (*it == 0) {
            // shape is trivially empty.
            return size_t{0};
        }

        if (*it == dynamic_size) {
            // Dynamically sized shapes do not have a static extent, but might
            // still by trivially empty.
            if (
                std::any_of(
                    std::next(it),
                    shape.end(),
                    [](auto sz){ return sz == 0; }
                )
            ) {
                // shape is trivially empty.
                return size_t{0};
            }

            // shape has a dynamic extent.
            return std::nullopt;
        }

        if (__builtin_mul_overflow(result, static_cast<size_t>(*it), &result)) {
            // Calculating the extent has overflowed.
            return std::nullopt;
        }
    }

    // Successfuly calculated a static extent for shape.
    return result;
}

/** Determines whether @p lhs is compatible (foldable) with @p rhs .
 *
 * If @p lhs and @p rhs contains mismatching non-dynamic dimensions, they will
 * be rejected.
 *
 * @param   [in]        lhs Left shape.
 * @param   [in]        rhs Right shape.
 *
 * @return  Whether @p lhs and @p rhs are compatible.
 */
inline bool are_compatible(shape_t lhs, shape_t rhs)
{
    if (lhs.size() != rhs.size()) {
        // Rank mismatch.
        return false;
    }

    for (
        auto [l, r] = std::make_pair(lhs.begin(), rhs.begin());
        l != lhs.end();
        ++l,++r
    ) {
        if (*l != *r && *l != dynamic_size && *r != dynamic_size) {
            // Dimension mismatch.
            return false;
        }
    }

    return true;
}

/** Attempts to fold @p rhs into @p lhs .
 *
 * If @p lhs contains dynamic dimensions, they will be replaced by their
 * respective counterparts in @p rhs . If both offer mismatched sizes, the
 * fold operation will fail.
 *
 * @param   [in,out]    lhs Left shape.
 * @param   [in]        rhs Right shape.
 *
 * @returns Whether folding was successful.
 */
template<class ShapeRange>
inline LogicalResult fold(ShapeBuilder &lhs, ShapeRange &&rhs)
{
    auto r_it = rhs.begin();
    for (auto l_it = lhs.begin(); l_it != lhs.end(); ++l_it,++r_it) {
        if (r_it == rhs.end()) {
            // rhs has lower rank.
            return failure();
        }

        if (*l_it == dynamic_size) {
            // Fold rhs into lhs.
            *l_it = *r_it;
            continue;
        }

        if (*l_it != *r_it && *r_it != dynamic_size) {
            // Dimension sizes mismatched.
            return failure();
        }
    }

    return success(r_it == rhs.end());
}

//===----------------------------------------------------------------------===//
// Shape concepts
//===----------------------------------------------------------------------===//

/** Dimension size type concept. */
struct DimSizeType : ConstrainedType<IntegerType, DimSizeType> {
    /** Determines whether @p intType is a DimSizeType. */
    static inline bool matches(IntegerType intType)
    {
        return intType.getWidth() <= 64 && !intType.isUnsigned();
    }

    /** Obtains the DimSizeType instance. */
    static inline DimSizeType get(MLIRContext *context)
    {
        return IntegerType::get(context, 64U).cast<DimSizeType>();
    }

    using ConstrainedType<IntegerType, DimSizeType>::ConstrainedType;
};

/** Dimension size value concept. */
struct DimSize : ConstrainedValue<DimSizeType> {
    using ConstrainedValue<DimSizeType>::ConstrainedValue;
};

/** Dimension size attribute concept. */
struct DimSizeAttr : ConstrainedAttribute<IntegerAttr, DimSizeType> {
    /** The underlying value type. */
    using ValueType = dim_size_t;

    /** Obtains a DimSizeAttr for @p value . */
    static inline DimSizeAttr get(MLIRContext *context, ValueType value)
    {
        return IntegerAttr::get(
            DimSizeType::get(context),
            APInt(64U, value, true)
        ).cast<DimSizeAttr>();
    }

    using ConstrainedAttribute<IntegerAttr, DimSizeType>::ConstrainedAttribute;

    /** Gets the underlying value. */
    inline ValueType getValue() const
    {
        return IntegerAttr::getValue().getSExtValue();
    }
};

/** Dimension size array attribute concept. */
struct DimSizeArrayAttr : ConstrainedArrayAttribute<DimSizeAttr> {
    /** Obtains a DimSizeArrayAttr for @p values . */
    static inline DimSizeArrayAttr get(
        MLIRContext *context,
        ArrayRef<dim_size_t> values
    )
    {
        return ArrayAttr::get(
            context,
            to_vector(
                llvm::map_range(
                    values,
                    [=](dim_size_t x) -> Attribute {
                        return DimSizeAttr::get(context, x);
                    }
                )
            )
        ).cast<DimSizeArrayAttr>();
    }

    using ConstrainedArrayAttribute<DimSizeAttr>::ConstrainedArrayAttribute;
};

/** Shape type concept. */
struct ShapeType : ConstrainedType<RankedTensorType, ShapeType> {
    /** Determines whether @p rankedTensorType is a shape type. */
    static inline bool matches(RankedTensorType rankedTensorType)
    {
        return rankedTensorType.getElementType().isa<DimSizeType>()
            && rankedTensorType.getRank() == 1;
    }

    /** Obtains the ShapeType for @p rank . */
    static inline ShapeType get(MLIRContext *context, rank_t rank)
    {
        auto shape = static_cast<int64_t>(rank);
        return RankedTensorType::get(
            llvm::ArrayRef(shape),
            DimSizeType::get(context)
        ).cast<ShapeType>();
    }

    using ConstrainedType<RankedTensorType, ShapeType>::ConstrainedType;
};

/** Shape value concept. */
struct Shape : ConstrainedValue<ShapeType> {
    using ConstrainedValue<ShapeType>::ConstrainedValue;

    /** Get the rank of this shape */
    inline rank_t getRank() const { return getType().getRank(); }
};

/** Dense shape attribute concept. */
struct DenseShapeAttr : ConstrainedAttribute<DenseIntElementsAttr, ShapeType> {
private:
    struct extract_f {
        inline dim_size_t operator()(const APInt &value) const
        {
            return value.getSExtValue();
        }
    };

public:
    /** Iterator type. */
    using iterator = llvm::mapped_iterator<
        DenseIntElementsAttr::iterator,
        extract_f
    >;
    /** Type of the underlying value. */
    using ValueType = ShapeStorage;

    /** Obtains a DenseShapeAttr for @p values . */
    static inline DenseShapeAttr get(MLIRContext *context, shape_t values)
    {
        return DenseIntElementsAttr::get(
            ShapeType::get(context, values.size()),
            values
        ).cast<DenseShapeAttr>();
    }

    using ConstrainedAttribute<DenseIntElementsAttr, ShapeType>
        ::ConstrainedAttribute;

    /** Gets the range start iterator. */
    inline iterator begin() const
    {
        return llvm::map_iterator(DenseIntElementsAttr::begin(), extract_f{});
    }
    /** Gets the range end iterator. */
    inline iterator end() const
    {
        return llvm::map_iterator(DenseIntElementsAttr::end(), extract_f{});
    }

    /** Copies the constrained attribute value to @p result . */
    inline void getValue(ShapeBuilder &result) const
    {
        result.clear();
        result.reserve(size());
        llvm::copy(*this, std::back_inserter(result));
    }
    /** Creates a copy of the constrained attribute value. */
    inline ValueType getValue() const
    {
        ValueType result;
        getValue(result);
        return result;
    }
};

} // namespace mlir::teil
