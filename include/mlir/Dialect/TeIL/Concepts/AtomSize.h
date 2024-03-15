/** Declares the TeIL atom size concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/iterator.h"
#include "mlir/Dialect/TeIL/Concepts/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

namespace mlir::teil {

/** Represents the runtime size of an Atom.
 *
 * This type deliberately allows for multiple representations of the same
 * runtime size, as both a static size and a dynamic Value can be stored for
 * each dimension.
 *
 * To compare values of this type, folding should be applied first, unless
 * trivial equality is sufficient.
 */
struct AtomSize {
    /** Container value type. */
    using value_type = std::pair<dim_size_t, DimSize>;
    /** Container size type. */
    using size_type = shape_t::size_type;

    /** Container iterator type. */
    struct iterator : llvm::iterator_facade_base<
        iterator,
        std::input_iterator_tag,
        value_type,
        std::ptrdiff_t,
        value_type,
        value_type
    > {
        /*implicit*/ iterator() = default;
        /*implicit*/ iterator(const iterator&) = default;
        explicit iterator(
            shape_t::iterator shape,
            ArrayRef<DimSize>::iterator values
        ) : m_shape(shape),
            m_values(values)
        {}

        iterator& operator=(const iterator&) = default;
        bool operator==(const iterator &rhs) const
        {
            return m_shape == rhs.m_shape;
        }
        value_type& operator*() const
        {
            // NOTE: This has to be a stashing iterator.
            m_result = std::make_pair(*m_shape, *m_values);
            return m_result;
        }
        iterator& operator++()
        {
            ++m_shape;
            ++m_values;
            return *this;
        }

    private:
        shape_t::iterator m_shape;
        ArrayRef<DimSize>::iterator m_values;
        mutable value_type m_result;
    };

    /** Gets a static AtomSize.
     *
     * @pre     `countDynamicDims(shape) == 0`
     */
    template<class ShapeRange>
    static AtomSize getStatic(ShapeRange &&shape)
    {
        assert(countDynamicDims(shape) == 0);

        AtomSize result;
        result.m_shape.assign(shape.begin(), shape.end());
        result.m_values.resize(shape.size(), Value());
        return result;
    }
    /** Gets a dynamic AtomSize.
     *
     * @pre     `llvm::all_of(sizes, [](Value x){ return x.isa<DimSize>(); })`
     */
    static AtomSize getDynamic(ValueRange sizes)
    {
        assert(llvm::all_of(sizes, [](Value x){ return x.isa<DimSize>(); }));

        AtomSize result;
        result.m_shape.resize(sizes.size(), dynamic_size);
        result.m_values.reserve(sizes.size());
        llvm::copy(
            llvm::map_range(sizes, [](Value x){ return x.cast<DimSize>(); }),
            std::back_inserter(result.m_values)
        );
        return result;
    }
    /** Gets a mixed AtomSize.
     *
     * For every dynamic dimension in @p shape , @p args is expected to contain
     * a Value that holds the size at runtime.
     *
     * @pre     `args.size() == countDynamicDims(shape)`
     * @pre     `llvm::all_of(args, [](Value x){ return x.isa<DimSize>(); })`
     */
    template<class ShapeRange>
    static AtomSize getMixed(ShapeRange &&shape, ValueRange args)
    {
        AtomSize result;
        auto arg = args.begin();
        for (
            auto [sz_in, sz_out, dyn_out] = std::make_tuple(
                shape.begin(),
                result.m_shape.begin(),
                result.m_values.begin()
            );
            sz_in != shape.end();
            ++sz_in,++sz_out,++dyn_out
        ) {
            *sz_out = *sz_in;
            if (*sz_in == dynamic_size) {
                assert(arg != args.end() && *arg);
                *dyn_out = (*arg++).cast<DimSize>();
            } else {
                *dyn_out = DimSize();
            }
        }
        assert(arg == args.end());
        return result;
    }

    /** Initializes the scalar AtomSize. */
    explicit AtomSize() = default;
    /** Initializes an AtomSize from @p shape and @p values . */
    explicit AtomSize(shape_t shape, ArrayRef<DimSize> values)
        : m_shape(shape.begin(), shape.end()),
          m_values(values.begin(), values.end())
    {}

    AtomSize(AtomSize&&) = default;
    AtomSize(const AtomSize&) = default;

    AtomSize& operator=(AtomSize&&) = default;
    AtomSize& operator=(const AtomSize&) = default;

    /** Gets the rank of this shape. */
    inline rank_t size() const { return m_shape.size(); }
    /** Gets a value indicating whether this is the scalar shape. */
    inline bool empty() const { return m_shape.empty(); }
    /** Gets the range start iterator. */
    inline iterator begin() const
    {
        return iterator(m_shape.begin(), m_values.begin());
    }
    /** Gets the range end iterator. */
    inline iterator end() const
    {
        return iterator(m_shape.end(), {});
    }

    /** Verifies that this AtomSize is concrete.
     *
     * An AtomSize is concrete if every dimension has either a static size, or
     * a dynamic size value attached.
     */
    inline LogicalResult verify()
    {
        for (
            auto [sz, val] = std::make_pair(
                getShape().begin(),
                getValues().begin()
            );
            sz != getShape().end();
            ++sz,++val
        ) {
            if (*sz == dynamic_size && !*val) {
                // Size is not concrete.
                return failure();
            }
        }

        return success();
    }

    /** Attempts to fold this AtomSize in place.
     *
     * After constant folding the dynamic values into @p args , this method will
     * merge the constant results into the static shape where possible.
     *
     * @pre         `args.size() == countDynamicDims(getShape())`
     *
     * @retval      success()   Folding has occurred.
     * @retval      failure()   Nothing was folded.
     */
    LogicalResult fold(ArrayRef<DimSizeAttr> args);
    /** Attempts to reify this AtomSize in place.
     *
     * For every statically sized dimension in this AtomSize, ensure that it is
     * represented by a Value
     *
     * @retval      success()   There are no undefined values.
     * @retval      failure()   The AtomSize is not concrete.
     */
    LogicalResult reify(OpBuilder &builder, Location loc);

    /** @copydoc empty() */
    inline bool isScalar() const { return empty(); }
    /** @copydoc size() */
    inline rank_t getRank() const { return size(); }
    /** Gets the shape. */
    inline shape_t getShape() const { return m_shape; }
    /** Gets the runtime values. */
    inline ArrayRef<DimSize> getValues() const { return m_values; }
    /** Gets the dynamic size arguments. */
    inline auto getArgs() const
    {
        return llvm::map_range(
            llvm::make_filter_range(
                *this,
                [](auto x) { return x.first == dynamic_size; }
            ),
            [](auto x) { return x.second; }
        );
    }

    /** @copydoc getShape() */
    /*implicit*/ operator shape_t() const { return getShape(); }
    /** @copydoc getValues() */
    /*implicit*/ operator ArrayRef<DimSize>() const { return getValues(); }

    /** Determines whether getShape() is equal to @p shape . */
    inline bool operator==(shape_t shape) const
    {
        return llvm::equal(getShape(), shape);
    }
    /** Determines whether getShape() is not equal to @p shape . */
    inline bool operator!=(shape_t shape) const
    {
        return !(*this == shape);
    }

    /** Determines whether this AtomSize is trivially equal to @p rhs . */
    inline bool operator==(const AtomSize &rhs) const
    {
        if (*this != rhs.getShape()) {
            // Shapes must match.
            return false;
        }

        // Dynamic size arguments must match.
        const auto lhs_args = getArgs();
        const auto rhs_args = rhs.getArgs();
        return std::mismatch(
            lhs_args.begin(),
            lhs_args.end(),
            rhs_args.begin()
        ) == std::make_pair(lhs_args.end(), rhs_args.end());
    }
    /** Determines whether this AtomSize is not trivially equal to @p rhs . */
    inline bool operator!=(const AtomSize &rhs) const
    {
        return !(*this == rhs);
    }

private:
    ShapeStorage m_shape;
    SmallVector<DimSize> m_values;
};

} // namespace mlir::teil
