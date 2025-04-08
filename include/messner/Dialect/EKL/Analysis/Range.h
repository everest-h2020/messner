/// Declares the Range type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Offset.h"
#include "messner/Support/arith_result.h"
#include "messner/Support/iter_facade.h"

#include <compare>
#include <cstddef>
#include <iterator>
#include <llvm/ADT/iterator.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::ekl {

/// Represents a strided range of offsets.
///
/// A Range is represented by an inclusive start offset, and exclusive end
/// offset, and a non-zero stride. If the stride is negative, the range is
/// iterated over in order of descending offsets.
class Range {
public:
    using value_type      = const Offset;
    using pointer         = void;
    using reference       = const value_type &;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    /// Implements a strided offset iterator.
    ///
    /// The iterator is saturating, i.e., when an attempt is made to move it
    /// past the value range of the Offset type, it saturates to the closest
    /// extremum. This way, some value will compare equal to the sentinel.
    class iterator : public messner::iter_facade<
                         iterator,
                         std::random_access_iterator_tag,
                         value_type> {
    public:
        /// Initializes an iterator at @p position with @p stride .
        ///
        /// @pre    `stride != 0`
        explicit constexpr iterator(value_type position, Offset stride = 1);

        /*implicit*/ constexpr iterator(const iterator &) = default;
        iterator &operator=(const iterator &)             = default;

        /// Obtains the current position of the iterator.
        constexpr auto getPosition() const -> Offset { return _position; }
        /// Obtains the stride of the iterator.
        ///
        /// @post   `stride != 0`
        constexpr auto getStride() const -> Offset { return _stride; }

        constexpr auto operator*() const -> reference { return _position; }
        constexpr auto operator-(const iterator &rhs) const -> difference_type;
        constexpr auto operator+=(difference_type dist) -> iterator &;
        constexpr auto operator-=(difference_type dist) -> iterator &;
        constexpr auto operator==(const iterator &) const -> bool;
        constexpr auto operator<=>(const iterator &rhs) const
            -> std::strong_ordering;

    private:
        Offset _position;
        Offset _stride;
    };

    /// Implements a sentinel for a strided offset iterator.
    ///
    /// The sentinel compares the iterator against the contained bound, with the
    /// relation being defined by the sign of the stride. Since the iterator is
    /// saturating, the sentinel will always compare equal at some point, if the
    /// range was not empty.
    class sentinel {
    public:
        explicit constexpr sentinel(Offset bound, bool isLowerBound);

        /*implicit*/ constexpr sentinel(const sentinel &) = default;
        sentinel &operator=(const sentinel &)             = default;

        /// Obtains the bound value of this sentinel.
        constexpr auto getBound() const -> Offset { return _bound; }
        /// Indicates whether this sentinel is a lower bound.
        constexpr auto isLowerBound() const -> bool { return _isLowerBound; }

        constexpr auto operator==(const sentinel &) const -> bool = default;
        constexpr auto operator==(const iterator &it) const -> bool;
        friend constexpr auto
        operator==(const iterator &it, const sentinel &sent) -> bool;

    private:
        Offset _bound;
        bool _isLowerBound;
    };

    /// Initializes an empty Range.
    /*implicit*/ constexpr Range() : Range(0, 0) {}
    /// Initializes a Range from @p begin to @p end with @p stride .
    ///
    /// @pre    `stride != 0`
    explicit constexpr Range(Offset begin, Offset end, Offset stride = 1);

    /// Obtains the begin offset.
    constexpr auto getBegin() const -> Offset { return _begin; }
    /// Obtains the end offset.
    constexpr auto getEnd() const -> Offset { return _end; }
    /// Obtains the stride of this range.
    ///
    /// @post   `stride != 0`
    constexpr auto getStride() const -> Offset { return _stride; }

    /// Obtains a range of the first @p count elements.
    ///
    /// If @p count is not positive or greater than `size()`, the resulting
    /// Range is empty.
    constexpr auto take_front(Offset::value_type count) const -> Range;
    /// Obtains a range without the first @p count elements.
    ///
    /// If @p count is greater than `size()`, the resulting Range is empty. If
    /// it is negative, the current range is returned.
    constexpr auto drop_front(Offset::value_type count) const -> Range;
    /// Obtains a range of the last @p count elements.
    ///
    /// If @p count is not positive or greater than `size()`, the resulting
    /// Range is empty.
    constexpr auto take_back(Offset::value_type count) const -> Range;
    /// Obtains a range without the last @p count elements.
    ///
    /// If @p count is greater than `size()`, the resulting Range is empty. If
    /// it is negative, the current range is returned.
    constexpr auto drop_back(Offset::value_type count) const -> Range;

    /// Determines whether the range is empty.
    constexpr auto empty() const -> bool;
    /// Determines the size of the range in elements.
    constexpr auto size() const -> std::size_t;
    /// Obtains the begin iterator.
    constexpr auto begin() const -> iterator;
    /// Obtains the end iterator.
    constexpr auto end() const -> sentinel;

    /// Converts to @c true if the range is not empty.
    explicit constexpr operator bool() const { return !empty(); }

    /// Determines whether two ranges are definitionally equal.
    constexpr auto operator==(const Range &) const -> bool = default;

private:
    Offset _begin, _end, _stride;
};

} // namespace mlir::ekl

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Range::iterator implementation
//===----------------------------------------------------------------------===//

constexpr Range::iterator::iterator(Offset position, Offset stride)
        : _position(position),
          _stride(stride)
{
    assert(stride != 0);
}

constexpr auto Range::iterator::operator-(const iterator &rhs) const
    -> difference_type
{
    const auto l = getPosition().getValue(), r = rhs.getPosition().getValue();

    if (getStride() >= 0) {
        auto bias = _stride.getValue() - 1;
        if (l < r) bias = -bias;
        return (l - r + bias) / getStride().getValue();
    }

    auto bias = _stride.getValue() + 1;
    if (l < r) bias = -bias;
    return (r - l + bias) / -getStride().getValue();
}

constexpr auto Range::iterator::operator+=(difference_type dist) -> iterator &
{
    const auto offset = getStride() * dist;
    switch (offset.quality()) {
    case messner::arith_quality::exact:
    case messner::arith_quality::rounded:
        _position = *saturate(getPosition() + *offset);
        break;
    default: _position = (getStride() < 0) ? Offset::min() : Offset::max();
    }

    return *this;
}

constexpr auto Range::iterator::operator-=(difference_type dist) -> iterator &
{
    const auto offset = getStride() * dist;
    switch (offset.quality()) {
    case messner::arith_quality::exact:
    case messner::arith_quality::rounded:
        _position = *saturate(getPosition() - *offset);
        break;
    default: _position = (getStride() < 0) ? Offset::min() : Offset::max();
    }

    return *this;
}

constexpr auto Range::iterator::operator==(const iterator &rhs) const -> bool
{
    return getPosition() == rhs.getPosition();
}

constexpr auto Range::iterator::operator<=>(const iterator &rhs) const
    -> std::strong_ordering
{
    return getPosition() <=> rhs.getPosition();
}

//===----------------------------------------------------------------------===//
// Range::sentinel implementation
//===----------------------------------------------------------------------===//

constexpr Range::sentinel::sentinel(Offset bound, bool isLowerBound)
        : _bound(bound),
          _isLowerBound(isLowerBound)
{}

constexpr auto Range::sentinel::operator==(const iterator &it) const -> bool
{
    const auto cmp = it.getPosition() <=> getBound();
    return isLowerBound() ? !std::is_gt(cmp) : !std::is_lt(cmp);
}

constexpr auto
operator==(const Range::iterator &it, const Range::sentinel &sent) -> bool
{
    return sent == it;
}

//===----------------------------------------------------------------------===//
// Range implementation
//===----------------------------------------------------------------------===//

constexpr Range::Range(Offset begin, Offset end, Offset stride)
        : _begin(begin),
          _end(end),
          _stride(stride)
{
    assert(stride != 1);
}

constexpr auto Range::take_front(Offset::value_type count) const -> Range
{
    if (count <= 0) return {};

    const auto offset = *saturate(getStride() * (count - 1));
    return Range(getBegin(), *saturate(getBegin() + offset), getStride());
}

constexpr auto Range::drop_front(Offset::value_type count) const -> Range
{
    if (count <= 0) return *this;

    const auto offset = *saturate(getStride() * (count - 1));
    return Range(*saturate(getBegin() + offset), getEnd(), getStride());
}

constexpr auto Range::take_back(Offset::value_type count) const -> Range
{
    if (count <= 0) return {};

    const auto offset = *saturate(getStride() * (count - 1));
    return Range(*saturate(getEnd() - offset), getEnd(), getStride());
}

constexpr auto Range::drop_back(Offset::value_type count) const -> Range
{
    if (count <= 0) return *this;

    const auto offset = *saturate(getStride() * (count - 1));
    return Range(getBegin(), *saturate(getEnd() - offset), getStride());
}

constexpr auto Range::empty() const -> bool
{
    const auto cmp = getBegin() <=> getEnd();
    return (getStride() >= 0) ? !std::is_lt(cmp) : !std::is_gt(cmp);
}

constexpr auto Range::size() const -> std::size_t
{
    if (getStride() >= 0) {
        if (getBegin() >= getEnd()) return 0U;
        const auto distance =
            static_cast<size_type>(getEnd().getValue() - getBegin().getValue())
            - 1U;
        return 1U + (distance / static_cast<size_type>(getStride().getValue()));
    }

    if (getEnd() >= getBegin()) return 0U;
    const auto distance =
        static_cast<size_type>(getBegin().getValue() - getEnd().getValue())
        - 1U;
    return 1U + (distance / static_cast<size_type>(-getStride().getValue()));
}

constexpr auto Range::begin() const -> iterator
{
    return iterator(getBegin(), getStride());
}

constexpr auto Range::end() const -> sentinel
{
    return sentinel(getEnd(), getStride() < 0);
}

} // namespace mlir::ekl
