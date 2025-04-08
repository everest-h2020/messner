/// Declares the Index type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/Analysis/Offset.h"
#include "messner/Support/arith_result.h"
#include "messner/Support/fallible_arith.h"

#include <compare>
#include <llvm/ADT/Hashing.h>
#include <optional>
#include <type_traits>

namespace mlir::ekl {

/// Represents an address into an indexed sequence.
///
/// This class implements an index type for sequences that can address elements
/// relative to both the start and end of the sequence. It can be in one of two
/// distinct states:
///
///     - Relative to begin
///
///       The stored offset is used directly as the offset into the sequence. An
///       offset of -1 is the "past-the-end" iterator for reverse iteration.
///
///     - Relative to end
///
///       The stored offset is added to the "past-the-end" offset of the
///       sequence, such that -1 addresses the last element of the sequence. An
///       offset of 0 is the "past-the-end" iterator for forward iteration.
class Index : messner::mixin::fallible_arith<Index> {
public:
    /// Type that holds the result of arithmetic operations.
    using arith_result = messner::arith_result<Index>;

    /// Obtains the address of the first element in a sequence.
    static constexpr auto begin() -> Index { return Index(0); }
    /// Obtains the address of the element after the last in a sequence.
    static constexpr auto end() -> Index { return Index(0, true); }
    /// Obtains the address of the last element in a sequence.
    static constexpr auto rbegin() -> Index { return Index(-1, true); }
    /// Obtains the address of the element before the first in a sequence.
    static constexpr auto rend() -> Index { return Index(-1); }

    /// Initializes the 0 index.
    /*implicit*/ constexpr Index() = default;
    /// Initializes an Index using @p offset relative to @p fromEnd .
    explicit constexpr Index(Offset offset, bool fromEnd = false);

    /// Gets the underlying offset value.
    constexpr auto getOffset() const -> Offset { return _offset; }
    /// Indicates whether the offset is relative to the end.
    constexpr auto isFromEnd() const -> bool { return _fromEnd; }

    /// Determines whether the index is trivially out of bounds.
    ///
    /// An index is trivially out of bounds if it is relative to the start and
    /// its offset is less than 0, or relative to the end and its offset is
    /// greater or equal to 0.
    constexpr auto isTriviallyOutOfBounds() const -> bool;
    /// Obtains an the nearest potentially in-bounds index.
    constexpr auto clamp() const -> Index;

    /// Converts into an offset from the start of some @p extent .
    ///
    /// The result is defined if @p extent is bounded and the result offset is
    /// within the representable value range.
    constexpr auto toAbsolute(Extent extent) const -> std::optional<Offset>;
    /// Converts into an in-bounds offset from the start of some @p extent .
    ///
    /// The result is defined iff the resulting offset is an in-bounds address
    /// into the @p extent .
    constexpr auto inBounds(Extent extent) const -> std::optional<Offset>;
    /// Obtains the minimum extent into which this is an in-bounds index.
    ///
    /// The result is defined iff the index is not trivially out of bounds (
    /// i.e., a negative absolute or non-negative relative index).
    ///
    /// @post   `!result || result->isBounded()`
    constexpr auto getBound() const -> std::optional<Extent>;

    template<std::size_t I>
    friend constexpr auto get(const Index &index)
    {
        if constexpr (I == 0)
            return index.getOffset();
        else
            return index.isFromEnd();
    }

    /// Adds two indices with the same reference point.
    ///
    /// The result is defined iff both indices have the same reference point,
    /// and the result is a representable offset.
    constexpr auto operator+(const Index &rhs) const -> arith_result;
    /// Advances an index by a value.
    ///
    /// The result is defined iff it is a representable offset.
    constexpr auto operator+(Offset rhs) const -> arith_result;

    /// Subtracts two indices with the same reference point.
    ///
    /// The result is defined iff both indices have the same reference point,
    /// and the result is a representable offset.
    constexpr auto operator-(const Index &rhs) const -> arith_result;
    /// Recedes an index by a value.
    ///
    /// The result is defined iff it is a representable offset.
    constexpr auto operator-(Offset rhs) const -> arith_result;

    /// @copydoc messner::saturate()
    friend constexpr auto saturate(arith_result result) -> std::optional<Index>;

    /// Determines whether two indices are definitionally equal.
    constexpr auto operator==(const Index &) const -> bool = default;

    friend auto hash_value(const Index &index) -> llvm::hash_code;

    /// Compares two indices, if they have the same reference point.
    constexpr auto operator<=>(const Index &rhs) const -> std::partial_ordering;

private:
    Offset _offset;
    bool _fromEnd;
};

} // namespace mlir::ekl

namespace std {

template<>
struct tuple_size<::mlir::ekl::Index> : integral_constant<size_t, 2> {};

template<size_t I>
struct tuple_element<I, ::mlir::ekl::Index> {
    using type = conditional_t<I == 0, ::mlir::ekl::Offset, bool>;
};

template<>
struct hash<::mlir::ekl::Index> {
    constexpr auto operator()(const ::mlir::ekl::Index &index) const -> size_t;
};

} // namespace std

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Index implementation
//===----------------------------------------------------------------------===//

constexpr Index::Index(Offset offset, bool fromEnd)
        : _offset(offset),
          _fromEnd(fromEnd)
{}

constexpr auto Index::isTriviallyOutOfBounds() const -> bool
{
    return isFromEnd() ? getOffset() >= 0 : getOffset() < 0;
}

constexpr auto Index::clamp() const -> Index
{
    if (isTriviallyOutOfBounds()) return isFromEnd() ? rbegin() : begin();
    return *this;
}

constexpr auto Index::toAbsolute(Extent extent) const -> std::optional<Offset>
{
    if (!isFromEnd()) return getOffset();
    if (!extent.isBounded()) return std::nullopt;
    return (Offset(extent.getValue()) + getOffset()).exact();
}

constexpr auto Index::inBounds(Extent extent) const -> std::optional<Offset>
{
    const auto maybeOffset = toAbsolute(extent);
    if (!maybeOffset || *maybeOffset < 0 || *maybeOffset >= extent.getValue())
        return std::nullopt;
    return *maybeOffset;
}

constexpr auto Index::getBound() const -> std::optional<Extent>
{
    if (!isFromEnd()) {
        if (getOffset() <= 0) return std::nullopt;
        const auto distance =
            static_cast<Extent::value_type>(getOffset().getValue());
        return Extent(distance + 1U);
    }

    if (getOffset() >= 0) return std::nullopt;
    const auto size = static_cast<Extent::value_type>(-getOffset().getValue());
    return Extent(size);
}

constexpr auto Index::operator<=>(const Index &rhs) const
    -> std::partial_ordering
{
    if (isFromEnd() ^ rhs.isFromEnd()) return std::partial_ordering::unordered;
    return getOffset() <=> rhs.getOffset();
}

constexpr auto Index::operator+(const Index &rhs) const -> arith_result
{
    if (isFromEnd() ^ rhs.isFromEnd()) return std::nullopt;
    const auto offset = getOffset() + rhs.getOffset();
    return arith_result(offset.quality(), offset.value(), isFromEnd());
}

constexpr auto Index::operator+(Offset rhs) const -> arith_result
{
    const auto offset = isFromEnd() ? getOffset() - rhs : getOffset() + rhs;
    return arith_result(offset.quality(), offset.value(), isFromEnd());
}

constexpr auto Index::operator-(const Index &rhs) const -> arith_result
{
    if (isFromEnd() ^ rhs.isFromEnd()) return std::nullopt;
    const auto offset = getOffset() - rhs.getOffset();
    return arith_result(offset.quality(), offset.value(), isFromEnd());
}

constexpr auto Index::operator-(Offset rhs) const -> arith_result
{
    const auto offset = isFromEnd() ? getOffset() + rhs : getOffset() - rhs;
    return arith_result(offset.quality(), offset.value(), isFromEnd());
}

constexpr auto saturate(Index::arith_result result) -> std::optional<Index>
{
    switch (result.quality()) {
    case messner::arith_quality::undefined: return std::nullopt;
    case messner::arith_quality::rounded:
    case messner::arith_quality::exact:     return result.value();
    case messner::arith_quality::underflow:
        return Index(Offset::min(), result.value().isFromEnd());
    case messner::arith_quality::overflow:
        return Index(Offset::max(), result.value().isFromEnd());
    }
}

inline auto hash_value(const Index &index) -> llvm::hash_code
{
    return std::hash<Index>{}(index);
}

} // namespace mlir::ekl

namespace std {

//===----------------------------------------------------------------------===//
// hash<::mlir::ekl::Index> implementation
//===----------------------------------------------------------------------===//

constexpr auto
hash<::mlir::ekl::Index>::operator()(const ::mlir::ekl::Index &index) const
    -> size_t
{
    const auto mask = index.isFromEnd() ? -1 : 0;
    return static_cast<size_t>(index.getOffset().getValue() ^ mask);
}

} // namespace std
