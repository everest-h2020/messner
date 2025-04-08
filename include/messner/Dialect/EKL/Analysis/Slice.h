/// Declares the Slice type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Dialect/EKL/Analysis/Index.h"
#include "messner/Dialect/EKL/Analysis/Offset.h"
#include "messner/Dialect/EKL/Analysis/Range.h"

#include <compare>
#include <llvm/ADT/Hashing.h>
#include <tuple>
#include <type_traits>

namespace mlir::ekl {

/// Represents a strided range of indices.
///
/// A Slice is represented by an inclusive start index, an exclusive end index,
/// and a non-zero stride. If the stride is negative, the sequence is iterated
/// over in reverse.
///
/// In contrast to a Range, a Slice can be relative to the extent of the
/// sequence due to the usage of Index bounds. Additionally, by convention, a
/// Slice is clamped during conversion to an in-bounds Range.
class Slice {
public:
    /// Obtains the identity slice.
    static constexpr auto id() -> Slice;
    /// Obtains the reverse identity slice.
    static constexpr auto rid() -> Slice;

    /// Initializes an empty Slice.
    /*implicit*/ constexpr Slice() : Slice(Index{}, Index{}) {}
    /// Initializes a Slice from @p begin , @p end and @p stride .
    ///
    /// The bounds are automatically clamped to the nearest potentially
    /// in-bounds slice.
    ///
    /// @pre    `stride != 0`
    explicit constexpr Slice(Index begin, Index end, Offset stride = 1);
    /// Initializes a Slice from optional @p begin and @p end with @p stride .
    ///
    /// If @p begin is std::nullopt, it will be set to Index::begin() or
    /// Index::rbegin() depending on the stride. Analogous for @p end .
    /// Otherwise the bounds are automatically clamped to the nearest
    /// potentially in-bounds slice.
    ///
    /// @pre    `stride != 0`
    explicit constexpr Slice(
        std::optional<Index> begin,
        std::optional<Index> end,
        Offset stride = 1);

    /// Gets the inclusive begin index.
    constexpr auto getBegin() const -> Index { return _begin; }
    /// Gets the exclusive end index.
    constexpr auto getEnd() const -> Index { return _end; }
    /// Gets the stride.
    ///
    /// @post   `result != 0`
    constexpr auto getStride() const -> Offset { return _stride; }

    template<std::size_t I>
    friend constexpr auto get(const Slice &slice)
    {
        if constexpr (I == 0)
            return slice.getBegin();
        else if constexpr (I == 1)
            return slice.getEnd();
        else
            return slice.getStride();
    }

    /// Determines whether this slice is trivially empty.
    ///
    /// A slice is trivially empty if its bounds are relative to the same point,
    /// but ordered opposite to the ordering of 0 and the stride.
    constexpr auto isTriviallyEmpty() const -> bool;

    /// Converts into a range inside some @p extent .
    ///
    /// The result is defined if @p extent is bounded.
    constexpr auto toRange(Extent extent) const -> std::optional<Range>;
    /// Obtains the minimum extent into which this is an in-bounds slice.
    ///
    /// @post   `result.isBounded()`
    constexpr auto getBound() const -> Extent;

    /// Obtains a slice of the first @p count elements.
    ///
    /// If @p count is not positive or greater than `size()`, the resulting
    /// Range is empty.
    constexpr auto take_front(Offset::value_type count) const -> Slice;
    /// Obtains a slice without the first @p count elements.
    ///
    /// If @p count is greater than `size()`, the resulting Range is empty. If
    /// it is negative, the current slice is returned.
    constexpr auto drop_front(Offset::value_type count) const -> Slice;
    /// Obtains a slice of the last @p count elements.
    ///
    /// If @p count is not positive or greater than `size()`, the resulting
    /// Range is empty.
    constexpr auto take_back(Offset::value_type count) const -> Slice;
    /// Obtains a slice without the last @p count elements.
    ///
    /// If @p count is greater than `size()`, the resulting Range is empty. If
    /// it is negative, the current slice is returned.
    constexpr auto drop_back(Offset::value_type count) const -> Slice;

    /// Determines whether two slices are definitionally equal.
    constexpr auto operator==(const Slice &) const -> bool = default;

    friend auto hash_value(const Slice &slice) -> llvm::hash_code;

private:
    Index _begin, _end;
    Offset _stride;
};

} // namespace mlir::ekl

namespace std {

template<>
struct tuple_size<::mlir::ekl::Slice> : integral_constant<size_t, 3> {};

template<size_t I>
struct tuple_element<I, ::mlir::ekl::Slice> {
    using type = conditional_t<I == 2, ::mlir::ekl::Offset, ::mlir::ekl::Index>;
};

template<>
struct hash<::mlir::ekl::Slice> {
    auto operator()(const ::mlir::ekl::Slice &slice) const -> size_t;
};

} // namespace std

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Slice implementation
//===----------------------------------------------------------------------===//

constexpr auto Slice::id() -> Slice
{
    return Slice(Index::begin(), Index::end());
}

constexpr auto Slice::rid() -> Slice
{
    return Slice(Index::rbegin(), Index::rend());
}

constexpr Slice::Slice(Index begin, Index end, Offset stride)
        : _begin(begin.clamp()),
          _end(end.isFromEnd() && end.getOffset() > 0 ? Index::end() : end),
          _stride(stride)
{
    assert(stride != 0);
}

constexpr Slice::Slice(
    std::optional<Index> begin,
    std::optional<Index> end,
    Offset stride)
        : Slice(
              begin.value_or(stride >= 0 ? Index::begin() : Index::rbegin()),
              end.value_or(stride >= 0 ? Index::end() : Index::rend()),
              stride)
{}

constexpr auto Slice::isTriviallyEmpty() const -> bool
{
    const auto cmp = getBegin() <=> getEnd();
    if (cmp == std::partial_ordering::unordered) return false;
    return (getStride() >= 0) ? !std::is_lt(cmp) : !std::is_gt(cmp);
}

constexpr auto Slice::toRange(Extent extent) const -> std::optional<Range>
{
    if (!extent.isBounded()) return std::nullopt;
    const auto begin = *getBegin().toAbsolute(extent);
    const auto end   = *getEnd().toAbsolute(extent);

    return Range(
        std::max(begin, Offset(0)),
        std::min(end, *Offset::end(extent)),
        getStride());
}

constexpr auto Slice::getBound() const -> Extent
{
    const auto maxBound = std::max(
        getBegin().getBound().value_or(Extent(0)),
        getEnd().getBound().value_or(Extent(0)));
    return Extent(maxBound);
}

constexpr auto Slice::take_front(Offset::value_type count) const -> Slice
{
    if (count <= 0) return {};

    const auto offset = *saturate(getStride() * (count - 1));
    return Slice(getBegin(), *saturate(getBegin() + offset), getStride());
}

constexpr auto Slice::drop_front(Offset::value_type count) const -> Slice
{
    if (count <= 0) return *this;

    const auto offset = *saturate(getStride() * (count - 1));
    return Slice(*saturate(getBegin() + offset), getEnd(), getStride());
}

constexpr auto Slice::take_back(Offset::value_type count) const -> Slice
{
    if (count <= 0) return {};

    const auto offset = *saturate(getStride() * (count - 1));
    return Slice(*saturate(getEnd() - offset), getEnd(), getStride());
}

constexpr auto Slice::drop_back(Offset::value_type count) const -> Slice
{
    if (count <= 0) return *this;

    const auto offset = *saturate(getStride() * (count - 1));
    return Slice(getBegin(), *saturate(getEnd() - offset), getStride());
}

inline auto hash_value(const Slice &slice) -> llvm::hash_code
{
    return llvm::hash_combine(
        slice.getBegin(),
        slice.getEnd(),
        slice.getStride());
}

} // namespace mlir::ekl

namespace std {

//===----------------------------------------------------------------------===//
// hash<::mlir::ekl::Slice> implementation
//===----------------------------------------------------------------------===//

inline auto
hash<::mlir::ekl::Slice>::operator()(const ::mlir::ekl::Slice &slice) const
    -> size_t
{
    using llvm::hash_value;
    return static_cast<size_t>(hash_value(slice));
}

} // namespace std
