/// Declares the Extent type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Support/fallible_arith.h"
#include "messner/Support/int.h"

#include <cassert>
#include <compare>
#include <cstdint>
#include <limits>
#include <llvm/ADT/Hashing.h>
#include <optional>
#include <type_traits>

namespace mlir::ekl {

// clang-format off
/// Tag that indicates an unbounded value.
constexpr struct unbounded_t {} unbounded;
// clang-format on

/// Represents the length of an indexed sequence.
///
/// This class implements a strong wrapper around an std::uint64_t that ensures
/// correct usage of static/dynamic extents in the compiler. An extent can be in
/// one of two states:
///
///     - Bounded
///
///       A bounded extent represents a statically known bound on the length of
///       an indexed sequence. This length can not exceed `value_max`, which is
///       the maximum of std::int64_t. This ensures that all offsets into the
///       sequence can be properly represented without overflow.
///
///     - Unbounded
///
///       An unbounded extent has no statically known bound.
class Extent : public messner::mixin::fallible_arith<Extent> {
public:
    /// Integer type that stores the extent value.
    using value_type                = std::uint64_t;
    /// Maximum in-bounds value held in the extent.
    static constexpr auto value_max = static_cast<value_type>(
        std::numeric_limits<std::make_signed_t<value_type>>::max());

    /// Type that holds the result of arithmetic operations.
    using arith_result = std::optional<Extent>;

    /// Obtains the largest Extent.
    static constexpr auto max() -> Extent { return Extent(value_max); }

    /// Initializes the 0 Extent.
    /*implicit*/ constexpr Extent() = default;
    /// Initializes the unbounded Extent.
    /*implicit*/ constexpr Extent(unbounded_t) : _value(-1UL) {}
    /// Initializes an Extent from @p value .
    ///
    /// @pre    `value <= max()`
    ///
    /// @post   `isBounded()`
    explicit constexpr Extent(value_type value);

    /// Gets the underlying integer value.
    constexpr auto getValue() const -> value_type { return _value; }

    /// Gets a value indicating whether this extent is statically bounded.
    constexpr auto isBounded() const -> bool { return _value <= value_max; }
    /// Obtains the extent value if it is bounded.
    ///
    /// @post   `!isBounded() || *result == getValue()`
    /// @post   `*result <= value_max`
    constexpr auto getBound() const -> std::optional<value_type>;

    /// Computes the sum of two extents.
    ///
    /// If either extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff it is less or equal to value_max.
    constexpr auto operator+(const Extent &rhs) const -> arith_result;
    /// Computes the sum of an extent and a value.
    ///
    /// If the extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff it is less or equal to value_max.
    constexpr auto operator+(value_type rhs) const -> arith_result;
    /// @copydoc operator+(value_type).
    friend constexpr auto operator+(value_type lhs, const Extent &rhs)
        -> arith_result;

    /// Computes the difference of two extents.
    ///
    /// If either extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff it is greater or equal to 0.
    constexpr auto operator-(const Extent &rhs) const -> arith_result;
    constexpr auto operator-(value_type rhs) const -> arith_result;

    /// Computes the product of an extent and a value.
    ///
    /// If the extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff it is less or equal to value_max.
    constexpr auto operator*(value_type rhs) const -> arith_result;
    /// @copydoc operator*(value_type).
    friend constexpr auto operator*(value_type lhs, const Extent &rhs)
        -> arith_result;

    /// Computes the quotient of an extent and a value, rounded towards zero.
    ///
    /// If the extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff @p rhs is not 0.
    constexpr auto operator/(value_type rhs) const -> arith_result;

    /// Computes the remainder of truncating division of an extent and a value.
    ///
    /// If the extent is unbounded, the result is unbounded. Otherwise the
    /// result is defined iff @p rhs is not 0.
    constexpr auto operator%(value_type rhs) const -> arith_result;

    /// Converts to @c true if the value is not zero.
    explicit constexpr operator bool() const { return _value; }

    /// Determines whether two extents are equal.
    constexpr auto operator==(const Extent &) const -> bool = default;
    /// Determines whether the extent holds the value @p rhs .
    constexpr auto operator==(value_type rhs) const -> bool;
    /// Determines whether this extent is unbounded.
    constexpr auto operator==(unbounded_t) const -> bool;

    friend auto hash_value(const Extent &extent) -> llvm::hash_code;

    /// Compares two extents, if they are bounded.
    constexpr auto operator<=>(const Extent &rhs) const
        -> std::partial_ordering;
    /// Compares the value of an extent, if it is bounded.
    constexpr auto operator<=>(value_type rhs) const -> std::partial_ordering;

private:
    value_type _value;
};

} // namespace mlir::ekl

namespace std {

template<>
struct hash<::mlir::ekl::Extent> {
    constexpr auto operator()(const ::mlir::ekl::Extent &extent) const
        -> size_t;
};

} // namespace std

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Extent implementation
//===----------------------------------------------------------------------===//

constexpr Extent::Extent(value_type value) : _value(value)
{
    assert(value <= value_max);
}

constexpr auto Extent::getBound() const -> std::optional<value_type>
{
    if (!isBounded()) return std::nullopt;
    return getValue();
}

constexpr auto Extent::operator+(const Extent &rhs) const -> arith_result
{
    if (const auto maybeRhs = rhs.getBound(); maybeRhs)
        return *this + *maybeRhs;
    return std::nullopt;
}

constexpr auto Extent::operator+(value_type rhs) const -> arith_result
{
    if (!isBounded()) return unbounded;
    if (const auto maybeExact = messner::checked_add(getValue(), rhs).exact();
        maybeExact)
        return Extent(*maybeExact);
    return std::nullopt;
}

constexpr auto operator+(Extent::value_type lhs, const Extent &rhs)
    -> Extent::arith_result
{
    return rhs + lhs;
}

constexpr auto Extent::operator-(const Extent &rhs) const -> arith_result
{
    if (const auto maybeRhs = rhs.getBound(); maybeRhs)
        return *this - *maybeRhs;
    return std::nullopt;
}

constexpr auto Extent::operator-(value_type rhs) const -> arith_result
{
    if (!isBounded()) return unbounded;
    if (const auto maybeExact = messner::checked_sub(getValue(), rhs).exact();
        maybeExact)
        return Extent(*maybeExact);
    return std::nullopt;
}

constexpr auto Extent::operator*(value_type rhs) const -> arith_result
{
    if (!isBounded()) return unbounded;
    if (const auto maybeExact = messner::checked_mul(getValue(), rhs).exact();
        maybeExact)
        return Extent(*maybeExact);
    return std::nullopt;
}

constexpr auto operator*(Extent::value_type lhs, const Extent &rhs)
    -> Extent::arith_result
{
    return rhs * lhs;
}

constexpr auto Extent::operator/(value_type rhs) const -> arith_result
{
    if (!isBounded()) return unbounded;
    if (const auto maybeExact =
            messner::checked_div_trunc(getValue(), rhs).exact();
        maybeExact)
        return Extent(*maybeExact);
    return std::nullopt;
}

constexpr auto Extent::operator%(value_type rhs) const -> arith_result
{
    if (!isBounded()) return unbounded;
    if (const auto maybeExact =
            messner::checked_rem_trunc(getValue(), rhs).exact();
        maybeExact)
        return Extent(*maybeExact);
    return std::nullopt;
}

constexpr auto Extent::operator==(value_type rhs) const -> bool
{
    return getValue() == rhs;
}

constexpr auto Extent::operator==(unbounded_t) const -> bool
{
    return !isBounded();
}

inline auto hash_value(const Extent &extent) -> llvm::hash_code
{
    return std::hash<Extent>{}(extent);
}

constexpr auto Extent::operator<=>(const Extent &rhs) const
    -> std::partial_ordering
{
    if (isBounded() ^ rhs.isBounded()) return std::partial_ordering::unordered;
    return getValue() <=> rhs.getValue();
}

constexpr auto Extent::operator<=>(value_type rhs) const
    -> std::partial_ordering
{
    if (!isBounded()) return std::partial_ordering::unordered;
    return getValue() <=> rhs;
}

} // namespace mlir::ekl

namespace std {

//===----------------------------------------------------------------------===//
// hash<::mlir::ekl::Extent> implementation
//===----------------------------------------------------------------------===//

constexpr auto
hash<::mlir::ekl::Extent>::operator()(const ::mlir::ekl::Extent &extent) const
    -> size_t
{
    return static_cast<size_t>(extent.getValue());
}

} // namespace std
