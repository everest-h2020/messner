/// Declares the Offset type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Extent.h"
#include "messner/Support/arith_result.h"

#include <compare>
#include <limits>
#include <llvm/ADT/Hashing.h>
#include <optional>
#include <type_traits>

namespace mlir::ekl {

/// Represents a relative position in an indexed sequence.
///
/// This class implements a strong wrapper around an std::int64_t that ensures
/// correct address calculations in the compiler.
class Offset {
public:
    /// Integer type that stores the offset value.
    using value_type                = std::make_signed_t<Extent::value_type>;
    /// Maximum in-bounds value held in the offset.
    static constexpr auto value_max = std::numeric_limits<value_type>::max();
    /// Minimum in-bounds value held in the offset.
    static constexpr auto value_min = std::numeric_limits<value_type>::min();

    /// Type that holds the result of arithmetic operations.
    using arith_result = messner::arith_result<Offset>;

    /// Obtains the smallest offset.
    static constexpr auto min() -> Offset { return Offset(value_min); }
    /// Obtains the largest offset.
    static constexpr auto max() -> Offset { return Offset(value_max); }

    /// Gets the end offset of @p extent , if it is bounded.
    static constexpr auto end(Extent extent) -> std::optional<Offset>;

    /// Initializes an Offset from @p value.
    /*implicit*/ constexpr Offset(value_type value = 0) : _value(value) {}

    /// Gets the underlying integer value.
    constexpr auto getValue() const -> value_type { return _value; }

    /// Obtains the additive inverse of the offset.
    ///
    /// The result is defined iff it is lessor equal to value_max.
    constexpr auto operator-() const -> arith_result;

    /// Computes the sum of two offsets.
    ///
    /// The result is defined iff it is within value_min and value_max.
    constexpr auto operator+(const Offset &rhs) const -> arith_result;
    /// Computes the sum of a value and an offset.
    ///
    /// The result is defined iff it is within value_min and value_max.
    friend constexpr auto operator+(value_type lhs, const Offset &rhs)
        -> arith_result;

    /// Computes the difference of two offsets.
    ///
    /// The result is defined iff it is within value_min and value_max.
    constexpr auto operator-(const Offset &rhs) const -> arith_result;

    /// Computes the product of an offset and a value.
    constexpr auto operator*(value_type rhs) const -> arith_result;
    /// Computes the product of a value and an offset.
    ///
    /// The result is defined iff it is within value_min and value_max.
    friend constexpr auto operator*(value_type lhs, const Offset &rhs)
        -> arith_result;

    /// Computes the quotient of an offset and a value, rounded towards zero.
    ///
    /// The result is defined iff @p rhs is not 0 and the result is less or
    /// equal to value_max.
    constexpr auto operator/(value_type rhs) const -> arith_result;

    /// Computes the remainder of truncating division of an offset and a value.
    ///
    /// The result is defined iff @p rhs is not 0.
    constexpr auto operator%(value_type rhs) const -> arith_result;

    /// @copydoc messner::saturate()
    friend constexpr auto saturate(arith_result result)
        -> std::optional<Offset>;

    /// Determines whether two offsets are equal.
    constexpr auto operator==(const Offset &) const -> bool = default;

    friend auto hash_value(const Offset &offset) -> llvm::hash_code;

    /// Compares two offsets.
    constexpr auto operator<=>(const Offset &) const
        -> std::strong_ordering = default;
    /// Compares an offset with size value.
    constexpr auto operator<=>(Extent::value_type size) const
        -> std::strong_ordering;
    /// Compares an offset with an extent.
    constexpr auto operator<=>(const Extent &rhs) const
        -> std::partial_ordering;

private:
    value_type _value;
};

} // namespace mlir::ekl

namespace std {

template<>
struct hash<::mlir::ekl::Offset> {
    constexpr auto operator()(const ::mlir::ekl::Offset &offset) const
        -> size_t;
};

} // namespace std

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Offset implementation
//===----------------------------------------------------------------------===//

constexpr auto Offset::end(Extent extent) -> std::optional<Offset>
{
    if (!extent.isBounded()) return std::nullopt;
    return Offset(static_cast<value_type>(extent.getValue()));
}

constexpr auto Offset::operator-() const -> arith_result
{
    const auto quality = getValue() == value_min
                           ? messner::arith_quality::overflow
                           : messner::arith_quality::exact;
    return arith_result(quality, -getValue());
}

constexpr auto Offset::operator+(const Offset &rhs) const -> arith_result
{
    const auto checked = messner::checked_add(getValue(), rhs.getValue());
    return arith_result(checked.quality(), Offset(checked.value()));
}

constexpr auto operator+(Offset::value_type lhs, const Offset &rhs)
    -> Offset::arith_result
{
    return rhs + lhs;
}

constexpr auto Offset::operator-(const Offset &rhs) const -> arith_result
{
    const auto checked = messner::checked_sub(getValue(), rhs.getValue());
    return arith_result(checked.quality(), Offset(checked.value()));
}

constexpr auto Offset::operator*(value_type rhs) const -> arith_result
{
    const auto checked = messner::checked_mul(getValue(), rhs);
    return arith_result(checked.quality(), Offset(checked.value()));
}

constexpr auto operator*(Offset::value_type lhs, const Offset &rhs)
    -> Offset::arith_result
{
    return rhs * lhs;
}

constexpr auto Offset::operator/(value_type rhs) const -> arith_result
{
    const auto checked = messner::checked_div_trunc(getValue(), rhs);
    if (!checked) return std::nullopt;
    return arith_result(checked.quality(), Offset(checked.value()));
}

constexpr auto Offset::operator%(value_type rhs) const -> arith_result
{
    const auto checked = messner::checked_rem_trunc(getValue(), rhs);
    if (!checked) return std::nullopt;
    return arith_result(checked.quality(), Offset(checked.value()));
}

constexpr auto saturate(Offset::arith_result result) -> std::optional<Offset>
{
    switch (result.quality()) {
    case messner::arith_quality::undefined: return std::nullopt;
    case messner::arith_quality::rounded:
    case messner::arith_quality::exact:     return result.value();
    case messner::arith_quality::underflow: return Offset::min();
    case messner::arith_quality::overflow:  return Offset::max();
    }
}

inline auto hash_value(const Offset &offset) -> llvm::hash_code
{
    return std::hash<Offset>{}(offset);
}

constexpr auto Offset::operator<=>(Extent::value_type rhs) const
    -> std::strong_ordering
{
    if (getValue() < 0) return std::strong_ordering::less;
    return static_cast<Extent::value_type>(getValue()) <=> rhs;
}

constexpr auto Offset::operator<=>(const Extent &rhs) const
    -> std::partial_ordering
{
    if (!rhs.isBounded()) return std::partial_ordering::unordered;
    return *this <=> rhs.getValue();
}

} // namespace mlir::ekl

namespace std {

//===----------------------------------------------------------------------===//
// hash<::mlir::ekl::Offset> implementation
//===----------------------------------------------------------------------===//

constexpr auto
hash<::mlir::ekl::Offset>::operator()(const ::mlir::ekl::Offset &offset) const
    -> size_t
{
    return static_cast<size_t>(offset.getValue());
}

} // namespace std
