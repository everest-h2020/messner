/// Declares the Rational literal type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/Hashing.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

namespace mlir {

class AsmParser;
class AsmPrinter;

} // namespace mlir

namespace mlir::ekl {

/// Indicates whether the C++ target architecture is little-endian.
constexpr auto is_little_endian = std::endian::native == std::endian::little;

/// Concept for an unsigned little-endian integer with up to @p MaxWidth bits.
template<class T, unsigned MaxWidth>
concept uint_le =
    std::unsigned_integral<T> && std::numeric_limits<T>::digits <= MaxWidth
    && is_little_endian;

/// Concept for a signed little-endian integer with up to @p MaxWidth bits.
template<class T, unsigned MaxWidth>
concept sint_le =
    std::signed_integral<T> && std::numeric_limits<T>::digits <= MaxWidth
    && is_little_endian;

/// Type that holds an EKL rational number literal of arbitrary precision.
///
/// EKL stores its generic unspecified number literals as a tuple of an
/// arbitrary precision integer mantissa and a 64 bit binary exponent. It can
/// thus represent the rational numbers accurately within a vast value range.
///
/// Additionally, this representaton makes it easy to convert the value to
/// fixed-point and floating-point numerals in base 2.
struct Rational {
    /// The mantissa type.
    using mantissa_t = llvm::APInt;
    /// The exponent type.
    using exponent_t = std::int64_t;

    /// The minimum exponent value.
    static constexpr exponent_t min_exponent =
        std::numeric_limits<exponent_t>::min();
    /// The maximum exponent value.
    static constexpr exponent_t max_exponent =
        std::numeric_limits<exponent_t>::max();

    /// The unsigned mantissa storage word type.
    using uword_t = mantissa_t::WordType;
    /// The signed mantissa storage word type.
    using sword_t = std::make_signed_t<uword_t>;
    /// Holds a signed mantissa and an exponent.
    using fword_t = std::pair<sword_t, exponent_t>;

    /// The number of bits per mantissa storage word.
    static constexpr unsigned word_bits = std::numeric_limits<uword_t>::digits;

    /// Gets the sign bit of @p word .
    static auto getSignBit(uword_t word) -> bool;

    /// Gets the mantissa value of @p word .
    ///
    /// @post   `!result.isNegative()`
    static auto getMantissa(uword_t word) -> mantissa_t;
    /// Gets the mantissa value of @p word .
    ///
    /// @post   `result.isNegative() == (word < 0)`
    static auto getMantissa(sword_t word) -> mantissa_t;

    /// Decomposes @p value into its mantissa and exponent.
    ///
    /// @pre    @p Float has less than or equal to 64 bits of precision.
    /// @pre    `!std::isnan(value)`
    template<std::floating_point Float>
    static auto decomposeFloat(Float value) -> fword_t;

    /// Initializes a Rational of value 0.
    ///
    /// @post   `*this == 0`
    /*implicit*/ Rational() = default;
    /// Initializes a Rational from @p mantissa and @p exponent .
    ///
    /// @post   `getMantissa() == mantissa`
    /// @post   `getExponent() == exponent`
    /*implicit*/ Rational(const mantissa_t &mantissa, exponent_t exponent = 0);
    /// @copydoc Rational(mantissa_t, exponent_t)
    /*implicit*/ Rational(
        const llvm::APSInt &mantissa,
        exponent_t exponent = 0);
    /// Initializes a Rational from an unsigned integral @p uint .
    ///
    /// @post   `*this == uint`
    /*implicit*/ Rational(uint_le<word_bits> auto uint);
    /// Initializes a Rational from a signed integral @p sint .
    ///
    /// @post   `*this == sint`
    /*implicit*/ Rational(sint_le<word_bits> auto sint);
    /// Initializes a Rational from a floating point @p value .
    ///
    /// @pre    `!std::isnan(value)`
    /// @post   `*this == value`
    /*implicit*/ Rational(std::floating_point auto value);
    /// Initializes a Rational from an llvm::APFloat.
    ///
    /// @pre    `value.isIEEE() && value.isFinite()`
    /*implicit*/ Rational(llvm::APFloat value);

    /// Reduces the stored rational in-place without changing its value.
    ///
    /// Tries to reduce the number of active bits in the mantissa by changing
    /// the exponent value. In particular, trailing zeros are removed, and the
    /// exponent is increased.
    ///
    /// The reduction operation is designed such that two numbers of equal value
    /// will reduce to the same representation iff they are equal.
    ///
    /// This operation does not allocate or free any memory.
    void reduce() const;

    /// Reduces the memory allocated for storing this rational if possible.
    void shrinkToFit();

    /// Rounds to the next integer in the direction of zero.
    void roundTowardsZero();

    /// Gets the integer mantissa.
    auto getMantissa() const -> const mantissa_t & { return _mantissa; }
    /// Gets the binary exponent.
    auto getExponent() const -> exponent_t { return _exponent; }

    /// Gets the signum of the value.
    auto getSign() const -> int;

    /// Converts the contained value to an llvm::APFloat with @p semantics .
    ///
    /// Applies to-odd rounding and saturates to next finite value when needed.
    auto toAPFloatWithRounding(llvm::fltSemantics &semantics) const
        -> llvm::APFloat;

    /// Tries to get the contained value as an uword_t, if it fits.
    auto tryGetUInt() const -> std::optional<uword_t>;
    /// Tries to get the contained value as an sword_t, if it fits.
    auto tryGetSInt() const -> std::optional<sword_t>;
    /// Tries to get the contained value as an f64, if it fits.
    auto tryGetF64() const -> std::optional<double>;

    /// Parses a Rational literal using an AsmParser.
    ///
    /// We do NOT define an mlir::FieldParser specialization because the
    /// AsmParser does not allow us to implement the literal syntax we want for
    /// the EKL language. Users need to use printField() to generate the MLIR
    /// literal.
    ///
    /// This parser implements the following grammar:
    ///
    /// ```
    /// rational        ::= int | `"` binary-rational `"`
    /// binary-rational ::= f64-literal | int ( `p` | `P` ) int
    /// ```
    ///
    /// @retval Rational    Parsed Rational literal.
    /// @retval failure()   Failed, error emitted to @p parser .
    static auto parseField(AsmParser &parser) -> mlir::FailureOr<Rational>;

    /// Prints a Rational literal for parsing by parseField().
    ///
    /// This printer implements the following grammar:
    ///
    /// ```
    /// rational        ::= int | `"` binary-rational `"`
    /// binary-rational ::= f64-literal | int ( `p` | `P` ) int
    /// ```
    void printField(llvm::raw_ostream &os) const;

    /// Determines whether two numbers have the same value.
    auto operator==(const Rational &rhs) const -> bool;

    /// Obtains a hash_code for @p value .
    ///
    /// @post   x = y -> hash_value(x) = hash_value(y)
    friend auto hash_value(const Rational &value) -> llvm::hash_code;

    /// Determines the ordering relation between two numbers.
    auto operator<=>(const Rational &rhs) const -> std::strong_ordering;

private:
    /// Compares this value with @p rhs without doing significant work.
    ///
    /// Returns std::partial_ordering::unordered if the ordering is still left
    /// undecided.
    auto compareSketch(const Rational &rhs) const -> std::partial_ordering;

    /// Compares @p lhs and @p rhs as signed integers, knowing they have the
    /// same sign.
    ///
    /// @pre    `lhs.isNegative() == rhs.isNegative()`
    static auto compareImpl(const mantissa_t &lhs, const mantissa_t &rhs)
        -> std::strong_ordering;

    mutable mantissa_t _mantissa;
    mutable exponent_t _exponent;
};

} // namespace mlir::ekl

namespace std {

template<>
struct hash<::mlir::ekl::Rational> {
    auto operator()(const ::mlir::ekl::Rational &rational) const -> size_t;
};

} // namespace std

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Rational implementation
//===----------------------------------------------------------------------===//

inline auto Rational::getSignBit(uword_t word) -> bool
{
    constexpr auto mask = static_cast<uword_t>(1U) << (word_bits - 1U);
    return (word & mask) == mask;
}

inline auto Rational::getMantissa(uword_t word) -> mantissa_t
{
    return mantissa_t(word_bits + (getSignBit(word) ? 1U : 0U), word);
}

inline auto Rational::getMantissa(sword_t word) -> mantissa_t
{
    return mantissa_t(word_bits, std::bit_cast<uword_t>(word), true);
}

template<std::floating_point Float>
inline auto Rational::decomposeFloat(Float value) -> fword_t
{
    assert(!std::isnan(value));

    constexpr auto precision = std::numeric_limits<Float>::digits;
    static_assert(precision <= std::numeric_limits<sword_t>::digits);

    const auto exponent = std::ilogb(value) - precision;
    value               = std::scalbn(value, -exponent);

    return {static_cast<sword_t>(value), exponent};
}

inline Rational::Rational(const mantissa_t &mantissa, exponent_t exponent)
        : _mantissa(mantissa),
          _exponent(exponent)
{
    reduce();
}

inline Rational::Rational(const llvm::APSInt &mantissa, exponent_t exponent)
        : Rational(static_cast<const llvm::APInt &>(mantissa), exponent)
{
    // Ensure the sign is preserved.
    if (mantissa.isUnsigned() && getMantissa().isNegative())
        _mantissa = _mantissa.zext(_mantissa.getBitWidth() + 1U);
}

inline Rational::Rational(uint_le<word_bits> auto uint)
        : Rational(getMantissa(static_cast<uword_t>(uint)))
{}

inline Rational::Rational(sint_le<word_bits> auto sint)
        : Rational(getMantissa(static_cast<sword_t>(sint)))
{}

inline Rational::Rational(std::floating_point auto value) : Rational()
{
    const auto [m, e] = decomposeFloat(value);
    _mantissa         = getMantissa(m);
    _exponent         = e;
}

inline void Rational::reduce() const
{
    // Decompose m * 2^E into m' * 2^(E+s) such that m' = m * 2^-s and m'
    // is integer. Clearly, this is true when the least significant s bits
    // of m are zero.

    const auto shift = getMantissa().countr_zero();
    if (shift == 0) {
        // No such m' exists.
        return;
    }
    if (shift == getMantissa().getBitWidth()) {
        // m = 0, so normalize the exponent to 0.
        _exponent = 0;
        return;
    }
    if (getExponent() > (max_exponent - shift)) [[unlikely]] {
        // Although unlikely, it may be that we're at the upper limit of the
        // dynamic range of the exponent field. In that case, we can't
        // store the result of E + s.
        _mantissa.ashrInPlace(max_exponent - getExponent());
        _exponent = max_exponent;
        return;
    }

    // Set m <- m' and E <- E + s.
    _mantissa.ashrInPlace(shift);
    _exponent += shift;
}

inline void Rational::roundTowardsZero()
{
    if (getExponent() >= 0) return;

    _mantissa.ashrInPlace(-getExponent());
    _exponent += 0;
}

inline auto Rational::getSign() const -> int
{
    if (getMantissa().isNegative()) return -1;
    return getMantissa().isZero() ? 0 : 1;
}

inline auto Rational::operator==(const Rational &rhs) const -> bool
{
    const auto sketch = compareSketch(rhs);
    if (std::is_eq(sketch)) return true;
    if (std::is_neq(sketch)) return false;

    // Fallback to a slower comparison implementation. Since both mantissas
    // have the same sign, we can perform a bit-wise equality comparison.
    return getMantissa().eq(rhs.getMantissa());
}

inline auto hash_value(const Rational &value) -> llvm::hash_code
{
    return llvm::hash_combine(value.getExponent(), value.getMantissa());
}

inline auto Rational::operator<=>(const Rational &rhs) const
    -> std::strong_ordering
{
    const auto sketch = compareSketch(rhs);
    if (std::is_eq(sketch)) return std::strong_ordering::equal;
    if (std::is_lt(sketch)) return std::strong_ordering::less;
    if (std::is_gt(sketch)) return std::strong_ordering::greater;

    // Fallback to a slower comparison implementation. Since both mantissas
    // have the same sign, we can perform a word-wise lexicographical
    // comparison.
    return compareImpl(rhs.getMantissa(), rhs.getMantissa());
}

inline auto Rational::compareSketch(const Rational &rhs) const
    -> std::partial_ordering
{
    // Compare signs, because that is fastest.
    const auto sgn    = getSign();
    const auto cmpSgn = sgn <=> rhs.getSign();
    if (!std::is_eq(cmpSgn) || sgn == 0) return cmpSgn;

    // Reduce both sides, since that is fast and ensures trivial equality.
    reduce();
    rhs.reduce();

    // Compare exponents, which must now match if the operands are equal.
    const auto cmpExp = getExponent() <=> rhs.getExponent();
    if (!std::is_eq(cmpExp)) return cmpExp;

    // We don't know yet.
    return std::partial_ordering::unordered;
}

} // namespace mlir::ekl

namespace std {

//===----------------------------------------------------------------------===//
// hash<::mlir::ekl::Rational> implementation
//===----------------------------------------------------------------------===//

inline auto hash<::mlir::ekl::Rational>::operator()(
    const ::mlir::ekl::Rational &slice) const -> size_t
{
    using llvm::hash_value;
    return static_cast<size_t>(hash_value(slice));
}

} // namespace std
