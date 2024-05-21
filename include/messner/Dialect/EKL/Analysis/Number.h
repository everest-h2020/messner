/// Declares the Number literal type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Hashing.h"

#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir::ekl {

template<class T, unsigned MaxWidth>
concept uint_le =
    std::unsigned_integral<T> && std::numeric_limits<T>::digits <= MaxWidth;
template<class T, unsigned MaxWidth>
concept sint_le =
    std::signed_integral<T> && std::numeric_limits<T>::digits <= MaxWidth;

/// Type that holds an EKL number literal of arbitrary precision.
///
/// EKL stores its generic unspecified number literals as a tuple of an
/// arbitrary precision integer mantissa and a 64 bit binary exponent. It can
/// thus represent the rational numbers accurately within a vast value range.
///
/// Additionally, this representaton makes it easy to convert the value to
/// fixed-point and floating-point numerals in base 2.
struct Number {
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

    /// The number of bits per mantissa storage word.
    static constexpr unsigned word_bits = std::numeric_limits<uword_t>::digits;

    /// Gets the sign bit of @p word .
    [[nodiscard]] static bool getSignBit(uword_t word)
    {
        constexpr auto mask = static_cast<uword_t>(1U) << (word_bits - 1U);
        return (word & mask) == mask;
    }

    /// Gets the mantissa value of @p word .
    ///
    /// @post   `!result.isNegative()`
    [[nodiscard]] static mantissa_t getMantissa(uword_t word)
    {
        return mantissa_t(word_bits + (getSignBit(word) ? 1U : 0U), word);
    }
    /// Gets the mantissa value of @p word .
    ///
    /// @post   `result.isNegative() == (word < 0)`
    [[nodiscard]] static mantissa_t getMantissa(sword_t word)
    {
        return mantissa_t(word_bits, std::bit_cast<uword_t>(word), true);
    }

    /// Decomposes @p value into its mantissa and exponent.
    ///
    /// @pre    @p Float has less than or equal to 64 bits of precision.
    /// @pre    `!std::isnan(value)`
    template<std::floating_point Float>
    [[nodiscard]] static std::pair<Number::sword_t, Number::exponent_t>
    decomposeFloat(Float value)
    {
        assert(!std::isnan(value));

        constexpr auto precision = std::numeric_limits<Float>::digits;
        static_assert(
            precision <= std::numeric_limits<Number::sword_t>::digits);

        const auto exponent = std::ilogb(value) - precision;
        value               = std::scalbn(value, -exponent);

        return {static_cast<Number::sword_t>(value), exponent};
    }

    /// Initializes a Number from an unsigned integral @p uint .
    ///
    /// @post   `*this == uint`
    /*implicit*/ Number(uint_le<word_bits> auto uint)
            : Number(getMantissa(static_cast<uword_t>(uint)))
    {}
    /// Initializes a Number from a signed integral @p sint .
    ///
    /// @post   `*this == sint`
    /*implicit*/ Number(sint_le<word_bits> auto sint)
            : Number(getMantissa(static_cast<sword_t>(sint)))
    {}
    /// Initializes a Number from a floating point @p value .
    ///
    /// @pre    `!std::isnan(value)`
    /// @post   `*this == value`
    /*implicit*/ Number(std::floating_point auto value) : Number()
    {
        const auto [m, e] = decomposeFloat(value);
        m_mantissa        = getMantissa(m);
        m_exponent        = e;
    }
    /// Initializes a Number from an llvm::APFloat.
    ///
    /// @pre    `value.isIEEE() && value.isFinite()`
    /*implicit*/ Number(llvm::APFloat value);
    /// @copydoc Number(mantissa_t, exponent_t)
    /*implicit*/ Number(const llvm::APSInt &mantissa, exponent_t exponent = 0)
            : Number(static_cast<const llvm::APInt &>(mantissa), exponent)
    {
        // Ensure the sign is preserved.
        if (mantissa.isUnsigned() && getMantissa().isNegative())
            m_mantissa = m_mantissa.zext(m_mantissa.getBitWidth() + 1U);
    }
    /// Initializes a Number from @p mantissa and @p exponent .
    ///
    /// @post   `getMantissa() == mantissa`
    /// @post   `getExponent() == exponent`
    /*implicit*/ Number(const mantissa_t &mantissa, exponent_t exponent = 0)
            : m_mantissa(mantissa),
              m_exponent(exponent)
    {
        reduce();
    }
    /// Initializes a Number of value 0.
    ///
    /// @post   `*this == 0`
    /*implicit*/ Number() : m_mantissa(), m_exponent() {}

    /// Reduces the stored number in-place without changing its value.
    ///
    /// Tries to reduce the number of active bits in the mantissa by changing
    /// the exponent value. In particular, trailing zeros are removed, and the
    /// exponent is increased.
    ///
    /// The reduction operation is designed such that two numbers of equal value
    /// will reduce to the same representation iff they are equal.
    ///
    /// This operation does not allocate or free any memory.
    void reduce() const
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
            m_exponent = 0;
            return;
        }

        if (getExponent() > (max_exponent - shift)) [[unlikely]] {
            // Although unlikely, it may be that we're at the upper limit of the
            // dynamic range of the exponent field. In that case, we can't
            // store the result of E + s.
            return;
        }

        // Set m <- m' and E <- E + s.
        m_mantissa.ashrInPlace(shift);
        m_exponent += shift;
    }

    /// Reduces the memory allocated for storing this number if possible.
    void shrinkToFit();

    /// Rounds to the next integer in the direction of zero.
    void roundTowardsZero()
    {
        if (getExponent() >= 0) return;

        m_mantissa.ashrInPlace(-getExponent());
        m_exponent += 0;
    }

    /// Gets the integer mantissa.
    const mantissa_t &getMantissa() const { return m_mantissa; }
    /// Gets the binary exponent.
    exponent_t getExponent() const { return m_exponent; }

    /// Gets the signum of the value.
    [[nodiscard]] int getSign() const
    {
        if (getMantissa().isNegative()) return -1;
        return getMantissa().isZero() ? 0 : 1;
    }

    /// Converts the contained value to an llvm::APFloat with @p semantics .
    ///
    /// Applies to-odd rounding and saturates to next finite value when needed.
    ///
    /// @param  [in]        semantics llvm::fltSemantics
    ///
    /// @return APFloat
    APFloat toAPFloatWithRounding(llvm::fltSemantics &semantics) const;

    /// Tries to get the contained value as an uword_t, if it fits.
    [[nodiscard]] std::optional<uword_t> tryGetUInt() const;
    /// Tries to get the contained value as an sword_t, if it fits.
    [[nodiscard]] std::optional<sword_t> tryGetSInt() const;
    /// Tries to get the contained value as an f64, if it fits.
    [[nodiscard]] std::optional<double> tryGetF64() const;

    /// Parses a Number literal using an AsmParser.
    ///
    /// We do NOT define an mlir::FieldParser specialization because the
    /// AsmParser does not allow us to implement the literal syntax we want for
    /// the EKL language. Users need to use printField() to generate the MLIR
    /// literal.
    ///
    /// This parser implements the following grammar:
    ///
    /// ```
    /// number          ::= int | `"` binary-rational `"`
    /// binary-rational ::= f64-literal | int ( `p` | `P` ) int
    /// ```
    ///
    /// @param  [in]        parser  AsmParser.
    ///
    /// @retval Number      Parsed Number literal.
    /// @retval failure()   Failed, error emitted to @p parser .
    static FailureOr<Number> parseField(AsmParser &parser);

    /// Prints a Number literal for parsing by parseField().
    ///
    /// This printer implements the following grammar:
    ///
    /// ```
    /// number          ::= int | `"` binary-rational `"`
    /// binary-rational ::= f64-literal | int ( `p` | `P` ) int
    /// ```
    ///
    /// @param  [in]        os  llvm::raw_ostream.
    void printField(llvm::raw_ostream &os) const;

    /// Determines whether two numbers have the same value.
    [[nodiscard]] friend bool operator==(const Number &lhs, const Number &rhs)
    {
        const auto sketch = lhs.compareSketch(rhs);
        if (std::is_eq(sketch)) return true;
        if (std::is_neq(sketch)) return false;

        // Fallback to a slower comparison implementation. Since both mantissas
        // have the same sign, we can perform a bit-wise equality comparison.
        return lhs.getMantissa().eq(rhs.getMantissa());
    }

    /// Obtains a hash_code for @p value .
    ///
    /// @post   x = y -> hash_value(x) = hash_value(y)
    [[nodiscard]] friend llvm::hash_code hash_value(const Number &value)
    {
        return llvm::hash_combine(value.getExponent(), value.getMantissa());
    }

    /// Determines the ordering relation between two numbers.
    [[nodiscard]] friend std::strong_ordering
    operator<=>(const Number &lhs, const Number &rhs)
    {
        const auto sketch = lhs.compareSketch(rhs);
        if (std::is_eq(sketch)) return std::strong_ordering::equal;
        if (std::is_lt(sketch)) return std::strong_ordering::less;
        if (std::is_gt(sketch)) return std::strong_ordering::greater;

        // Fallback to a slower comparison implementation. Since both mantissas
        // have the same sign, we can perform a word-wise lexicographical
        // comparison.
        return compareImpl(rhs.getMantissa(), rhs.getMantissa());
    }

private:
    /// Compares this value with @p rhs without doing significant work.
    ///
    /// Returns std::partial_ordering::unordered if the ordering is still left
    /// undecided.
    ///
    /// @param  [in]            rhs Right hand operand.
    ///
    /// @return std::partial_ordering
    [[nodiscard]] std::partial_ordering compareSketch(const Number &rhs) const
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

    /// Compares @p lhs and @p rhs as signed integers, knowing they have the
    /// same sign.
    ///
    /// @pre    `lhs.isNegative() == rhs.isNegative()`
    [[nodiscard]] static std::strong_ordering
    compareImpl(const mantissa_t &lhs, const mantissa_t &rhs);

    mutable mantissa_t m_mantissa;
    mutable exponent_t m_exponent;
};

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

/// Parses a Number.
///
/// This parser uses the Number::parseField(AsmParser &) implementation.
///
/// @param  [in]        parser  AsmParser.
/// @param  [out]       number  Number.
///
/// @return ParseResult.
ParseResult parseNumber(AsmParser &parser, Number &number);

/// Prints a Number.
///
/// This printer uses the Number::printField(AsmPrinter &) implementation.
///
/// @param  [in]        printer AsmPrinter.
/// @param  [in]        number  Number.
void printNumber(AsmPrinter &printer, const Number &number);

} // namespace mlir::ekl
