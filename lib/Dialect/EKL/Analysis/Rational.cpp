/// Implements the Rational literal type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Rational.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/OpImplementation.h>
#include <system_error>

using namespace llvm;
using namespace mlir;
using namespace mlir::ekl;

/// Obtains the range of words in @p apint  up to the highest non-zero word.
///
/// @post   `!result.empty()`
static auto getActiveWords(const APInt &apint) -> ArrayRef<APInt::WordType>
{
    const auto numWords = std::max(1U, apint.getActiveWords());
    return ArrayRef<APInt::WordType>(apint.getRawData(), numWords);
}

/// Compares @p lhs and @p rhs as if they were unsigned integers.
static auto cmp_uint(const APInt &lhs, const APInt &rhs) -> std::strong_ordering
{
    // Get the words up to the most significant non-zero word.
    const auto lhsWords = getActiveWords(lhs);
    const auto rhsWords = getActiveWords(rhs);

    // Compare sizes, because that is fastest.
    const auto cmpSz = lhsWords.size() <=> rhsWords.size();
    if (!std::is_eq(cmpSz)) return cmpSz;

    // Perform lexicographial comparison of the contained data.
    for (auto i : llvm::iota_range<std::size_t>(0, lhsWords.size(), false)) {
        const auto j       = lhsWords.size() - i;
        const auto cmpWord = lhsWords[j] <=> rhsWords[j];
        if (!std::is_eq(cmpWord)) return cmpWord;
    }
    return std::strong_ordering::equal;
}

//===----------------------------------------------------------------------===//
// Rational implementation
//===----------------------------------------------------------------------===//

Rational::Rational(llvm::APFloat value) : Rational()
{
    assert(value.isIEEE() && value.isFinite());

    // Decompose value into exponent and a normalized fraction.
    int exp;
    // NOTE: Is always exact, no rounding mode used in base2.
    value =
        llvm::frexp(value, exp, llvm::APFloat::roundingMode::NearestTiesToEven);
    _exponent = exp;

    // Convert the normalized fraction into an integer.
    const auto prec = APFloat::semanticsPrecision(value.getSemantics());
    // NOTE: Is always exact..
    value           = llvm::scalbn(
        value,
        -prec,
        llvm::APFloat::roundingMode::NearestTiesToEven);
    _exponent -= prec;

    // Extract the integer mantissa.
    bool isExact;
    llvm::APSInt mantissa;
    value.convertToInteger(
        mantissa,
        llvm::APFloat::roundingMode::NearestTiesToEven,
        &isExact);
    assert(isExact);

    // Store the mantissa, and ensure that we didn't break its sign.
    _mantissa = mantissa;
    if (mantissa.isUnsigned() && _mantissa.isNegative())
        _mantissa = _mantissa.zext(_mantissa.getBitWidth() + 1U);

    reduce();
}

void Rational::shrinkToFit()
{
    reduce();

    auto padding = getMantissa().countr_zero();
    if (padding <= 1) {
        // We can't remove the last bit of padding or we would change the sign.
        return;
    }
    if (padding == getMantissa().getBitWidth()) {
        // The value is zero, normalize to the default-constructed instance.
        _mantissa = llvm::APInt{};
        _exponent = 0;
        return;
    }

    const auto shrink = --padding / word_bits;
    if (shrink == 0) {
        // Removing less than word_bits of padding will not free up any memory.
        return;
    }

    // Reallocate the mantissa.
    const auto newWords = ArrayRef<APInt::WordType>(
        getMantissa().getRawData(),
        getMantissa().getNumWords() - shrink);
    _mantissa = llvm::APInt(getMantissa().getBitWidth() - padding, newWords);
}

auto Rational::toAPFloatWithRounding(llvm::fltSemantics &semantics) const
    -> llvm::APFloat
{
    auto mantissa = getMantissa();
    auto exponent = getExponent();

    // Shift the mantissa so that it fits in the available precision.
    const auto prec = llvm::APFloat::semanticsPrecision(semantics);
    if (mantissa.getActiveBits() > prec) {
        const auto delta = mantissa.getActiveBits() - prec;
        mantissa.ashrInPlace(delta);
        exponent += delta;
    }

    // Saturate if necessary.
    const auto expMin = llvm::APFloat::semanticsMinExponent(semantics);
    const auto expMax = llvm::APFloat::semanticsMaxExponent(semantics);
    if (exponent < expMin)
        return llvm::APFloat::getSmallest(semantics, mantissa.isNegative());
    if (exponent > expMax)
        return llvm::APFloat::getLargest(semantics, mantissa.isNegative());

    // Convert the integer mantissa and then apply the exponent.
    llvm::APFloat result(semantics);
    result.convertFromAPInt(
        mantissa,
        true,
        llvm::APFloat::roundingMode::NearestTiesToEven);
    return llvm::scalbn(
        result,
        exponent,
        llvm::APFloat::roundingMode::NearestTiesToEven);
}

auto Rational::tryGetUInt() const -> std::optional<Rational::uword_t>
{
    if (getExponent() < 0) return std::nullopt;
    if (getMantissa().isNegative()) return std::nullopt;

    const auto bits = getMantissa().getActiveBits() + getExponent();
    if (bits > word_bits) return std::nullopt;

    return getMantissa().getZExtValue() << getExponent();
}

auto Rational::tryGetSInt() const -> std::optional<Rational::sword_t>
{
    if (getExponent() < 0) return std::nullopt;
    if (!getMantissa().isNegative()) {
        const auto bits = getMantissa().getActiveBits() + getExponent();
        if (bits >= word_bits) return std::nullopt;

        return getMantissa().getSExtValue() << getExponent();
    }

    const auto padding = getMantissa().countl_one() - 1U;
    const auto bits    = getMantissa().getBitWidth() - padding + getExponent();
    if (bits > word_bits) return std::nullopt;

    return getMantissa().getSExtValue() << getExponent();
}

auto Rational::tryGetF64() const -> std::optional<double>
{
    reduce();

    // NOTE: Technically, if the exponent doesn't fit, we could try to shift
    //       the mantissa to reduce the exponent. However, we would gain very
    //       little from that.

    if (getExponent() >= std::numeric_limits<double>::max_exponent)
        return std::nullopt;
    if (getExponent() < std::numeric_limits<double>::min_exponent)
        return std::nullopt;

    if (!getMantissa().isNegative()) {
        if (getMantissa().getActiveBits()
            >= std::numeric_limits<double>::digits)
            return std::nullopt;

        const auto intMag = getMantissa().getZExtValue();
        return std::scalbn(static_cast<double>(intMag), getExponent());
    }

    const auto padding = getMantissa().countr_one() - 1U;
    if (getMantissa().getBitWidth() - padding
        >= std::numeric_limits<double>::digits)
        return std::nullopt;

    const auto intMag = getMantissa().getSExtValue();
    return std::scalbln(static_cast<double>(intMag), getExponent());
}

auto Rational::parseField(AsmParser &parser) -> FailureOr<Rational>
{
    // Try parsing a bare integer literal.
    APInt mantissa;
    const auto maybeInt = parser.parseOptionalInteger(mantissa);
    if (maybeInt.has_value()) {
        if (*maybeInt) return failure();
        return Rational(mantissa);
    }

    // The value must be wrapped in a string literal, because we can't parse
    // the tokens otherwise.
    std::string str;
    const auto loc = parser.getCurrentLocation();
    if (parser.parseString(&str)) return failure();
    StringRef window = str;

    // Try parsing that string as a double literal.
    APFloat value(APFloat::IEEEdouble());
    auto maybeFloat = value.convertFromString(
        window,
        APFloat::roundingMode::NearestTiesToEven);
    auto error = maybeFloat.takeError();
    if (!error) {
        if (!value.isFinite() || value.isNaN())
            return parser.emitError(loc, "expected rational value");
        return Rational(value.convertToDouble());
    } else
        consumeError(std::move(error));

    // Parse a binary rational literal.
    exponent_t exponent;
    if (window.consumeInteger(10, mantissa)
        || !window.consume_front_insensitive("p")
        || window.consumeInteger(10, exponent) || !window.empty()) {
        parser.emitError(loc, "expected binary rational literal");
        return failure();
    }

    return Rational(mantissa, exponent);
}

void Rational::printField(raw_ostream &os) const
{
    reduce();

    // Print a bare integer literal if possible.
    if (const auto i64 = tryGetSInt()) {
        os << *i64;
        return;
    }

    // Print a quoted binary rational literal otherwise.
    os << "\"";
    if (const auto f64 = tryGetF64()) {
        // Prefer the floating-point short hand.
        // NOTE: Default float printing to llvm::raw_ostream is lossy, and the
        //       other formatting options have too much overhead. Let C++20 do
        //       the right thing for once!
        std::array<char, 32> buffer;
        const auto [end, ec] =
            std::to_chars(buffer.begin(), buffer.end(), *f64);
        assert(ec == std::error_code{});
        os << StringRef(buffer.begin(), std::distance(buffer.begin(), end));
    } else {
        getMantissa().print(os, true);
        os << "P";
        os << getExponent();
    }
    os << "\"";
}

auto Rational::compareImpl(const APInt &lhs, const APInt &rhs)
    -> std::strong_ordering
{
    // The preceeding sketch must have handled sign differences.
    assert(lhs.isNegative() == rhs.isNegative());

    // Perform a regular unsigned comparison, and then flip the relation if the
    // sign of both operands was negative.
    const auto result = cmp_uint(lhs, rhs);
    return lhs.isNegative() ? 0 <=> result : result;
}
