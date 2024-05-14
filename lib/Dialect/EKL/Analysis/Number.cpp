/// Implements the Number literal type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/Number.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Error.h"

#include <algorithm>

using namespace llvm;
using namespace mlir;
using namespace mlir::ekl;

/// Obtains the range of words in @p apint  up to the highest non-zero word.
///
/// @post   `!result.empty()`
[[nodiscard]] static ArrayRef<APInt::WordType>
getActiveWords(const APInt &apint)
{
    const auto numWords = std::max(1U, apint.getActiveWords());
    return ArrayRef<APInt::WordType>(apint.getRawData(), numWords);
}

/// Compares @p lhs and @p rhs as if they were unsigned integers.
[[nodiscard]] static std::strong_ordering
cmp_uint(const APInt &lhs, const APInt &rhs)
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

/// Compares @p lhs and @p rhs as if they were signed integers of the same sign.
///
/// @pre    `lhs.isNegative() == rhs.isNegative()`
[[nodiscard]] static std::strong_ordering
cmp_sint_same_sign(const APInt &lhs, const APInt &rhs)
{
    assert(lhs.isNegative() == rhs.isNegative());

    // Perform a regular unsigned comparison, and then flip the relation if the
    // sign of both operands was negative.
    const auto result = cmp_uint(lhs, rhs);
    return lhs.isNegative() ? 0 <=> result : result;
}

//===----------------------------------------------------------------------===//
// Number implementation
//===----------------------------------------------------------------------===//

Number::Number(llvm::APFloat value) : Number()
{
    assert(value.isIEEE() && value.isFinite());

    // Decompose value into exponent and a normalized fraction.
    int exp;
    // NOTE: Is always exact, no rounding mode used in base2.
    value =
        llvm::frexp(value, exp, llvm::APFloat::roundingMode::NearestTiesToEven);
    m_exponent = exp;

    // Convert the normalized fraction into an integer.
    const auto prec = APFloat::semanticsPrecision(value.getSemantics());
    // NOTE: Is always exact..
    value           = llvm::scalbn(
        value,
        -prec,
        llvm::APFloat::roundingMode::NearestTiesToEven);
    m_exponent -= prec;

    // Extract the integer mantissa.
    bool isExact;
    llvm::APSInt mantissa;
    value.convertToInteger(
        mantissa,
        llvm::APFloat::roundingMode::NearestTiesToEven,
        &isExact);
    assert(isExact);

    // Store the mantissa, and ensure that we didn't break its sign.
    m_mantissa = mantissa;
    if (mantissa.isUnsigned() && m_mantissa.isNegative())
        m_mantissa = m_mantissa.zext(m_mantissa.getBitWidth() + 1U);

    reduce();
}

void mlir::ekl::Number::shrinkToFit()
{
    reduce();

    auto padding = getMantissa().countr_zero();
    if (padding <= 1) {
        // We can't remove the last bit of padding or we would change the sign.
        return;
    }
    if (padding == getMantissa().getBitWidth()) {
        // The value is zero, normalize to the default-constructed instance.
        m_mantissa = llvm::APInt{};
        m_exponent = 0;
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
    m_mantissa = llvm::APInt(getMantissa().getBitWidth() - padding, newWords);
}

llvm::APFloat
mlir::ekl::Number::toAPFloatWithRounding(llvm::fltSemantics &semantics) const
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

std::optional<Number::uword_t> mlir::ekl::Number::tryGetUInt() const
{
    if (getExponent() < 0) return std::nullopt;
    if (getMantissa().isNegative()) return std::nullopt;

    const auto bits = getMantissa().getActiveBits() + getExponent();
    if (bits > word_bits) return std::nullopt;

    return getMantissa().getZExtValue() << getExponent();
}

std::optional<Number::sword_t> mlir::ekl::Number::tryGetSInt() const
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

std::optional<double> mlir::ekl::Number::tryGetF64() const
{
    reduce();

    // NOTE: Technically, if the exponent doesn't fit, we could try to shift
    //       the mantissa to reduce the exponent. However, we would gain very
    //       little from that.

    if (getExponent() > std::numeric_limits<double>::max_exponent)
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

FailureOr<Number> mlir::ekl::Number::parseField(AsmParser &parser)
{
    // Try parsing a bare integer literal.
    APInt mantissa;
    const auto maybeInt = parser.parseOptionalInteger(mantissa);
    if (maybeInt.has_value()) {
        if (*maybeInt) return failure();
        return Number(mantissa);
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
        if (maybeFloat.get() == APFloat::opStatus::opOK)
            return Number(value.convertToDouble());
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

    return Number(mantissa, exponent);
}

void mlir::ekl::Number::printField(raw_ostream &os) const
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
        os << *f64;
    } else {
        getMantissa().print(os, true);
        os << "P";
        os << getExponent();
    }
    os << "\"";
}

std::strong_ordering
mlir::ekl::Number::compareImpl(const APInt &lhs, const APInt &rhs)
{
    // The preceeding sketch comparison has gotten the sign out of the way.
    return cmp_sint_same_sign(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

ParseResult mlir::ekl::parseNumber(AsmParser &parser, Number &number)
{
    auto maybeNumber = Number::parseField(parser);
    if (failed(maybeNumber)) return failure();

    number = std::move(*maybeNumber);
    return success();
}

void mlir::ekl::printNumber(AsmPrinter &printer, const Number &number)
{
    number.printField(printer.getStream());
}
