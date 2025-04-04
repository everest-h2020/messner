/// Implement common utility functions of the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"

#include "messner/Dialect/EKL/Analysis/Number.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <bit>
#include <limits>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::ekl;

[[nodiscard]]
LiteralAttr mlir::ekl::makeLiteral(BroadcastType type, int value)
{
    assert(type);

    using value_type    = std::remove_cvref_t<decltype(value)>;
    constexpr auto bits = std::numeric_limits<value_type>::digits;
    llvm::APInt apValue(
        bits,
        std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(value)),
        true);

    return llvm::TypeSwitch<BroadcastType, LiteralAttr>(type)
        .Case([&](NumberType numTy) -> LiteralAttr {
            return NumberAttr::get(numTy.getContext(), Number(value));
        })
        .Case([&](ekl::IndexType indexTy) -> LiteralAttr {
            if (value < 0
                || static_cast<unsigned>(value) > indexTy.getUpperBound())
                return {};

            return ekl::IndexAttr::get(indexTy.getContext(), value);
        })
        .Case([&](mlir::IntegerType intTy) -> LiteralAttr {
            if (intTy.isSigned()) {
                const auto maybeMin =
                    llvm::APInt::getSignedMinValue(intTy.getWidth())
                        .trySExtValue();
                if (maybeMin && value < *maybeMin) return {};
                const auto maybeMax =
                    llvm::APInt::getSignedMaxValue(intTy.getWidth())
                        .trySExtValue();
                if (maybeMax && value > *maybeMax) return {};
            } else {
                const auto maybeMax =
                    llvm::APInt::getAllOnes(intTy.getWidth()).tryZExtValue();
                if (value < 0) return {};
                if (maybeMax && static_cast<unsigned>(value) > *maybeMax)
                    return {};
            }

            return llvm::cast<LiteralAttr>(
                mlir::IntegerAttr::get(intTy, apValue));
        })
        .Case([&](FloatType floatTy) -> LiteralAttr {
            auto &sema = floatTy.getFloatSemantics();
            llvm::APFloat value(sema);

            const auto status = value.convertFromAPInt(
                apValue,
                true,
                llvm::APFloat::roundingMode::NearestTiesToEven);
            if (status & llvm::APFloat::opStatus::opInexact) return {};

            return FloatAttr::get(floatTy, value);
        })
        .Case([&](ArrayType arrayTy) -> LiteralAttr {
            const auto scalar = llvm::cast_if_present<ScalarAttr>(
                makeLiteral(arrayTy.getScalarType(), value));
            if (!scalar) return {};
            return ekl::ArrayAttr::get(scalar, arrayTy.getExtents());
        })
        .Default([](auto) -> LiteralAttr { return {}; });
}

//===----------------------------------------------------------------------===//
// coerce
//===----------------------------------------------------------------------===//

ekl::IntegerAttr mlir::ekl::coerce(ScalarAttr input, ekl::IntegerType output)
{
    assert(input && output);

    return llvm::TypeSwitch<ScalarAttr, ekl::IntegerAttr>(input)
        .Case([&](NumberAttr attr) {
            auto value = attr.getValue();
            value.roundTowardsZero();
            return ::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(value.getMantissa(), true)),
                output);
        })
        .Case([&](ekl::IntegerAttr attr) {
            auto value    = attr.getValue();
            auto adjValue = attr.getType().isSigned()
                              ? value.sextOrTrunc(output.getWidth())
                              : value.zextOrTrunc(output.getWidth());
            return ekl::IntegerAttr::get(
                input.getContext(),
                llvm::APSInt(adjValue, output.isUnsigned()));
        })
        .Case([&](FloatAttr attr) {
            llvm::APSInt result(output.getWidth(), output.isUnsigned());
            bool isExact;
            attr.getValue().convertToInteger(
                result,
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &isExact);
            return ekl::IntegerAttr::get(attr.getContext(), result);
        })
        .Case([&](ekl::IndexAttr attr) {
            return ::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(llvm::APInt(64U, attr.getValue()), true)),
                output);
        })
        .Default(ekl::IntegerAttr{});
}

FloatAttr mlir::ekl::coerce(ScalarAttr input, FloatType output)
{
    assert(input && output);

    return llvm::TypeSwitch<ScalarAttr, FloatAttr>(input)
        .Case([&](NumberAttr attr) {
            return FloatAttr::get(
                output,
                attr.getValue().toAPFloatWithRounding(
                    const_cast<llvm::fltSemantics &>(
                        output.getFloatSemantics())));
        })
        .Case([&](ekl::IntegerAttr attr) {
            llvm::APFloat value(output.getFloatSemantics());
            value.convertFromAPInt(
                attr.getValue(),
                attr.getType().isSigned(),
                llvm::APFloat::roundingMode::NearestTiesToEven);
            return FloatAttr::get(output, value);
        })
        .Case([&](FloatAttr attr) {
            auto value = attr.getValue();
            bool losesInfo;
            value.convert(
                output.getFloatSemantics(),
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &losesInfo);
            return FloatAttr::get(output, value);
        })
        .Case([&](ekl::IndexAttr attr) {
            return FloatAttr::get(output, static_cast<double>(attr.getValue()));
        })
        .Default(FloatAttr{});
}

ekl::IndexAttr mlir::ekl::coerce(ScalarAttr input, ekl::IndexType output)
{
    assert(input && output);

    return llvm::TypeSwitch<ScalarAttr, ekl::IndexAttr>(input)
        .Case([&](NumberAttr attr) {
            auto value = attr.getValue();
            value.roundTowardsZero();
            return ekl::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(value.getMantissa(), true)),
                output);
        })
        .Case([&](ekl::IntegerAttr attr) {
            auto value = attr.getValue();
            if (value.getActiveBits() > 64U) return ekl::IndexAttr{};
            const auto intValue = value.getZExtValue();
            if (intValue > output.getUpperBound()) return ekl::IndexAttr{};
            return ekl::IndexAttr::get(input.getContext(), intValue);
        })
        .Case([&](FloatAttr attr) {
            llvm::APSInt intValue(64U, true);
            bool isExact;
            attr.getValue().convertToInteger(
                intValue,
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &isExact);
            return ekl::IndexAttr::get(
                input.getContext(),
                intValue.getZExtValue());
        })
        .Case([&](ekl::IndexAttr attr) {
            if (attr.getValue() > output.getUpperBound())
                return ekl::IndexAttr{};
            return attr;
        })
        .Default(ekl::IndexAttr{});
}

ScalarAttr mlir::ekl::coerce(ScalarAttr input, ScalarType output)
{
    assert(input && output);

    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](ekl::IntegerType type) { return ::coerce(input, type); })
        .Case([&](FloatType type) { return ::coerce(input, type); })
        .Case([&](ekl::IndexType type) { return ::coerce(input, type); })
        .Default(ScalarAttr{});
}

ekl::ArrayAttr mlir::ekl::coerce(ekl::ArrayAttr input, ScalarType output)
{
    assert(input && output);

    SmallVector<Attribute> stack(input.getStack().getValue());
    for (auto &attr : stack) {
        attr =
            llvm::TypeSwitch<Attribute, Attribute>(attr)
                .Case(
                    [&](ekl::ArrayAttr array) { return coerce(array, output); })
                .Case([&](ScalarAttr scalar) { return coerce(scalar, output); })
                .Default([](auto) -> Attribute { return {}; });

        if (!attr) return {};
    }

    return ekl::ArrayAttr::get(input.getArrayType().cloneWith(output), stack);
}
