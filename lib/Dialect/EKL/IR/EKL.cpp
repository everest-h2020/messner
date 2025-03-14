/// Implement common utility functions of the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/EKL.h"

#include "messner/Dialect/EKL/Analysis/Number.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::ekl;

[[nodiscard]]
static LiteralAttr makeLiteral(BroadcastType type, int value)
{
    assert(type);

    return llvm::TypeSwitch<BroadcastType, LiteralAttr>(type)
        .Case([&](NumberType numTy) -> LiteralAttr {
            return NumberAttr::get(numTy.getContext(), Number(value));
        })
        .Case([&](ekl::IndexType indexTy) -> LiteralAttr {
            if (value < 0 || (unsigned)value > indexTy.getUpperBound())
                return {};
            return ekl::IndexAttr::get(indexTy.getContext(), value);
        })
        .Case([&](mlir::IntegerType intTy) -> LiteralAttr {
            return llvm::cast<LiteralAttr>(
                mlir::IntegerAttr::get(intTy, value));
        })
        .Case([&](FloatType floatTy) -> LiteralAttr {
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

Value makeZero(OpBuilder &builder, Location loc, BroadcastType type)
{
    assert(type);

    const auto literal = makeLiteral(type, 0);
    assert(literal);

    return builder.create<LiteralOp>(loc, literal).getResult();
}

Value makeOne(OpBuilder &builder, Location loc, BroadcastType type)
{
    assert(type);

    const auto literal = makeLiteral(type, 1);
    assert(literal);

    return builder.create<LiteralOp>(loc, literal).getResult();
}
