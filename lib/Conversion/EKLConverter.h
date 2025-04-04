#pragma once

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir/Support/LLVM.h>

namespace mlir::ekl {

inline Value createUnrealizedCast(
    OpBuilder &builder,
    Type resultTy,
    ValueRange inputs,
    Location loc)
{
    if (inputs.size() != 1) return {};
    return builder.create<UnrealizedConversionCastOp>(loc, resultTy, inputs)
        .getResult(0);
};

[[nodiscard]]
inline TypeConverter createStandardConverter()
{
    // Create a helper TypeConverter to provide arith & index op types.
    TypeConverter converter;

    // Float types pass through unchanged.
    converter.addConversion([](mlir::FloatType type) { return type; });
    // Integer types must be converted to signless integers.
    converter.addConversion([](mlir::IntegerType type) {
        if (type.isSignless()) return type;
        return mlir::IntegerType::get(type.getContext(), type.getWidth());
    });
    // The index type is converted to the mlir index type.
    converter.addConversion([](ekl::IndexType type) {
        return mlir::IndexType::get(type.getContext());
    });

    // In any case, none of these conversions are actually performed, we
    // always use an unrealized cast rely on them going away once everything
    // is converted to standard.
    converter.addTargetMaterialization(createUnrealizedCast);
    converter.addSourceMaterialization(createUnrealizedCast);

    return converter;
}

[[nodiscard]]
inline TypeConverter createEKLConverter(TypeConverter &stdConverter)
{
    TypeConverter converter;

    // Unpack the contained expression type and convert it with the helper.
    converter.addConversion([&](ekl::ExpressionType exprTy) -> Type {
        if (const auto boundTy = exprTy.getTypeBound())
            return stdConverter.convertType(boundTy);
        return {};
    });

    // Unwrap expressions as values using the ekl.eval op.
    converter.addTargetMaterialization(
        [&](OpBuilder &builder, Type resultTy, ValueRange inputs, Location loc)
            -> Value {
            if (inputs.size() != 1) return {};
            const auto exprTy =
                llvm::dyn_cast<ekl::ExpressionType>(inputs.front().getType());
            if (!exprTy) return {};

            const auto eval = builder
                                  .create<ekl::EvalOp>(
                                      loc,
                                      inputs.front(),
                                      exprTy.getTypeBound())
                                  .getResult();

            if (eval.getType() == resultTy) return eval;

            // Use the helper to perform the standard conversion.
            return stdConverter
                .materializeTargetConversion(builder, loc, resultTy, {eval});
        });

    // Wrap values as expressions using the ekl.intro op.
    converter.addSourceMaterialization(
        [&](OpBuilder &builder,
            ekl::ExpressionType resultTy,
            ValueRange inputs,
            Location loc) -> Value {
            if (inputs.size() != 1 || !resultTy.getTypeBound()) return {};

            auto intro = inputs.front();
            if (intro.getType() != resultTy.getTypeBound()) {
                // Use the helper to perform the standard conversion.
                intro = stdConverter.materializeSourceConversion(
                    builder,
                    loc,
                    resultTy.getTypeBound(),
                    inputs);
            }

            return builder.create<ekl::IntroOp>(loc, intro);
        });

    return converter;
}

} // namespace mlir::ekl
