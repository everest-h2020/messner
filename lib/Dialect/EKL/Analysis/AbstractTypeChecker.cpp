/// Implements the AbstractTypeChecker.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"

#include "messner/Dialect/EKL/IR/EKL.h"

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Subtype relation
//===----------------------------------------------------------------------===//

bool mlir::ekl::isSubtype(Type subtype, Type supertype)
{
    // T :> T
    if (subtype == supertype) return true;

    // nullptr :> T
    if (!supertype) return true;
    // T :> nullptr <=> T = nullptr
    if (!subtype) return false;

    // Number :> Float, Number :> Integer, Number :> Index
    if (llvm::isa<NumberType>(supertype))
        return llvm::isa<NumericType>(subtype);

    // ... -> Float :> T
    if (const auto supFloat = llvm::dyn_cast<FloatType>(supertype)) {
        const auto maybe =
            llvm::TypeSwitch<Type, std::optional<bool>>(subtype)
                .Case([&](FloatType subFloat) {
                    return isSubtype(subFloat, supFloat);
                })
                .Case([&](ekl::IntegerType subInt) {
                    return isSubtype(subInt, supFloat);
                })
                .Case([&](ekl::IndexType subIndex) {
                    return isSubtype(subIndex.getIntegerType(), supFloat);
                })
                .Default(std::optional<bool>{});
        if (maybe) return *maybe;
    }

    // ... -> Integer :> T
    if (const auto supInt = llvm::dyn_cast<ekl::IntegerType>(supertype)) {
        const auto maybe =
            llvm::TypeSwitch<Type, std::optional<bool>>(subtype)
                .Case([&](ekl::IntegerType subInt) {
                    return isSubtype(subInt, supInt);
                })
                .Case([&](ekl::IndexType subIndex) {
                    return isSubtype(subIndex.getIntegerType(), supInt);
                })
                .Default(std::optional<bool>{});
        if (maybe) return *maybe;
    }

    // ... -> Index :> T
    if (const auto supIndex = llvm::dyn_cast<ekl::IndexType>(supertype)) {
        const auto maybe =
            llvm::TypeSwitch<Type, std::optional<bool>>(subtype)
                .Case([&](ekl::IndexType subIndex) {
                    return subIndex.getUpperBound() <= supIndex.getUpperBound();
                })
                .Case([&](ekl::IntegerType subInt) {
                    return isSubtype(subInt, supIndex.getIntegerType());
                })
                .Default(std::optional<bool>{});
        if (maybe) return *maybe;
    }

    // ... -> Array :> T
    if (const auto supArray = llvm::dyn_cast<ArrayType>(supertype)) {
        // T :> U -> Array(T, []) :> U
        if (supArray.getNumExtents() == 0
            && isSubtype(subtype, supArray.getScalarType()))
            return true;

        // T :> U -> Array(T, x) :> Array(U, x)
        if (const auto subArray = llvm::dyn_cast<ArrayType>(subtype)) {
            if (supArray.getExtents() == subArray.getExtents()
                && isSubtype(
                    subArray.getScalarType(),
                    supArray.getScalarType()))
                return true;
        }
    }

    // ... -> Ref :> T
    if (const auto supRef = llvm::dyn_cast<ReferenceType>(supertype)) {
        if (const auto subRef = llvm::dyn_cast<ReferenceType>(subtype)) {
            // Ref(inout, T) :> Ref(out, T)
            // Ref(inout, T) :> Ref(in, T)
            // T :> U -> Ref(x, T) :> Ref(x, U)
            if (isSubtype(subRef.getArrayType(), supRef.getArrayType())
                && bitEnumContainsAll(supRef.getKind(), subRef.getKind()))
                return true;
        }
    }

    return false;
}

bool mlir::ekl::isSubtype(ekl::IntegerType subtype, ekl::IntegerType supertype)
{
    assert(subtype && supertype);

    if (subtype.isSigned()) {
        // Signed types can not fit into unsigned types.
        if (supertype.isUnsigned()) return false;

        // With same signedness, we can just compare bit widths.
        return subtype.getWidth() <= supertype.getWidth();
    }

    // If the supertype is signed, we have to subtract the sign bit from the
    // width, but then we can just compare widths as normal.
    const auto superWidth =
        supertype.getWidth() - (supertype.isSigned() ? 1U : 0U);
    return subtype.getWidth() <= superWidth;
}

bool mlir::ekl::isSubtype(FloatType subtype, FloatType supertype)
{
    assert(subtype && supertype);

    auto result = supertype.isF128();
    if (subtype.isF128()) return result;
    result |= supertype.isF80();
    if (subtype.isF80()) return result;
    result |= supertype.isF64();
    if (subtype.isF64()) return result;
    result |= supertype.isF32();
    if (subtype.isF32() || subtype.isBF16()) return result;
    result |= supertype.isF16();
    if (subtype.isF16()) return result;

    // NOTE: We only use this as a fallback.
    return APFloat::isRepresentableAsNormalIn(
        subtype.getFloatSemantics(),
        supertype.getFloatSemantics());
}

bool mlir::ekl::isSubtype(ekl::IntegerType subtype, FloatType supertype)
{
    assert(subtype && supertype);

    if (!APFloat(supertype.getFloatSemantics()).isIEEE()) return false;

    // If the subtype is signed, we have to subtract the sign bit from the
    // width, but then we can just compare the width with the precision.
    const auto subWidth = subtype.getWidth() - (subtype.isSigned() ? 1U : 0U);
    return subWidth
        <= APFloat::semanticsPrecision(supertype.getFloatSemantics());
}

//===----------------------------------------------------------------------===//
// AbstractTypeChecker implementation
//===----------------------------------------------------------------------===//

AbstractTypeChecker::~AbstractTypeChecker()
{
    // NOTE: This ensures that the vtable is emitted in this translation unit.
}
