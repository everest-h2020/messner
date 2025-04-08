/// Declaration of the EKL dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/Types.h" // IWYU pragma: keep

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ODS types
//===----------------------------------------------------------------------===//
//
// Forward declarations of the ODS-generated attributes, so that we can use them
// to declare the named constraints.

class RationalAttr;
class IndexAttr;
class ArrayAttr;
class SliceAttr;
class AxisAttr;
class EllipsisAttr;

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

struct IntegerAttr : mlir::IntegerAttr {
    static auto classof(mlir::IntegerAttr attr) -> bool;
    static auto classof(Attribute attr) -> bool;

    /// Obtains an IntegerAttr for @p type with @p value .
    ///
    /// No check is performed to determine whether @p value is an appropriate
    /// value for @p type .
    ///
    /// @pre    `type`
    static auto get(IntegerType type, const llvm::APInt &value) -> IntegerAttr;

    /*implicit*/ IntegerAttr() = default;
    /*implicit*/ IntegerAttr(const mlir::Attribute::ImplType *impl);

    auto getType() const -> IntegerType;
};

struct ScalarAttr : Attribute {
    static auto classof(FloatAttr) -> bool;
    static auto classof(IntegerAttr) -> bool;
    static auto classof(BoolAttr) -> bool;
    static auto classof(RationalAttr) -> bool;
    static auto classof(IndexAttr) -> bool;
    static auto classof(Attribute attr) -> bool;

    using Attribute::Attribute;
    /*implicit*/ ScalarAttr(FloatAttr attr);
    /*implicit*/ ScalarAttr(IntegerAttr attr);
    /*implicit*/ ScalarAttr(BoolAttr attr);
    /*implicit*/ ScalarAttr(RationalAttr attr);
    /*implicit*/ ScalarAttr(IndexAttr attr);

    auto getType() const -> ScalarType;
};

struct LiteralAttr : Attribute {
    static auto classof(ScalarAttr) -> bool;
    static auto classof(StringAttr) -> bool;
    static auto classof(ArrayAttr) -> bool;
    static auto classof(SliceAttr) -> bool;
    static auto classof(AxisAttr) -> bool;
    static auto classof(EllipsisAttr) -> bool;
    static auto classof(Attribute attr) -> bool;

    using Attribute::Attribute;
    /*implicit*/ LiteralAttr(ScalarAttr attr);
    /*implicit*/ LiteralAttr(StringAttr attr);
    /*implicit*/ LiteralAttr(ArrayAttr attr);
    /*implicit*/ LiteralAttr(SliceAttr attr);
    /*implicit*/ LiteralAttr(AxisAttr attr);
    /*implicit*/ LiteralAttr(EllipsisAttr attr);

    auto getType() const -> LiteralType;
};

struct NumberAttr : ScalarAttr {
    static auto classof(FloatAttr) -> bool;
    static auto classof(IntegerAttr) -> bool;
    static auto classof(RationalAttr) -> bool;
    static auto classof(Attribute attr) -> bool;

    /*implicit*/ NumberAttr() = default;
    /*implicit*/ NumberAttr(const ImplType *impl);
    /*implicit*/ NumberAttr(FloatAttr attr);
    /*implicit*/ NumberAttr(IntegerAttr attr);
    /*implicit*/ NumberAttr(IndexAttr attr);
    /*implicit*/ NumberAttr(RationalAttr attr);

    auto getType() const -> NumberType;
};

struct BroadcastAttr : Attribute {
    static auto classof(ScalarAttr) -> bool;
    static auto classof(ArrayAttr) -> bool;
    static auto classof(Attribute attr) -> bool;

    using Attribute::Attribute;
    /*implicit*/ BroadcastAttr(ScalarAttr attr);
    /*implicit*/ BroadcastAttr(ArrayAttr attr);

    auto getType() const -> BroadcastType;
};

struct LogicAttr : BroadcastAttr {
    static auto classof(BoolAttr) -> bool;
    static auto classof(ArrayAttr attr) -> bool;
    static auto classof(Attribute attr) -> bool;

    /*implicit*/ LogicAttr() = default;
    /*implicit*/ LogicAttr(const ImplType *impl);
    /*implicit*/ LogicAttr(BoolAttr attr);

    auto getType() const -> LogicType;
};

struct ArithmeticAttr : BroadcastAttr {
    static auto classof(NumberAttr) -> bool;
    static auto classof(ArrayAttr attr) -> bool;
    static auto classof(Attribute attr) -> bool;

    /*implicit*/ ArithmeticAttr() = default;
    /*implicit*/ ArithmeticAttr(const ImplType *impl);
    /*implicit*/ ArithmeticAttr(NumberAttr attr);

    auto getType() const -> ArithmeticType;
};

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "messner/Dialect/EKL/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// IntegerAttr implementation
//===----------------------------------------------------------------------===//

inline auto IntegerAttr::classof(mlir::IntegerAttr attr) -> bool
{
    return llvm::isa<IntegerType>(attr.getType());
}

inline auto IntegerAttr::classof(Attribute attr) -> bool
{
    if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr); intAttr)
        return classof(intAttr);
    return false;
}

inline auto IntegerAttr::get(IntegerType type, const llvm::APInt &value)
    -> IntegerAttr
{
    assert(type);

    return llvm::cast<IntegerAttr>(mlir::IntegerAttr::get(type, value));
}

inline IntegerAttr::IntegerAttr(const mlir::Attribute::ImplType *impl)
        : mlir::IntegerAttr(impl)
{}

inline auto IntegerAttr::getType() const -> IntegerType
{
    return llvm::cast<IntegerType>(mlir::IntegerAttr::getType());
}

//===----------------------------------------------------------------------===//
// ScalarAttr implementation
//===----------------------------------------------------------------------===//

inline auto ScalarAttr::classof(FloatAttr) -> bool { return true; }

inline auto ScalarAttr::classof(IntegerAttr) -> bool { return true; }

inline auto ScalarAttr::classof(BoolAttr) -> bool { return true; }

inline auto ScalarAttr::classof(RationalAttr) -> bool { return true; }

inline auto ScalarAttr::classof(IndexAttr) -> bool { return true; }

inline auto ScalarAttr::classof(Attribute attr) -> bool
{
    return llvm::isa<FloatAttr, IntegerAttr, BoolAttr, RationalAttr, IndexAttr>(
        attr);
}

inline ScalarAttr::ScalarAttr(FloatAttr attr)
        : ScalarAttr(static_cast<Attribute>(attr).getImpl())
{}

inline ScalarAttr::ScalarAttr(IntegerAttr attr)
        : ScalarAttr(static_cast<Attribute>(attr).getImpl())
{}

inline ScalarAttr::ScalarAttr(BoolAttr attr)
        : ScalarAttr(static_cast<Attribute>(attr).getImpl())
{}

inline ScalarAttr::ScalarAttr(RationalAttr attr)
        : ScalarAttr(static_cast<Attribute>(attr).getImpl())
{}

inline ScalarAttr::ScalarAttr(IndexAttr attr)
        : ScalarAttr(static_cast<Attribute>(attr).getImpl())
{}

inline auto ScalarAttr::getType() const -> ScalarType
{
    return llvm::TypeSwitch<Attribute, ScalarType>(*this)
        .Case([](FloatAttr attr) -> ScalarType {
            return llvm::cast<FloatType>(attr.getType());
        })
        .Case([](IntegerAttr attr) -> ScalarType { return attr.getType(); })
        .Case([](BoolAttr attr) -> ScalarType {
            return BoolType::get(attr.getContext());
        })
        .Case([](RationalAttr attr) -> ScalarType { return attr.getType(); })
        .Case([](IndexAttr attr) -> ScalarType { return attr.getType(); });
}

//===----------------------------------------------------------------------===//
// LiteralAttr implementation
//===----------------------------------------------------------------------===//

inline auto LiteralAttr::classof(ScalarAttr) -> bool { return true; }

inline auto LiteralAttr::classof(StringAttr) -> bool { return true; }

inline auto LiteralAttr::classof(ArrayAttr) -> bool { return true; }

inline auto LiteralAttr::classof(SliceAttr) -> bool { return true; }

inline auto LiteralAttr::classof(AxisAttr) -> bool { return true; }

inline auto LiteralAttr::classof(EllipsisAttr) -> bool { return true; }

inline auto LiteralAttr::classof(Attribute attr) -> bool
{
    return llvm::isa<
        ScalarAttr,
        StringAttr,
        ArrayAttr,
        SliceAttr,
        AxisAttr,
        EllipsisAttr>(attr);
}

inline LiteralAttr::LiteralAttr(ScalarAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(StringAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(ArrayAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(SliceAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(AxisAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(EllipsisAttr attr)
        : LiteralAttr(static_cast<Attribute>(attr).getImpl())
{}

inline auto LiteralAttr::getType() const -> LiteralType
{
    return llvm::TypeSwitch<Attribute, LiteralType>(*this)
        .Case([](ScalarAttr attr) -> LiteralType { return attr.getType(); })
        .Case([](StringAttr attr) -> LiteralType {
            return StringType::get(attr.getContext());
        })
        .Case([](ArrayAttr attr) -> LiteralType { return attr.getType(); })
        .Case([](SliceAttr attr) -> LiteralType { return attr.getType(); })
        .Case([](AxisAttr attr) -> LiteralType { return attr.getType(); })
        .Case([](EllipsisAttr attr) -> LiteralType { return attr.getType(); });
}

//===----------------------------------------------------------------------===//
// NumberAttr implementation
//===----------------------------------------------------------------------===//

inline auto NumberAttr::classof(FloatAttr) -> bool { return true; }

inline auto NumberAttr::classof(IntegerAttr) -> bool { return true; }

inline auto NumberAttr::classof(RationalAttr) -> bool { return true; }

inline auto NumberAttr::classof(Attribute attr) -> bool
{
    return llvm::isa<FloatAttr, IntegerAttr, IndexAttr, RationalAttr>(attr);
}

inline NumberAttr::NumberAttr(const ImplType *impl) : ScalarAttr(impl) {}

inline NumberAttr::NumberAttr(FloatAttr attr)
        : NumberAttr(static_cast<Attribute>(attr).getImpl())
{}

inline NumberAttr::NumberAttr(IntegerAttr attr)
        : NumberAttr(static_cast<Attribute>(attr).getImpl())
{}

inline NumberAttr::NumberAttr(RationalAttr attr)
        : NumberAttr(static_cast<Attribute>(attr).getImpl())
{}

inline auto NumberAttr::getType() const -> NumberType
{
    return llvm::TypeSwitch<Attribute, NumberType>(*this)
        .Case([](FloatAttr attr) -> NumberType {
            return llvm::cast<FloatType>(attr.getType());
        })
        .Case([](IntegerAttr attr) -> NumberType { return attr.getType(); })
        .Case([](RationalAttr attr) -> NumberType { return attr.getType(); });
}

//===----------------------------------------------------------------------===//
// BroadcastAttr implementation
//===----------------------------------------------------------------------===//

inline auto BroadcastAttr::classof(ScalarAttr) -> bool { return true; }

inline auto BroadcastAttr::classof(ArrayAttr) -> bool { return true; }

inline auto BroadcastAttr::classof(Attribute type) -> bool
{
    return llvm::isa<ScalarAttr, ArrayAttr>(type);
}

inline BroadcastAttr::BroadcastAttr(ScalarAttr type)
        : BroadcastAttr(static_cast<Attribute>(type).getImpl())
{}

inline BroadcastAttr::BroadcastAttr(ArrayAttr type)
        : BroadcastAttr(static_cast<Attribute>(type).getImpl())
{}

inline auto BroadcastAttr::getType() const -> BroadcastType
{
    return llvm::TypeSwitch<Attribute, BroadcastType>(*this)
        .Case([](ScalarAttr attr) -> BroadcastType { return attr.getType(); })
        .Case([](ArrayAttr attr) -> BroadcastType { return attr.getType(); });
}

//===----------------------------------------------------------------------===//
// LogicAttr implementation
//===----------------------------------------------------------------------===//

inline auto LogicAttr::classof(BoolAttr) -> bool { return true; }

inline auto LogicAttr::classof(ArrayAttr attr) -> bool
{
    return llvm::isa<BoolType>(attr.getType().getScalarType());
}

inline auto LogicAttr::classof(Attribute type) -> bool
{
    if (const auto arrayAttr = llvm::dyn_cast<ArrayAttr>(type); arrayAttr)
        return classof(arrayAttr);
    return llvm::isa<BoolAttr>(type);
}

inline LogicAttr::LogicAttr(const ImplType *impl) : BroadcastAttr(impl) {}

inline LogicAttr::LogicAttr(BoolAttr type)
        : LogicAttr(static_cast<Attribute>(type).getImpl())
{}

inline auto LogicAttr::getType() const -> LogicType
{
    return llvm::TypeSwitch<Attribute, LogicType>(*this)
        .Case([](BoolAttr attr) -> LogicType {
            return BoolType::get(attr.getContext());
        })
        .Case([](ArrayAttr attr) -> LogicType {
            return llvm::cast<LogicType>(attr.getType());
        });
}

//===----------------------------------------------------------------------===//
// ArithmeticAttr implementation
//===----------------------------------------------------------------------===//

inline auto ArithmeticAttr::classof(NumberAttr) -> bool { return true; }

inline auto ArithmeticAttr::classof(ArrayAttr attr) -> bool
{
    return llvm::isa<NumberType>(attr.getType().getScalarType());
}

inline auto ArithmeticAttr::classof(Attribute attr) -> bool
{
    if (const auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr); arrayAttr)
        return classof(arrayAttr);
    return llvm::isa<NumberAttr>(attr);
}

inline ArithmeticAttr::ArithmeticAttr(const ImplType *impl)
        : BroadcastAttr(impl)
{}

inline ArithmeticAttr::ArithmeticAttr(NumberAttr type)
        : ArithmeticAttr(static_cast<Attribute>(type).getImpl())
{}

inline auto ArithmeticAttr::getType() const -> ArithmeticType
{
    return llvm::TypeSwitch<Attribute, ArithmeticType>(*this)
        .Case([](NumberAttr attr) -> ArithmeticType { return attr.getType(); })
        .Case([](ArrayAttr attr) -> ArithmeticType {
            return llvm::cast<ArithmeticType>(attr.getType());
        });
}

//===----------------------------------------------------------------------===//
// RationalAttr implementation
//===----------------------------------------------------------------------===//

inline auto RationalAttr::getType() const -> RationalType
{
    return RationalType::get(getContext());
}

//===----------------------------------------------------------------------===//
// IndexAttr implementation
//===----------------------------------------------------------------------===//

inline auto IndexAttr::getType() const -> IndexType
{
    return IndexType::get(getContext(), getValue());
}

//===----------------------------------------------------------------------===//
// ArrayAttr implementation
//===----------------------------------------------------------------------===//

inline auto ArrayAttr::get(ArrayType arrayType, TupleAttr::ValueType stack)
    -> ArrayAttr
{
    assert(arrayType);

    return get(arrayType, TupleAttr::get(arrayType.getContext(), stack));
}

inline auto ArrayAttr::get(ArrayType arrayType, ScalarAttr splatValue)
    -> ArrayAttr
{
    assert(arrayType);
    assert(splatValue);

    return get(arrayType, TupleAttr::get(arrayType.getContext(), splatValue));
}

inline auto ArrayAttr::get(ScalarAttr splatValue, ShapeRef shape) -> ArrayAttr
{
    assert(splatValue);

    return get(ArrayType::get(splatValue.getType(), shape), splatValue);
}

inline auto ArrayAttr::get(ArrayAttr bcastValue, Extent extent) -> ArrayAttr
{
    assert(bcastValue);

    const auto bcastTy = bcastValue.getType();
    return get(
        bcastTy.cloneWith(concat(extent, bcastTy.getShape())),
        TupleAttr::get(bcastValue.getContext(), bcastValue));
}

inline auto ArrayAttr::isBroadcast() const -> bool
{
    return getStack().size() == 1U;
}

inline auto ArrayAttr::getBroadcastValue() const -> BroadcastAttr
{
    return isBroadcast() ? llvm::cast<BroadcastAttr>(*getStack().begin())
                         : BroadcastAttr{};
}

inline auto ArrayAttr::isSplat() const -> bool { return !!getSplatValue(); }

inline auto ArrayAttr::getSplatValue() const -> ScalarAttr
{
    return llvm::dyn_cast_if_present<ScalarAttr>(getBroadcastValue());
}

//===----------------------------------------------------------------------===//
// SliceAttr implementation
//===----------------------------------------------------------------------===//

inline auto SliceAttr::getType() const -> SliceType
{
    return SliceType::get(getContext(), getValue());
}

//===----------------------------------------------------------------------===//
// AxisAttr implementation
//===----------------------------------------------------------------------===//

inline auto AxisAttr::getType() const -> AxisType
{
    return AxisType::get(getContext());
}

//===----------------------------------------------------------------------===//
// EllipsisAttr implementation
//===----------------------------------------------------------------------===//

inline auto EllipsisAttr::getType() const -> EllipsisType
{
    return EllipsisType::get(getContext());
}

} // namespace mlir::ekl
