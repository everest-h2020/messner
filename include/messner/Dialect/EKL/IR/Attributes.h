/// Declaration of the EKL dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/Analysis/Number.h"
#include "messner/Dialect/EKL/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ODS attributes
//===----------------------------------------------------------------------===//
//
// Forward declarations of the ODS-generated attributes, so that we can use them
// to declare the named constraints.

class NumberAttr;
class IndexAttr;
class ArrayAttr;
class InitializerAttr;
class IdentityAttr;
class ExtentAttr;

//===----------------------------------------------------------------------===//
// Named constraints
//===----------------------------------------------------------------------===//

/// Implements the IntegerAttr named constraint.
struct IntegerAttr : mlir::IntegerAttr {
    using mlir::IntegerAttr::IntegerAttr;

    /// The type of the attribute value in C++.
    using ValueType = llvm::APSInt;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(mlir::IntegerAttr attr)
    {
        return llvm::isa<IntegerType>(attr.getType());
    }
    /// Determines whether @p attr is an IntegerAttr.
    [[nodiscard]] static bool classof(Attribute attr)
    {
        if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
            return classof(intAttr);
        return false;
    }

    /// Obtains the canonical IntegerAttr for @p value .
    ///
    /// @pre    `context`
    [[nodiscard]] static IntegerAttr
    get(MLIRContext *context, llvm::APSInt value)
    {
        return llvm::cast<IntegerAttr>(mlir::IntegerAttr::get(context, value));
    }

    /// Gets the underlying APSInt value.
    [[nodiscard]] llvm::APSInt getValue()
    {
        return llvm::APSInt(
            mlir::IntegerAttr::getValue(),
            getType().isUnsigned());
    }
    /// Gets the underlying IntegerType.
    [[nodiscard]] IntegerType getType()
    {
        return llvm::cast<IntegerType>(mlir::IntegerAttr::getType());
    }
};

/// Implements the ScalarAttr named constraint.
struct ScalarAttr : Attribute {
    using Attribute::Attribute;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(NumberAttr);
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ekl::IntegerAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(FloatAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ekl::IndexAttr);
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(BoolAttr) { return true; }
    /// Determines whether @p attr is a ScalarAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr);

    /*implicit*/ ScalarAttr(NumberAttr attr);
    /*implicit*/ ScalarAttr(ekl::IntegerAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}
    /*implicit*/ ScalarAttr(FloatAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}
    /*implicit*/ ScalarAttr(ekl::IndexAttr attr);
    /*implicit*/ ScalarAttr(BoolAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}

    /// Gets the underlying ScalarType of the contained value.
    [[nodiscard]] ScalarType getType() const;
};

/// Implements the LiteralAttr named constraint.
struct LiteralAttr : Attribute {
    using Attribute::Attribute;

    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ScalarAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(StringAttr) { return true; }
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ekl::ArrayAttr attr);
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(InitializerAttr attr);
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(IdentityAttr attr);
    /// @copydoc classof(Attribute)
    [[nodiscard]] static bool classof(ExtentAttr attr);
    /// Determines whether @p attr is a LiteralAttr.
    ///
    /// @pre    `attr`
    [[nodiscard]] static bool classof(Attribute attr);

    /*implicit*/ LiteralAttr(ScalarAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}
    /*implicit*/ LiteralAttr(StringAttr attr)
            : Attribute(static_cast<Attribute>(attr).getImpl())
    {}
    /*implicit*/ LiteralAttr(ekl::ArrayAttr attr);
    /*implicit*/ LiteralAttr(InitializerAttr attr);
    /*implicit*/ LiteralAttr(IdentityAttr attr);
    /*implicit*/ LiteralAttr(ExtentAttr attr);

    /// Gets the associated LiteralType.
    [[nodiscard]] LiteralType getType() const;
};

} // namespace mlir::ekl

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "messner/Dialect/EKL/IR/Attributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// ScalarAttr implementation
//===----------------------------------------------------------------------===//

inline bool ScalarAttr::classof(NumberAttr) { return true; }

inline bool ScalarAttr::classof(ekl::IndexAttr) { return true; }

inline ScalarAttr::ScalarAttr(NumberAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline ScalarAttr::ScalarAttr(ekl::IndexAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline bool ScalarAttr::classof(Attribute attr)
{
    return llvm::TypeSwitch<Attribute, bool>(attr)
        .Case([](NumberAttr) { return true; })
        .Case([](ekl::IntegerAttr) { return true; })
        .Case([](FloatAttr) { return true; })
        .Case([](ekl::IndexAttr) { return true; })
        .Case([](BoolAttr) { return true; })
        .Default(false);
}

inline ScalarType ScalarAttr::getType() const
{
    return llvm::TypeSwitch<Attribute, ScalarType>(*this)
        .Case([](NumberAttr attr) { return attr.getType(); })
        .Case([](ekl::IntegerAttr attr) {
            return llvm::cast<ekl::IntegerType>(attr.getType());
        })
        .Case([](FloatAttr attr) {
            return llvm::cast<FloatType>(attr.getType());
        })
        .Case([](ekl::IndexAttr attr) { return attr.getType(); })
        .Case([](BoolAttr attr) { return BoolType::get(attr.getContext()); });
}

//===----------------------------------------------------------------------===//
// LiteralAttr implementation
//===----------------------------------------------------------------------===//

inline bool LiteralAttr::classof(ekl::ArrayAttr) { return true; }

inline bool LiteralAttr::classof(InitializerAttr) { return true; }

inline bool LiteralAttr::classof(IdentityAttr) { return true; }

inline bool LiteralAttr::classof(ExtentAttr) { return true; }

inline bool LiteralAttr::classof(Attribute attr)
{
    return llvm::TypeSwitch<Attribute, bool>(attr)
        .Case([](ScalarAttr) { return true; })
        .Case([](ekl::ArrayAttr) { return true; })
        .Case([](InitializerAttr) { return true; })
        .Case([](IdentityAttr) { return true; })
        .Case([](ExtentAttr) { return true; })
        .Default(false);
}

inline LiteralAttr::LiteralAttr(ekl::ArrayAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(InitializerAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(IdentityAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralAttr::LiteralAttr(ExtentAttr attr)
        : Attribute(static_cast<Attribute>(attr).getImpl())
{}

inline LiteralType LiteralAttr::getType() const
{
    return llvm::TypeSwitch<Attribute, LiteralType>(*this)
        .Case([](ScalarAttr attr) { return attr.getType(); })
        .Case([&](StringAttr) { return StringType::get(getContext()); })
        .Case([](ekl::ArrayAttr attr) { return attr.getType(); })
        .Case([](InitializerAttr attr) { return attr.getType(); })
        .Case([](IdentityAttr attr) { return attr.getType(); })
        .Case([](ExtentAttr attr) { return attr.getType(); });
}

} // namespace mlir::ekl
