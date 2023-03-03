/** Provides a mechanism for declaring MLIR concepts.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <type_traits>

namespace mlir::concepts {

// TODO: The actual concepts proposal must use a metaprogramming library to
//       provide both the union_iterator and the UnionConcept types!

/** Integral constant indicating whether @p T is derived from @p From . */
template<class From, class T>
inline constexpr bool is_derived_v =
    std::is_base_of_v<From, T> || std::is_same_v<From, T>;

template<class T>
inline constexpr bool is_attribute_v = is_derived_v<Attribute, T>;
template<class T>
inline constexpr bool is_type_v = is_derived_v<Type, T>;
template<class T>
inline constexpr bool is_value_v = is_derived_v<Value, T>;

struct NoConstraint {
    template<class Arg> static inline bool matches(Arg&&) { return true; }
};

//===----------------------------------------------------------------------===//
// Concept base
//===----------------------------------------------------------------------===//
//
// A concept acts as a typesafe wrapper around a base MLIR smart pointer type to
// check and/or carry information about what it represents.
//

/** Tag type that indicates a concept. */
struct concept_tag {};

/** Integral constant indicating whether @p T is a concept. */
template<class T>
inline constexpr bool is_concept_v = is_derived_v<concept_tag, T>;

namespace detail {

template<class T, class = void>
struct concept_target;
template<class T>
struct concept_target<T, std::enable_if_t<is_attribute_v<T>>> {
    using type = Attribute;
};
template<class T>
struct concept_target<T, std::enable_if_t<is_type_v<T>>> {
    using type = Type;
};
template<class T>
struct concept_target<T, std::enable_if_t<is_value_v<T>>> {
    using type = Value;
};
template<class T>
using concept_target_t = typename concept_target<T>::type;

} // namespace detail

/** Base class for a concept around @p Base .
 *
 * An instance of a concept must always be implicitly convertible to @p Base ,
 * bool (nullity), and default constructible (nullity).
 *
 * @tparam  Base        Base MLIR smart pointer type.
 */
template<class Base>
struct Concept : Base, concept_tag {
    static_assert(
        std::is_default_constructible_v<Base>
        && std::is_constructible_v<bool, Base>
    );

    /** The MLIR target type over which this concept is defined. */
    using target = detail::concept_target_t<Base>;
    /** The base type over which this concept is defined. */
    using base = Base;

    using Base::Base;
};

/** Integral constant indicating whether @p T is a concept of @p Base . */
template<class Base, class T>
inline constexpr bool is_concept_of_v = is_derived_v<Concept<Base>, T>;

//===----------------------------------------------------------------------===//
// Constraint concepts
//===----------------------------------------------------------------------===//
//
// Constraint concepts mirror the declarative style of TableGen Type and Attr
// definitions and extend them to the C++ interface. They are implemented using
// static polymorphism, meaning that constrained instances are not allowed to
// declare additional fields, only methods.
//
// Ideally, these constraints could be generated from the aforementioned
// TableGen records. For now, we import them in TableGen and define them in C++.
//

/** Base class for constraining mlir::Type.
 *
 * @note    Mirrors the TableGen `Type` record class.
 *
 * @tparam  Type        The mlir::Type that is being constrained.
 * @tparam  Constraint  Optional constraint that is being evaluated.
 */
template<class Type, class Constraint = NoConstraint>
struct ConstrainedType : Concept<Type> {
    static_assert(is_type_v<Type>, "`Type` must be derived from `mlir::Type`");

    /** Determines whether @p type matches this concept. */
    static inline bool classof(Type type)
    {
        return Constraint::matches(type);
    }
    /** Determines whether @p type matches this concept. */
    template<
        class Dependent = Type,
        class = std::enable_if_t<!std::is_same_v<Dependent, mlir::Type>>
    >
    static inline bool classof(mlir::Type type)
    {
        return type.isa<Type>() && classof(type.cast<Type>());
    }

    using Concept<Type>::Concept;
};

/** Base class for constraining mlir::Value.
 *
 * @note    Has no direct equivalent in TableGen, since uses of `Type` in the
 *          place of a use for Value automatically materialize as a constraint
 *          on the Value's type.
 *
 * @tparam  Type        The mlir::Type the value is constrained to.
 * @tparam  Constraint  Optional constraint on Value that is being evaluated.
 */
template<class Type, class Constraint = NoConstraint>
struct ConstrainedValue : Concept<Value> {
    static_assert(is_type_v<Type>, "`Type` must be derived from `mlir::Type`");

    /** Determines whether @p value matches this concept. */
    static inline bool classof(Value value)
    {
        return value.getType().isa<Type>() && Constraint::matches(value);
    }

    using Concept<Value>::Concept;

    /** Gets the constrained type. */
    inline Type getType() const
    {
        return Value::getType().template cast<Type>();
    }
};

/** Adaptor for transforming a constant on an attribute value into a constraint
 *  on an mlir::Attribute.
 *
 * @tparam  Attribute   The mlir::Atribute that is being constrained.
 * @tparam  Constraint  Constraint on the attribute value.
 */
template<class Attribute, class Constraint>
struct AttributeValueConstraint {
    static_assert(
        is_attribute_v<Attribute>,
        "`Attribute` must be derived from `mlir::Attribute`"
    );

    /** Determines whether the value of @p attr matches this constraint. */
    static inline bool matches(Attribute attr)
    {
        return Constraint::matches(attr.getValue());
    }
};

/** Base class for constraining mlir::Attribute.
 *
 * @note    Mirrors the TableGen `Attr` record class.
 *
 * @tparam  Attribute   The mlir::Attribute that is being constrained.
 * @tparam  Type        Optional mlir::Type the attribute is constrained to.
 * @tparam  Constraint  Optional constraint on @p Attribute .
 */
template<
    class Attribute,
    class Type = decltype(std::declval<Attribute>().getType()),
    class Constraint = NoConstraint
>
struct ConstrainedAttribute : Concept<Attribute> {
    static_assert(
        is_attribute_v<Attribute>,
        "`Attribute` must be derived from `mlir::Attribute`"
    );

    /** Determines whether @p attr matches this concept. */
    static inline bool classof(Attribute attr)
    {
        if constexpr (is_derived_v<ArrayAttr, Attribute>) {
            return Constraint::matches(attr);
        } else {
            return attr.getType().template isa<Type>() && Constraint::matches(attr);
        }
    }
    /** Determines whether @p attr matches this concept. */
    template<
        class Dependent = Attribute,
        class = std::enable_if_t<!std::is_same_v<Dependent, mlir::Attribute>>
    >
    static inline bool classof(mlir::Attribute attr)
    {
        return attr.isa<Attribute>() && classof(attr.cast<Attribute>());
    }

    using Concept<Attribute>::Concept;

    /** Gets the constrained type. */
    inline Type getType() const
    {
        return Attribute::getType().template cast<Type>();
    }
};

/** Adaptor for transforming an attribute concept into a constraint on an
 *  ArrayAttr.
 *
 * @tparam  Attribute   The attribute concept to constrain all values to.
 */
template<class Attribute>
struct ArrayAttributeConstraint {
    static_assert(
        is_attribute_v<Attribute>,
        "`Attribute` must be derived from `mlir::Attribute`"
    );

    static inline bool matches(ArrayAttr arrayAttr)
    {
        return llvm::all_of(
            arrayAttr,
            [](mlir::Attribute attr) { return attr.isa<Attribute>(); }
        );
    }
};

/** Base class for constraining ArrayAttr.
 *
 * @tparam  Attribute   The attribute concept to constrain all values to.
 */
template<class Attribute>
struct ConstrainedArrayAttribute : ConstrainedAttribute<
    ArrayAttr,
    mlir::Type,
    ArrayAttributeConstraint<Attribute>
> {
    /** Iterator type. */
    using iterator = decltype(
        std::declval<ArrayAttr>().getAsRange<Attribute>().begin()
    );

    /** The constrained array element type. */
    using ElementType = typename Attribute::ValueType;
    /** The constrained value type. */
    using ValueType = decltype(std::declval<ArrayAttr>().getAsRange<Attribute>());

    using ConstrainedAttribute<
        ArrayAttr,
        mlir::Type,
        ArrayAttributeConstraint<Attribute>
    >::ConstrainedAttribute;

    /** Gets the constrained attributes. */
    inline auto getValue() const
    {
        return ArrayAttr::getAsRange<Attribute>();
    }

    /** Gets the underlying attribute values as a range. */
    template<class AttTy = Attribute, class ValueTy = typename AttTy::ValueType>
    inline auto getAsValueRange() const
    {
        return ArrayAttr::getAsValueRange<AttTy, ValueTy>();
    }

    /** Gets the constrained range start iterator. */
    inline iterator begin() const
    {
        return getValue().begin();
    }
    /** Gets the constrained range end iterator. */
    inline iterator end() const
    {
        return getValue().end();
    }

    /** Copies the constrained attribute values to @p result . */
    inline void getValues(SmallVectorImpl<ElementType> &result) const
    {
        result.clear();
        result.reserve(ArrayAttr::size());
        llvm::copy(getAsValueRange(), std::back_inserter(result));
    }
    /** Creates a copy of the constrained attribute values. */
    inline SmallVector<ElementType> getValues() const
    {
        SmallVector<ElementType> result;
        getValues(result);
        return result;
    }
};

} // namespace mlir::concepts
