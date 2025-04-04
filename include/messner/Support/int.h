/// Abstractions for the built-in integral types.
///
/// We consider all built-in integral types, as well as enumerations, to be
/// integer-like. For integer-like types, we implement helper functions to
/// strengthen program semantics.
///
/// Generally, integer operations should only be performed on values of the same
/// type, due to C++ integer promotion and implicit conversion footguns. To make
/// all the necessary transitions safer and less error-prone, we provide some
/// more explicit operations.
///
/// Explicit casts:
///
///   - `as_int` helps with polymorphic functions on integers and enums.
///   - `as_uint` and `as_sint` reinterpret bit sequences.
///   - `trunc` performs bit truncation only.
///   - `zext` and `zext_or_trunc` perform unsigned-stylecasts.
///   - `sext` and `sext_or_trunc` perform signed-style casts.
///   - `ext` performs integer promotions.
///   - `value_cast` performs runtime checked value-preserving casts.
///
/// Arithmetic operations:
///
///   - `cmp` performs true comparison.
///
/// Logical operations:
///
///   - `test_any` tests for the presence of any masked bit.
///   - `test_all` tests for the presence of all masked bits.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Support/arith_result.h"

#include <array>
#include <bit>
#include <compare>
#include <concepts>
#include <limits>
#include <type_traits>

namespace messner {

/// Concept that matches any (cv-qualified) built-in integral or enum type.
template<class T>
concept int_or_enum =
    std::is_integral_v<T> || std::is_enum_v<std::remove_cv_t<T>>;

/// Type trait that determines the underlying built-in integral type of @p T .
///
/// Specializations must define the following members:
///
/// ```c++
/// using type = /* integral type */;
/// ```
template<class T>
struct make_integral;

/// Obtains the underlying built-in integral type of @p T .
template<int_or_enum T>
using make_integral_t = make_integral<T>::type;

// These all work for int_or_enum, so they are perfect for us already.
using std::make_signed;
using std::make_signed_t;
using std::make_unsigned;
using std::make_unsigned_t;

// The bit reinterpretation cast.
using std::bit_cast;

/// Obtains the integer value of @p value .
///
/// Given that @p value has an underlying integer representation, this function
/// returns it. The resulting type is a built-in integral type that matches the
/// semantical signedness of @p value .
///
/// @post   `std::is_eq(cmp(value, result))`
[[nodiscard]]
constexpr auto as_int(int_or_enum auto value)
    -> make_integral_t<decltype(value)>;

/// Interprets the bit sequence @p value as an unsigned integer.
///
/// Given that @p value has an underlying integer representation, there exists
/// some unsigned built-in integral type with the same bit width. This function
/// performs a bit_cast to that type.
[[nodiscard]]
constexpr auto as_uint(int_or_enum auto value)
    -> make_unsigned_t<decltype(value)>;

/// Interprets the bit sequence @p value as a signed integer.
///
/// Given that @p value has an underlying integer representation, there exists
/// some signed built-in integral type with the same bit width. This function
/// performs a bit_cast to that type.
[[nodiscard]]
constexpr auto as_sint(int_or_enum auto value)
    -> make_signed_t<decltype(value)>;

/// Extends @p value to type @p T by prepending zeros.
///
/// This function performs a zero-extending cast regardless of the signedness
/// of the involved types, ensured via bit casting. Note that a zero-extending
/// cast is not value-preserving for all signed integers.
///
/// @pre    @p T is not narrower than @p value .
template<int_or_enum T>
[[nodiscard]]
constexpr auto zext(int_or_enum auto value) -> T;

/// Extends @p value to type @p T by repeating the most significant (sign) bit.
///
/// This function performs a sign-extending cast regardless of the signedness
/// of the involved types, ensured via bit casting. Note that a sign-extending
/// cast is not value-preserving for all unsigned integers.
///
/// @pre    @p T is not narrower than @p value .
template<int_or_enum T>
[[nodiscard]]
constexpr auto sext(int_or_enum auto value) -> T;

/// Shortens @p value to type @p T by truncating the most significant bits.
///
/// Note that a truncating cast is not value-preserving for all integers.
///
/// @pre    @p T is not wider than @p value .
template<int_or_enum T>
[[nodiscard]]
constexpr auto trunc(int_or_enum auto value) -> T;

/// Extends or shortens @p value to type @p T using zero-extension.
///
/// This is equivalent to the semantics of `static_cast` on unsigned integers,
/// but does this regardless of the involved types.
template<int_or_enum T>
[[nodiscard]]
constexpr auto zext_or_trunc(int_or_enum auto value) -> T;

/// Extends or shortens @p value to type @p T using sign-extension.
///
/// This is equivalent to the semantics of `static_cast` on signed integers in
/// C++20, but does this regardless of the involved types.
template<int_or_enum T>
[[nodiscard]]
constexpr auto sext_or_trunc(int_or_enum auto value) -> T;

/// Type trait that determines whether @p T integer promotes to @p Super .
///
/// Specializations must define the following members:
///
/// ```c++
/// static constexpr auto value = /* subset relation */;
/// using type = Super;
/// constexpr auto operator()(T value) const -> type;
/// ```
template<class T, class Super>
struct promote_to_int;

/// Determines whether @p T integer promotes to @p Super .
template<int_or_enum T, int_or_enum Super>
static constexpr auto promote_to_int_v = promote_to_int<T, Super>::value;

/// Concept that matches any integral type that promotes to @p Super .
template<class T, class Super>
concept promotes_to_int =
    int_or_enum<T> && int_or_enum<Super> && promote_to_int_v<T, Super>;

/// Type trait that determines the result of promoting integer @p Ts .
///
/// Specializations must define the following members:
///
/// ```c++
/// static constexpr auto value = /* legality */;
/// using type = /* promoted type */;
/// constexpr auto operator()(/* any of Ts */ value) const -> type;
/// ```
template<class... Ts>
struct int_promotion;

/// Determines whether @p Ts can promote to a common integral supertype.
template<int_or_enum... Ts>
static constexpr auto int_promotion_v = int_promotion<Ts...>::value;

/// Obtains the common integral supertype of @p Ts .
///
/// @pre    `int_promotion_v<Ts...>`
template<int_or_enum... Ts>
using int_promotion_t = int_promotion<Ts...>::type;

namespace detail {

template<class T>
struct ext_impl;

} // namespace detail

/// Performs controlled integer promotion.
///
/// Given that @p T is a known supertype of all the argument integers, performs
/// a value-preserving cast on all arguments to @p T . If only one argument is
/// specified, the result value is returned. If more than one argument is given,
/// an std::array of the results is returned (which allows structured binding).
///
/// If @p T is @c void , it is inferred as the argument type that is the common
/// supertype of all specified arguments, if it exists. In that case, at least
/// 2 arguments must be specified.
///
/// @pre    `std::is_void_v<T> || promotes_to_int<decltype(value), T>`
/// @pre    `!std::is_void_v<T> || int_promotion_v<decltype(values)...>`
///
/// @post   `std::is_eq(cmp(value, result))`
template<class T = void>
static constexpr detail::ext_impl<T> ext{};

/// Performs a checked value-preserving cast to @p T .
///
/// Given some integer @p value and a target type @p T , this function will
/// produce a representation of @p value in @p T and indicate the quality of
/// this result. The possible result qualities are:
///
///   - arith_quality::exact, when @p value fits in @p T .
///   - arith_quality::underflow, when @p value is less than the min of @p T .
///   - arith_quality::overflow, when @p value is larger than the max of @p T .
///
/// In all cases, the result is defined as being the same as the equivalent C++
/// `static_cast`. When @p T is a known supertype of @p value , the result is
/// always `arith_quality::exact`.
template<int_or_enum T>
[[nodiscard]]
constexpr auto value_cast(int_or_enum auto value) -> arith_result<T>;

/// Performs true comparison between @p lhs and @p rhs .
///
/// Given two integers @p lhs and @p rhs , this function compares their
/// contained values mathematically. This means that values of types that don't
/// promote can be compared safely by this function.
[[nodiscard]]
constexpr auto cmp(int_or_enum auto lhs, int_or_enum auto rhs)
    -> std::strong_ordering;

/// Determines whether any bits in @p mask are set in @p value .
template<int_or_enum Int>
[[nodiscard]]
constexpr auto test_any(Int value, Int mask) -> bool;

/// Determines whether all bits in @p mask are set in @p value .
template<int_or_enum Int>
[[nodiscard]]
constexpr auto test_all(Int value, Int mask) -> bool;

} // namespace messner

namespace messner {

//===----------------------------------------------------------------------===//
// make_integral
//===----------------------------------------------------------------------===//

template<class T>
struct make_integral {};

template<std::integral Int>
struct make_integral<Int> {
    using type = Int;
};

template<class Enum>
    requires(std::is_enum_v<std::remove_cv_t<Enum>>)
struct make_integral<Enum> {
    using type = std::underlying_type_t<Enum>;
};

//===----------------------------------------------------------------------===//
// as_int
//===----------------------------------------------------------------------===//

constexpr auto as_int(int_or_enum auto value)
    -> make_integral_t<decltype(value)>
{
    return bit_cast<make_integral_t<decltype(value)>>(value);
}

//===----------------------------------------------------------------------===//
// as_uint
//===----------------------------------------------------------------------===//

constexpr auto as_uint(int_or_enum auto value)
    -> make_unsigned_t<decltype(value)>
{
    return bit_cast<make_unsigned_t<decltype(value)>>(value);
}

//===----------------------------------------------------------------------===//
// as_sint
//===----------------------------------------------------------------------===//

constexpr auto as_sint(int_or_enum auto value) -> make_signed_t<decltype(value)>
{
    return bit_cast<make_signed_t<decltype(value)>>(value);
}

//===----------------------------------------------------------------------===//
// zext
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto zext(int_or_enum auto value) -> T
{
    // Interpret the input value as an unsigned integer.
    const auto source = as_uint(value);

    // Determine the legality of this cast.
    using Source              = decltype(source);
    constexpr auto sourceBits = std::numeric_limits<Source>::digits;
    using Target              = make_unsigned_t<T>;
    constexpr auto targetBits = std::numeric_limits<Target>::digits;
    static_assert(sourceBits <= targetBits, "zext does not extend");

    // Use static_cast to perform the zero-extension.
    const auto result = static_cast<Target>(source);
    return bit_cast<T>(result);
}

//===----------------------------------------------------------------------===//
// sext
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto sext(int_or_enum auto value) -> T
{
    // Interpret the input value as a signed integer.
    const auto source = as_sint(value);

    // Determine the legality of this cast.
    using Source              = decltype(source);
    constexpr auto sourceBits = std::numeric_limits<Source>::digits;
    using Target              = make_signed_t<T>;
    constexpr auto targetBits = std::numeric_limits<Target>::digits;
    static_assert(sourceBits <= targetBits, "sext does not extend");

    // Use static_cast to perform the sign-extension.
    const auto result = static_cast<Target>(source);
    return bit_cast<T>(result);
}

//===----------------------------------------------------------------------===//
// trunc
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto trunc(int_or_enum auto value) -> T
{
    // Interpret the input value as an integer. The signedness does not matter,
    // since static_cast will always truncate if the invariants below are met.
    const auto source = as_int(value);

    // Determine the legality of this cast.
    using Source              = decltype(source);
    constexpr auto sourceBits = std::numeric_limits<Source>::digits;
    using Target              = make_integral_t<T>;
    constexpr auto targetBits = std::numeric_limits<Target>::digits;
    static_assert(sourceBits >= targetBits, "trunc does not truncate");

    // Use static_cast to perform the truncation.
    const auto result = static_cast<Target>(source);
    return bit_cast<T>(result);
}

//===----------------------------------------------------------------------===//
// zext_or_trunc
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto zext_or_trunc(int_or_enum auto value) -> T
{
    // Interpret the input value as an unsigned integer.
    const auto source = as_uint(value);
    // Use static_cast to perform zero-extension or truncation.
    const auto result = static_cast<make_unsigned_t<T>>(source);
    return bit_cast<T>(result);
}

//===----------------------------------------------------------------------===//
// sext_or_trunc
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto sext_or_trunc(int_or_enum auto value) -> T
{
    // Interpret the input value as a signed integer.
    const auto source = as_sint(value);
    // Use static_cast to perform sign-extension or truncation.
    const auto result = static_cast<make_signed<T>>(source);
    return bit_cast<T>(result);
}

//===----------------------------------------------------------------------===//
// promote_to_int
//===----------------------------------------------------------------------===//

namespace detail {

template<std::integral Subset, std::integral Superset>
[[nodiscard]]
consteval auto is_int_subset_eq() -> bool
{
    using SubsetLimits   = std::numeric_limits<Subset>;
    using SupersetLimits = std::numeric_limits<Superset>;

    auto needed_bits = SubsetLimits::digits;
    if (SubsetLimits::is_signed ^ SupersetLimits::is_signed) {
        if (!SupersetLimits::is_signed) {
            // Signed integers are never a subset of unsigned integers.
            return false;
        }

        // Unsigned integers may require an additional bit in two's complement
        // representation.
        ++needed_bits;
    }

    // Check if there are always enough bits to accomodate the subset values.
    return SupersetLimits::digits >= needed_bits;
}

} // namespace detail

template<class T, class Super>
struct promote_to_int {
    static constexpr auto value = false;
};

template<int_or_enum T, int_or_enum Super>
    requires(
        detail::is_int_subset_eq<make_integral_t<T>, make_integral_t<Super>>())
struct promote_to_int<T, Super> {
    static constexpr auto value = true;

    using type = Super;

    [[nodiscard]]
    constexpr auto operator()(T value) const -> type
    {
        return bit_cast<Super>(static_cast<make_integral_t<Super>>(
            bit_cast<make_integral_t<T>>(value)));
    }
};

//===----------------------------------------------------------------------===//
// int_promotion
//===----------------------------------------------------------------------===//

namespace detail {

template<class Guess, class... Ts>
struct find_max_int {};

template<class Guess>
struct find_max_int<Guess> {
    using type = Guess;
};

template<class Guess, int_or_enum Rhs>
[[nodiscard]]
consteval auto is_larger_int() -> bool
{
    if constexpr (std::is_void_v<Guess>) {
        return true;
    } else {
        using GuessLimits = std::numeric_limits<make_integral_t<Guess>>;
        using RhsLimits   = std::numeric_limits<make_integral_t<Rhs>>;

        return GuessLimits::digits < RhsLimits::digits
            || (!GuessLimits::is_signed && RhsLimits::is_signed);
    }
}

template<class Guess, class Head, class... Tail>
struct find_max_int<Guess, Head, Tail...>
        : find_max_int<
              std::conditional_t<is_larger_int<Guess, Head>(), Head, Guess>,
              Tail...> {};

template<class... Ts>
using find_max_int_t = find_max_int<void, Ts...>::type;

} // namespace detail

template<class... Ts>
struct int_promotion {
    static constexpr auto value = false;
};

template<int_or_enum... Ts>
    requires(promote_to_int_v<Ts, detail::find_max_int_t<Ts...>> && ...)
struct int_promotion<Ts...> {
    static constexpr auto value = true;

    using type = detail::find_max_int_t<Ts...>;

    [[nodiscard]]
    constexpr auto operator()(auto value) const -> type
    {
        return promote_to_int<decltype(value), type>{}(value);
    }
};

//===----------------------------------------------------------------------===//
// ext
//===----------------------------------------------------------------------===//

template<int_or_enum T>
struct detail::ext_impl<T> {
    [[nodiscard]]
    constexpr auto operator()(int_or_enum auto value) const -> T
    {
        static_assert(
            promotes_to_int<decltype(value), T>,
            "ext does not promote");
        return promote_to_int<decltype(value), T>{}(value);
    }

    [[nodiscard]]
    constexpr auto operator()(int_or_enum auto... values) const
        -> std::array<T, sizeof...(values)>
        requires(sizeof...(values) > 1)
    {
        return {operator()(values)...};
    }
};

template<>
struct detail::ext_impl<void> {
    [[nodiscard]]
    constexpr auto operator()(int_or_enum auto... values) const
        requires(sizeof...(values) >= 2)
    {
        using promotion = int_promotion<decltype(values)...>;
        static_assert(promotion::value, "cannot infer common supertype");
        return std::array{promotion{}(values)...};
    }
};

//===----------------------------------------------------------------------===//
// value_cast
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto value_cast(int_or_enum auto value) -> arith_result<T>
{
    using promotion = promote_to_int<decltype(value), T>;
    if constexpr (promotion::value) {
        // Known subsets don't need to go through the runtime checking.
        return promotion{}(value);
    } else {
        // Since the subset check has failed, this must be a narrowing cast.
        // Interpret the input value as an integer.
        const auto source         = as_int(value);
        using Source              = decltype(source);
        using Target              = make_integral_t<T>;
        constexpr auto target_min = std::numeric_limits<Target>::min();
        constexpr auto target_max = std::numeric_limits<Target>::max();

        // Perform the cast, which may not be value preserving as of yet.
        const auto result = bit_cast<T>(static_cast<Target>(source));

        // Determine if any overflow occured.
        if constexpr (!std::numeric_limits<Source>::is_signed) {
            // Since the source is unsigned, and Target is not a subset of
            // Source, this must be a narrowing cast. In particular, the maximum
            // value of Target must fit in Source, and is the only bound that
            // this cast could potentially violate.

            if (source > static_cast<Source>(target_max))
                return {result, arith_quality::overflow};
        } else if constexpr (!std::numeric_limits<Target>::is_signed) {
            // We're casting from a signed to an unsigned type, which is never
            // a subset of one another. Thus, we may also be in a widening cast.
            // First, we test against the lower bound of Target, since that is
            // always representible in Source.

            if (source < Source{0}) return {result, arith_quality::underflow};

            if constexpr (
                std::numeric_limits<Target>::digits
                < std::numeric_limits<Source>::digits) {
                // This is a narrowing cast, which means the upper bound of
                // Target must fit within Source, and is the only bound that
                // could still be violated.

                if (source > static_cast<Source>(target_max))
                    return {result, arith_quality::overflow};
            }
        } else {
            // Since both types are signed, and Target is not a subset of
            // Source, this must be a narrowing cast. In particular, the bounds
            // of Target must fit in Source.

            if (source > ext<Source>(target_max))
                return {result, arith_quality::overflow};
            if (source < ext<Source>(target_min))
                return {result, arith_quality::underflow};
        }

        // The result is known-exact.
        return result;
    }
}

//===----------------------------------------------------------------------===//
// cmp
//===----------------------------------------------------------------------===//

constexpr auto cmp(int_or_enum auto lhs, int_or_enum auto rhs)
    -> std::strong_ordering
{
    // Cast both operands to the same type with a value preserving cast.
    const auto cast = value_cast<decltype(lhs)>(rhs);

    // If the cast failed, the failure implies the comparison result.
    if (cast == arith_quality::underflow) return std::strong_ordering::greater;
    if (cast == arith_quality::overflow) return std::strong_ordering::less;

    // Otherwise, use the built-in comparison operator.
    return lhs <=> cast.value();
}

//===----------------------------------------------------------------------===//
// test_any
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto test_any(T value, T mask) -> bool
{
    const auto mask_bits = as_uint(mask);
    return (as_uint(value) & mask_bits) != decltype(mask_bits){};
}

//===----------------------------------------------------------------------===//
// test_all
//===----------------------------------------------------------------------===//

template<int_or_enum T>
constexpr auto test_all(T value, T mask) -> bool
{
    const auto mask_bits = as_uint(mask);
    return (as_uint(value) & mask_bits) == mask_bits;
}

} // namespace messner
