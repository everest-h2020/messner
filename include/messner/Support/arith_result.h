/// Implements the arith_result strong wrapper type.
///
/// Results of arithmetic operations should always indicate their quality, i.e.,
/// how the returned value relates to the mathematically exact result. The
/// types defined in this file provide a unified way to achieve this.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <bit>
#include <cassert>
#include <compare>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace messner {

/// Indicates the quality of an arithmetic result.
///
/// The quality of an arihmetic result is defined by its relation to the
/// expected mathematically exact value.
enum class arith_quality {
    /// The result is undefined.
    undefined = 0b000,

    /// The exact value is below the range of representable results.
    underflow = 0b001,
    /// The exact value is above the range of representable results.
    overflow  = 0b010,
    /// The exact value is within range, but can't be represented exactly.
    ///
    /// The result value can further be constrained by the rounding mode that
    /// applied to the operation. The rounding mode indicates which possible
    /// candidates from the representable range can be selected as the result.
    rounded   = 0b011,

    /// The result is mathematically exact.
    exact = 0b100
};

/// Determines whether @p quality is not mathematically exact.
///
/// The @p quality is inexact if it is defined but not exact.
[[nodiscard]]
constexpr auto is_inexact(arith_quality quality) -> bool;

namespace detail {

class arith_result_base {
public:
    /// Obtains the arith_quality of this result.
    [[nodiscard]]
    constexpr auto quality() const -> arith_quality;

    /// @copydoc quality()
    /*implicit*/ constexpr operator arith_quality() const;

    [[nodiscard]]
    constexpr auto has_value() const -> bool;

    /// @copydoc has_value()
    explicit constexpr operator bool() const;

    /// Resets this result to the undefined state.
    constexpr void reset();

protected:
    /*implicit*/ constexpr arith_result_base(
        arith_quality quality = arith_quality::undefined);

    arith_quality _quality;
};

template<class T>
class arith_result_storage : public arith_result_base {
public:
    /// Destroys the contained value, if any.
    constexpr ~arith_result_storage();

    /// Resets this result to the undefined state, destroying the value, if any.
    constexpr void reset();

protected:
    /*implicit*/ constexpr arith_result_storage() = default;

    explicit constexpr arith_result_storage(
        arith_quality quality,
        auto &&...args);

    union {
        char _undefined;
        T _defined;
    };
};

template<class T>
    requires(std::is_trivially_destructible_v<T>)
class arith_result_storage<T> : public arith_result_base {
public:
    constexpr ~arith_result_storage() = default;

protected:
    /*implicit*/ constexpr arith_result_storage() : _undefined{} {}

    explicit constexpr arith_result_storage(
        arith_quality quality,
        auto &&...args);

    union {
        char _undefined;
        T _defined;
    };
};

} // namespace detail

/// Holds an arithmetic operation result of type @p T and its arith_quality.
///
/// This type is a mix between a pair of @p T and arith_quality and an
/// std::optional. If the arith_quality is undefined, no value will be stored
/// inside. Otherwise, a value is stored alongside the quality. A simplified
/// optional-style interface is provided on this class.
///
/// Changing the stored value in-place without changing the arith_quality is
/// not allowed, and therefore no mutable reference can be obtained to the
/// value. Moving the stored value out of the result is allowed, however.
///
/// @tparam T   Value type.
template<class T>
class [[nodiscard]] arith_result : public detail::arith_result_storage<T> {
    static_assert(std::is_same_v<T, std::remove_cv_t<T>>);

public:
    /// The stored value type.
    using value_type = T;

    /// Initializes an undefined arith_result.
    ///
    /// @post   `!has_value()`
    /*implicit*/ constexpr arith_result(std::nullopt_t = std::nullopt);

    /// Initializes an arith_result with @p value .
    ///
    /// If @p quality is arith_result::undefined, no value will be constructed.
    ///
    /// @param              value   Value.
    /// @param              quality arith_quality.
    ///
    /// @pre    @p T is constructible from @p value .
    template<class U = T>
        requires(std::same_as<std::remove_cvref_t<U>, T>)
    /*implicit*/ constexpr arith_result(
        U &&value,
        arith_quality quality = arith_quality::exact);

    /// Initializes an arith_result of @p quality via in-place construction.
    ///
    /// If @p quality is arith_result::undefined, no value will be constructed.
    ///
    /// @param              quality arith_quality.
    /// @param              args    Constructor arguments.
    ///
    /// @pre    @p T is constructible from @p args .
    explicit constexpr arith_result(arith_quality quality, auto &&...args);

    /*implicit*/ constexpr arith_result(arith_result &&move);
    /*implicit*/ constexpr arith_result(const arith_result &copy);
    constexpr auto operator=(arith_result &move) -> arith_result &;
    constexpr auto operator=(const arith_result &copy) -> arith_result &;

    constexpr ~arith_result() = default;

    /// Obtains the contained value, if any.
    ///
    /// @pre    `has_value()`
    [[nodiscard]]
    constexpr auto value() const & -> const T &;
    /// @copydoc value()
    [[nodiscard]]
    constexpr auto value() const && -> const T &&;
    /// @copydoc value()
    [[nodiscard]]
    constexpr auto value() && -> T &&;

    /// Obtains the contained value, or constructs a default if undefined.
    ///
    /// @pre    @p T is constructible from @p default_value .
    template<class U>
    [[nodiscard]]
    constexpr auto value_or(U &&default_value) const & -> T;
    /// @copydoc value_or()
    template<class U>
    [[nodiscard]]
    constexpr auto value_or(U &&default_value) && -> T;

    /// Obtains the contained value if it is exact, or std::nullopt.
    constexpr auto exact() const & -> std::optional<T>;
    /// @copydoc exact()
    constexpr auto exact() && -> std::optional<T>;

    /// Overwrites this result via in-place construction.
    ///
    /// If @p quality is arith_result::undefined, no value will be constructed.
    ///
    /// @param              quality arith_quality.
    /// @param              args    Constructor arguments.
    ///
    /// @pre    @p T is constructible from @p args .
    constexpr void emplace(arith_quality quality, auto &&...args);

    /// Obtains a pointer to the contained exact value.
    ///
    /// @pre    `*this == arith_quality::exact`
    constexpr auto operator->() const -> const T *;

    /// Obtains the contained exact value.
    ///
    /// @pre    `*this == arith_quality::exact`
    constexpr auto operator*() const & -> const T &;
    /// @copydoc operator*()
    constexpr auto operator*() const && -> const T &&;
    /// @copydoc operator*()
    constexpr auto operator*() && -> T &&;

    /// Determines whether two arith_result instances are indentical.
    [[nodiscard]]
    constexpr auto operator==(const arith_result &rhs) const -> bool;

    /// Determines whether @p rhs is contained with the result.
    [[nodiscard]]
    constexpr auto operator==(const T &rhs) const -> bool;
    /// Compares the contained value with @p rhs , if any.
    [[nodiscard]]
    constexpr auto operator<=>(const T &rhs) const -> std::partial_ordering;

    /// Swaps the qualities and results of two arith_result instances.
    template<class U>
    friend constexpr void swap(arith_result<U> &lhs, arith_result<U> &rhs);
    /// Computes a hash value for an arith_result.
    friend struct ::std::hash<arith_result>;
};

} // namespace messner

namespace messner {

//===----------------------------------------------------------------------===//
// is_inexact
//===----------------------------------------------------------------------===//

[[nodiscard]]
constexpr auto is_inexact(arith_quality quality) -> bool
{
    using bit_type = std::underlying_type_t<arith_quality>;

    // return test_any(quality, arith_quality::rounded);
    constexpr auto mask = std::bit_cast<bit_type>(arith_quality::rounded);
    const auto bits     = std::bit_cast<bit_type>(quality);
    return (bits & mask) > 0;
}

//===----------------------------------------------------------------------===//
// detail::arith_result_base
//===----------------------------------------------------------------------===//

constexpr auto detail::arith_result_base::quality() const -> arith_quality
{
    return _quality;
}

constexpr detail::arith_result_base::operator arith_quality() const
{
    return _quality;
}

constexpr auto detail::arith_result_base::has_value() const -> bool
{
    return _quality != arith_quality::undefined;
}

constexpr detail::arith_result_base::operator bool() const
{
    return has_value();
}

constexpr void detail::arith_result_base::reset()
{
    _quality = arith_quality::undefined;
}

constexpr detail::arith_result_base::arith_result_base(arith_quality quality)
        : _quality(quality)
{}

//===----------------------------------------------------------------------===//
// detail::arith_result_storage
//===----------------------------------------------------------------------===//

template<class T>
constexpr detail::arith_result_storage<T>::~arith_result_storage()
{
    if (!has_value()) return;

    _defined.~T();
}

template<class T>
constexpr void detail::arith_result_storage<T>::reset()
{
    if (!has_value()) return;

    _defined.~T();
    detail::arith_result_base::reset();
}

template<class T>
constexpr detail::arith_result_storage<T>::arith_result_storage(
    arith_quality quality,
    auto &&...args)
        : detail::arith_result_base(quality)
{
    if (quality == arith_quality::undefined) return;

    std::construct_at(&_defined, std::forward<decltype(args)>(args)...);
}

template<class T>
    requires(std::is_trivially_destructible_v<T>)
constexpr detail::arith_result_storage<T>::arith_result_storage(
    arith_quality quality,
    auto &&...args)
        : detail::arith_result_base(quality)
{
    if (quality == arith_quality::undefined) return;

    std::construct_at(&_defined, std::forward<decltype(args)>(args)...);
}

//===----------------------------------------------------------------------===//
// arith_result
//===----------------------------------------------------------------------===//

template<class T>
constexpr arith_result<T>::arith_result(std::nullopt_t)
        : detail::arith_result_storage<T>()
{}

template<class T>
constexpr arith_result<T>::arith_result(arith_result &&move)
        : arith_result(move._quality, std::move(move._defined))
{
    move.reset();
}

template<class T>
constexpr arith_result<T>::arith_result(const arith_result &copy)
        : arith_result(copy._quality, copy._defined)
{}

template<class T>
constexpr auto arith_result<T>::operator=(arith_result &move) -> arith_result &
{
    this->emplace(move._quality, std::move(move._defined));
    return *this;
}

template<class T>
constexpr auto arith_result<T>::operator=(const arith_result &copy)
    -> arith_result &
{
    this->emplace(copy._quality, copy._defined);
    return *this;
}

template<class T>
template<class U>
    requires(std::same_as<std::remove_cvref_t<U>, T>)
constexpr arith_result<T>::arith_result(U &&value, arith_quality quality)
        : detail::arith_result_storage<T>(quality, std::forward<U>(value))
{}

template<class T>
constexpr arith_result<T>::arith_result(arith_quality quality, auto &&...args)
        : detail::arith_result_storage<T>(
              quality,
              std::forward<decltype(args)>(args)...)
{}

template<class T>
constexpr auto arith_result<T>::value() const & -> const T &
{
    assert(this->has_value());
    return this->_defined;
}

template<class T>
constexpr auto arith_result<T>::value() const && -> const T &&
{
    assert(this->has_value());
    return std::move(this->_defined);
}

template<class T>
constexpr auto arith_result<T>::value() && -> T &&
{
    assert(this->has_value());
    return std::move(this->_defined);
}

template<class T>
template<class U>
constexpr auto arith_result<T>::value_or(U &&default_value) const & -> T
{
    return this->has_value() ? this->_defined
                             : static_cast<T>(std::forward<U>(default_value));
}

template<class T>
template<class U>
constexpr auto arith_result<T>::value_or(U &&default_value) && -> T
{
    return this->has_value() ? std::move(this->_defined)
                             : static_cast<T>(std::forward<U>(default_value));
}

template<class T>
constexpr auto arith_result<T>::exact() const & -> std::optional<T>
{
    if (*this != arith_quality::exact) return std::nullopt;
    return std::optional<T>(std::in_place, this->value());
}

template<class T>
constexpr auto arith_result<T>::exact() && -> std::optional<T>
{
    if (*this != arith_quality::exact) return std::nullopt;
    return std::optional<T>(std::in_place, std::move(*this).value());
}

template<class T>
constexpr auto arith_result<T>::operator->() const -> const T *
{
    assert(*this == arith_quality::exact);
    return &this->_defined;
}

template<class T>
constexpr auto arith_result<T>::operator*() const & -> const T &
{
    assert(*this == arith_quality::exact);
    return this->_defined;
}

template<class T>
constexpr auto arith_result<T>::operator*() const && -> const T &&
{
    assert(*this == arith_quality::exact);
    return std::move(this->_defined);
}

template<class T>
constexpr auto arith_result<T>::operator*() && -> T &&
{
    assert(*this == arith_quality::exact);
    return std::move(this->_defined);
}

template<class T>
constexpr auto arith_result<T>::operator==(const arith_result &rhs) const
    -> bool
{
    if (this->_quality != rhs._quality) return false;
    if (!this->has_value()) return true;
    return this->_defined == rhs._defined;
}

template<class T>
constexpr auto arith_result<T>::operator==(const T &rhs) const -> bool
{
    if (!this->has_value()) return false;
    return this->_defined == rhs;
}

template<class T>
constexpr auto arith_result<T>::operator<=>(const T &rhs) const
    -> std::partial_ordering
{
    if (!this->has_value()) return std::partial_ordering::unordered;
    return this->_defined <=> rhs;
}

template<class T>
constexpr void arith_result<T>::emplace(arith_quality quality, auto &&...args)
{
    this->reset();
    if ((this->_quality = quality) == arith_quality::undefined) return;

    std::construct_at(&this->_defined, std::forward<decltype(args)>(args)...);
}

template<class T>
constexpr void swap(arith_result<T> &lhs, arith_result<T> &rhs)
{
    using std::swap;

    if (lhs.has_value() && rhs.has_value()) {
        // Swap the contained values in-place.
        swap(lhs._defined, rhs._defined);
        swap(lhs._quality, rhs._quality);
    } else if (lhs.has_value()) {
        // Move the value from lhs to rhs.
        rhs.emplace(lhs._quality, std::move(lhs._defined));
        lhs.reset();
    } else if (rhs.has_value()) {
        // Move the value from rhs to lhs.
        lhs.emplace(rhs._quality, std::move(rhs._defined));
        rhs.reset();
    } else {
        // Both are undefined, nothing needs to happen.
    }
}

} // namespace messner

namespace std {

template<class T>
    requires(requires(const T &value) { hash<T>{}(value); })
struct hash<::messner::arith_result<T>> {
    [[nodiscard]]
    constexpr auto operator()(const ::messner::arith_result<T> &value) const
        -> size_t
    {
        if (!value.has_value()) return {};
        return hash<T>{}(value._defined);
    }
};

} // namespace std
