/// Mixin for implementing types with fallible arithmetic.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Support/optional.h"

#include <compare>
#include <utility>

namespace messner::mixin {

/// Implements fallible arithmetic operations via std::optional.
template<class Derived>
struct fallible_arith {
    /*implicit*/ constexpr fallible_arith() = default;

#define BINOP(op)                                                              \
    template<any_optional Rhs>                                                 \
    friend constexpr auto operator op(const Derived &lhs, const Rhs &rhs)      \
        ->ensure_optional_t<decltype(lhs op rhs.value())>                      \
    {                                                                          \
        if (!rhs.has_value()) return std::nullopt;                             \
        return lhs op rhs.value();                                             \
    }                                                                          \
    template<any_optional Lhs>                                                 \
    friend constexpr auto operator op(const Lhs &lhs, const Derived &rhs)      \
        ->ensure_optional_t<decltype(lhs.value() op rhs)>                      \
    {                                                                          \
        if (!lhs.has_value()) return std::nullopt;                             \
        return lhs.value() op rhs;                                             \
    }                                                                          \
    template<class Rhs, optional_of<Derived> Lhs>                              \
    friend constexpr auto operator op(const Lhs &lhs, Rhs &&rhs)               \
        ->ensure_optional_t<decltype(lhs.value() op std::forward<Rhs>(rhs))>   \
    {                                                                          \
        if (!lhs.has_value()) return std::nullopt;                             \
        return lhs.value() op std::forward<Rhs>(rhs);                          \
    }                                                                          \
    template<class Lhs, optional_of<Derived> Rhs>                              \
    friend constexpr auto operator op(Lhs &&lhs, const Rhs &rhs)               \
        ->ensure_optional_t<decltype(std::forward<Lhs>(lhs) op rhs.value())>   \
    {                                                                          \
        if (!rhs.has_value()) return std::nullopt;                             \
        return std::forward<Lhs>(lhs) op rhs.value();                          \
    }

    BINOP(+)
    BINOP(-)
    BINOP(*)
    BINOP(/)
    BINOP(%)

#undef BINOP

    template<optional_of<Derived> Self>
    friend constexpr auto operator+(const Self &self)
        -> ensure_optional_t<decltype(+self.value())>
    {
        if (!self.has_value()) return std::nullopt;
        return +self.value();
    }
    template<optional_of<Derived> Self>
    friend constexpr auto operator-(const Self &self)
        -> ensure_optional_t<decltype(-self.value())>
    {
        if (!self.has_value()) return std::nullopt;
        return -self.value();
    }

    constexpr auto operator==(const fallible_arith &) const -> bool = default;
    constexpr auto operator<=>(const fallible_arith &) const
        -> std::strong_ordering = default;

    template<any_optional Rhs>
        requires(requires(const Derived &lhs, const Rhs &rhs) {
            lhs <=> rhs.value();
        })
    friend constexpr auto operator<=>(const Derived &lhs, const Rhs &rhs)
        -> std::partial_ordering
    {
        if (!rhs.has_value()) return std::partial_ordering::unordered;
        return lhs <=> rhs.value();
    }
    template<class Rhs, optional_of<Derived> Lhs>
        requires(requires(const Derived &lhs, Rhs &&rhs) {
            lhs <=> std::forward<Rhs>(rhs);
        })
    friend constexpr auto operator<=>(const Lhs &lhs, Rhs &&rhs)
        -> std::partial_ordering
    {
        if (!lhs.has_value()) return std::partial_ordering::unordered;
        return lhs.value() <=> std::forward<Rhs>(rhs);
    }
};

} // namespace messner::mixin
