/// Implements a C++20 constexpr friendly llvm::iterator_facade_base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>

namespace messner {

/// Concept that matches any (at least) forward iterator tag.
template<class T>
concept some_forward_iter_tag = std::same_as<T, std::forward_iterator_tag>
                             || std::same_as<T, std::bidirectional_iterator_tag>
                             || std::same_as<T, std::random_access_iterator_tag>
                             || std::same_as<T, std::contiguous_iterator_tag>;

/// See llvm::iterator_facade_base for more information.
template<
    class Derived,
    class IteratorCategory,
    class ValueType,
    class DifferenceType = std::ptrdiff_t,
    class Pointer        = ValueType *,
    class Reference      = ValueType &>
class iter_facade {
public:
    static_assert(
        some_forward_iter_tag<IteratorCategory>,
        "only applies to forward iterators");

    using iterator_category = IteratorCategory;
    using value_type        = ValueType;
    using difference_type   = DifferenceType;
    using pointer           = Pointer;
    using reference         = Reference;

protected:
    static constexpr auto is_random_access =
        std::same_as<iterator_category, std::random_access_iterator_tag>;
    static constexpr auto is_bidirectional =
        std::same_as<iterator_category, std::bidirectional_iterator_tag>
        || is_random_access;

    class proxy {
    public:
        constexpr auto operator->() const -> pointer;
        /*implicit*/ constexpr operator reference() const { return *_it; }

    private:
        /*implicit*/ constexpr proxy(Derived it) : _it(std::move(it)) {}

        Derived _it;

        friend iter_facade;
    };

    constexpr auto self() -> Derived &;
    constexpr auto self() const -> const Derived &;

public:
    constexpr auto operator->() const -> proxy;

    constexpr auto operator++(int) -> Derived;

    template<class Result = Derived>
        requires(is_bidirectional)
    constexpr auto operator--(int) -> Result;

    template<class Result = Derived &>
        requires(is_random_access)
    constexpr auto operator++() -> Result;
    template<class Result = Derived>
        requires(is_random_access)
    constexpr auto operator+(difference_type rhs) const -> Result;
    template<std::same_as<Derived> Rhs>
        requires(is_random_access)
    friend constexpr auto operator+(difference_type lhs, const Rhs &rhs)
        -> Derived
    {
        return rhs + lhs;
    }

    template<class Result = Derived &>
        requires(is_random_access)
    constexpr auto operator--() -> Result;
    template<class Result = Derived>
        requires(is_random_access)
    constexpr auto operator-(difference_type rhs) const -> Result;

    template<class Result = proxy>
        requires(is_random_access)
    constexpr auto operator[](difference_type dist) const -> Result;
};

} // namespace messner

namespace messner {

//===----------------------------------------------------------------------===//
// iter_facade::proxy implementation
//===----------------------------------------------------------------------===//

template<class T0, class T1, class T2, class T3, class T4, class T5>
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::proxy::operator->() const
    -> pointer
{
    return std::pointer_traits<pointer>::pointer_to(*_it);
}

//===----------------------------------------------------------------------===//
// iter_facade implementation
//===----------------------------------------------------------------------===//

template<class T0, class T1, class T2, class T3, class T4, class T5>
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::self() -> T0 &
{
    return static_cast<T0 &>(*this);
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::self() const -> const T0 &
{
    return static_cast<const T0 &>(*this);
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::operator->() const -> proxy
{
    return proxy(this->self());
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::operator++(int) -> T0
{
    auto copy(this->self());
    ++this->self();
    return copy;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_bidirectional)
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::operator--(int) -> Result
{
    auto copy(this->self());
    --this->self();
    return copy;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_random_access)
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::operator++() -> Result
{
    return this->self() += 1;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_random_access)
constexpr auto
iter_facade<T0, T1, T2, T3, T4, T5>::operator+(difference_type rhs) const
    -> Result
{
    auto copy(this->self());
    copy += rhs;
    return copy;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_random_access)
constexpr auto iter_facade<T0, T1, T2, T3, T4, T5>::operator--() -> Result
{
    return this->self() -= 1;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_random_access)
constexpr auto
iter_facade<T0, T1, T2, T3, T4, T5>::operator-(difference_type rhs) const
    -> Result
{
    auto copy(this->self());
    copy -= rhs;
    return copy;
}

template<class T0, class T1, class T2, class T3, class T4, class T5>
template<class Result>
    requires(iter_facade<T0, T1, T2, T3, T4, T5>::is_random_access)
constexpr auto
iter_facade<T0, T1, T2, T3, T4, T5>::operator[](difference_type dist) const
    -> Result
{
    return Result(this->self()->operator+(dist));
}

} // namespace messner
