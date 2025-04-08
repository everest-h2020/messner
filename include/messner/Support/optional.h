/// Declares some helpers for working with the std::optional type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <optional>
#include <type_traits>

namespace messner {

/// Trait that detects std::optional specializations.
template<class T>
struct is_optional;

/// Indicates whether @p T is a specialization of std::optional.
template<class T>
constexpr auto is_optional_v = is_optional<T>::value;

/// Concept that matches any std::optional specialization.
template<class T>
concept any_optional = is_optional_v<T>;

/// Concept that matches std::optional<U>.
template<class T, class U>
concept optional_of = std::same_as<T, std::optional<U>>;

/// Obtains an std::optional specialization for @p T , if it isn't already.
template<class T>
using ensure_optional_t =
    std::conditional_t<any_optional<T>, T, std::optional<T>>;

} // namespace messner

namespace messner {

//===----------------------------------------------------------------------===//
// is_optional implementation
//===----------------------------------------------------------------------===//

template<class T>
struct is_optional : std::false_type {};

template<class T>
struct is_optional<std::optional<T>> : std::true_type {};

} // namespace messner
