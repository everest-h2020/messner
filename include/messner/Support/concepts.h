/// Declares additional C++ concepts.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <concepts>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>
#include <type_traits>

namespace messner {

/// Concept matching @p Base or any of its derived types.
template<class T, class Base>
concept derived_or_same = std::same_as<T, Base> || std::is_base_of_v<Base, T>;

/// Concept matching any mlir::Attribute smart pointer type.
template<class T>
concept attr_constraint = derived_or_same<T, mlir::Attribute>;

/// Concept matching any mlir::Type smart pointer type.
template<class T>
concept type_constraint = derived_or_same<T, mlir::Type>;

} // namespace messner
