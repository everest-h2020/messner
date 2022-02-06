/** Declares utilities for the lexer implementation.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/IR/Base.h"

#include <cassert>
#include <charconv>
#include <system_error>

namespace mlir::cfdlang::detail {

/** Parses an integer from [ @p first, @p last ). */
template<class Int>
inline Int parseInt(const char *first, const char *last)
{
    Int result;
    auto [ptr, ec] = std::from_chars(first, last, result);
    assert(ec == std::errc() && ptr == last);
    return result;
}
/** Parses an integer from [ @p first, @p first + @p len ). */
template<class Int>
inline Int parseInt(const char *first, std::size_t len)
{
    return parseInt<Int>(first, first + len);
}

} // namespace mlir::cfdlang::detail
