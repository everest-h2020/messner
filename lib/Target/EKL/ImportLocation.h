/// Declares the ImportLocation struct.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "llvm/Support/SMLoc.h"

#include <cassert>

namespace mlir::ekl {

/// Implements a Bison-compatible adaptor for llvm::SMRange.
///
/// Bison requires the fields to called @c begin and @c end to be publically
/// accessible, which is why we can't use llvm::SMRange directly.
///
/// NOTE: llvm::SourceMgr is fundamentally incompatible with Unicode. So, we
///       don't make an effort to deal with it either.
struct ImportLocation {
    /// Initializes an invalid ImportLocation.
    /*implicit*/ ImportLocation() = default;
    /// Initializes an ImportLocation from the range [ @p begin, @p end ).
    /*implicit*/ ImportLocation(llvm::SMLoc begin, llvm::SMLoc end)
            : begin(begin),
              end(end)
    {
        assert(begin.getPointer() <= end.getPointer());
    }
    /// Initializes an ImportLocation from @p begin spanning @p size bytes.
    /*implicit*/ ImportLocation(llvm::SMLoc begin, std::size_t size = 0)
            : ImportLocation(
                begin,
                llvm::SMLoc::getFromPointer(begin.getPointer() + size))
    {}
    /// Initializes an ImportLocation from a @p range .
    /*implicit*/ ImportLocation(llvm::SMRange range)
            : ImportLocation(range.Start, range.End)
    {}

    /// Obtains an llvm::SMRange with the same contents.
    /*implicit*/ operator llvm::SMRange() const
    {
        return llvm::SMRange(begin, end);
    }

    /// The inclusive range start pointer.
    llvm::SMLoc begin;
    /// The exclusive range end pointer.
    llvm::SMLoc end;
};

} // namespace mlir::ekl
