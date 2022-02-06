/** Declares the Tokenizer class.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "Lexer.hpp"

namespace mlir::cfdlang::detail {

/** Derives from Lexer to enable the llvm::SMLoc support. */
class Tokenizer : public Lexer {
public:
    using Lexer::Lexer;

    virtual mlir::concepts::ImportRange location() override
    {
        auto result = Lexer::location();

        // BUG: This does not work.
        /*
        // Calculate the location pointers.
        const auto begin = Lexer::text();
        const auto end = begin + Lexer::size();

        // Augment the location information using the pointers.
        result.begin.location = llvm::SMLoc::getFromPointer(begin);
        result.end.location = llvm::SMLoc::getFromPointer(end);
        */

        const auto fileId = getContext().getSource().getMainFileID();
        result.begin.location = getContext().getSource()
            .FindLocForLineAndColumn(fileId, result.begin.line, result.begin.column + 1);
        result.end.location = getContext().getSource()
            .FindLocForLineAndColumn(fileId, result.end.line, result.end.column + 1);
        return result;
    }
};

} // namespace mlir::cfdlang::detail
