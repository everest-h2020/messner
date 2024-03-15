/** Declares the parser driver.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Translation.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Target/CFDlang/Utils/Symbols.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <cassert>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mlir::cfdlang::detail {

class Tokenizer;

using ParseResult   = OwningOpRef<ProgramOp>;

/** List of dim_size_t values. */
using DimList       = std::vector<dim_size_t>;
/** List of natural_t values. */
using NatList       = std::vector<natural_t>;

/** Assists the parser in building a ProgramOp. */
class ProgramBuilder {
public:
    explicit        ProgramBuilder(
        ImportContext &context,
        ImportLocation location,
        const Twine &name = {}
    )
    : m_context(context),
      m_builder(context.getContext()),
      m_program(),
      m_symbols()
    {
        // Create the program op.
        m_program = getBuilder()
            .create<ProgramOp>(
                getContext().getLocation(location, name),
                name.isTriviallyEmpty()
                    ? nullptr
                    : getBuilder().getStringAttr(name)
            );

        // Set the builder's insertion point.
        m_builder.setInsertionPointToStart(&m_program->getBody().front());
    }

    ImportContext&  getContext() const { return m_context; }
    OpBuilder&      getBuilder() { return m_builder; }

    bool            hasResult() const { return static_cast<bool>(m_program); }
    ParseResult     takeResult()
    {
        assert(hasResult());
        return std::move(m_program);
    }

    SymbolTable&    getSymbols() { return m_symbols; }

private:
    ImportContext   &m_context;
    OpBuilder       m_builder;
    ParseResult     m_program;
    SymbolTable     m_symbols;
};

/** Driver that implement CFDlang parsing. */
class ParseDriver {
public:
    explicit        ParseDriver(ImportContext &context);
                    ~ParseDriver();

    LogicalResult   parse();

    ImportContext&  getContext() const { return m_context; }

    bool            hasResult() const
    {
        return m_state.has_value() && m_state->hasResult();
    }
    ParseResult     takeResult()
    {
        assert(hasResult());
        return m_state->takeResult();
    }

private:
    friend class Parser;

    Tokenizer&      tokenizer() const { return *m_tokenizer; }

    void            error(ImportRange location, const Twine &message)
    {
        getContext().emitError(location, message);
    }
    void            warning(ImportRange location, const Twine &message)
    {
        getContext().emitWarning(location, message);
    }
    void            remark(ImportRange location, const Twine &message)
    {
        getContext().emitRemark(location, message);
    }
    void            note(ImportRange location, const Twine &message)
    {
        getContext().emitNote(location, message);
    }

    void            program();
    bool            stmt_begin(ImportRange location, StringRef id);
    bool            stmt_end(ImportRange location, StringRef id, AtomOp value);

    bool            decl(
        ImportRange location,
        StringRef id,
        AtomType type,
        DeclarationKind kind
    );

    AtomType        type_expr(ImportRange location, StringRef id);
    AtomType        type_expr(ImportRange location, shape_t shape);

    AtomOp          eval(ImportRange location, StringRef id);
    AtomOp          add(ImportRange location, AtomOp lhs, AtomOp rhs);
    AtomOp          sub(ImportRange location, AtomOp lhs, AtomOp rhs);
    AtomOp          mul(ImportRange location, AtomOp lhs, AtomOp rhs);
    AtomOp          div(ImportRange location, AtomOp lhs, AtomOp rhs);
    AtomOp          prod(ImportRange location, AtomOp lhs, AtomOp rhs);
    AtomOp          cont(
        ImportRange location,
        AtomOp op,
        NatList indices
    );

private:
    using State     = std::optional<ProgramBuilder>;
    ImportContext   &m_context;
    Tokenizer*      m_tokenizer;
    State           m_state;

    ProgramBuilder& result() { return *m_state; }
};

} // namespace mlir::cfdlang::detail
