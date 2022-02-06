/** Declares helpers for dealing with symbols in CFDlang programs.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Dialect/CFDlang/IR/Ops.h"

#include <set>
#include <string>
#include <utility>

namespace mlir::cfdlang::detail {

/** Kinds of CFDlang declarations. */
enum class DeclarationKind {
    Variable    = 0,
    Input       = 1,
    Output      = 2,
    Type        = 3
};

/** Stores information about a CFDlang declaration. */
class Declaration {
public:
    using Location      = ImportRange;
    using Definition    = Optional<Location>;

    explicit            Declaration(
        StringRef id,
        Location location,
        DeclarationKind kind,
        AtomType type
    )
    : m_id(id), m_location(location), m_kind(kind), m_type(type)
    {}

    StringRef           getId() const { return m_id; }
    Location            getLocation() const { return m_location; }

    DeclarationKind     getKind() const { return m_kind; }
    bool                isType() const
    {
        return getKind() == DeclarationKind::Type;
    }
    bool                isAtom() const { return !isType(); }
    bool                canHaveDefinition() const
    {
        return getKind() == DeclarationKind::Variable
            || getKind() == DeclarationKind::Output;
    }

    AtomType            getType() const { return m_type; }

    Definition          getDefinition() const { return m_definition; }
    bool                isDefined() const { return m_definition.hasValue(); }
    void                define(Location location) const
    {
        m_definition = location;
    }

    bool                operator<(const Declaration& rhs) const
    {
        return getId() < rhs.getId();
    }
    bool                operator<(StringRef rhs) const { return getId() < rhs; }
    friend bool         operator<(StringRef lhs, const Declaration &rhs)
    {
        return lhs < rhs.getId();
    }

private:
    friend class SymbolTable;

    std::string         m_id;
    Location            m_location;
    DeclarationKind     m_kind;
    AtomType            m_type;
    mutable Definition  m_definition;
};

/** Stores a table of declarations in a CFDlang program. */
class SymbolTable {
public:
    using Symbol        = const Declaration&;
    using InsertResult  = std::pair<LogicalResult, Symbol>;

    /*implicit*/        SymbolTable() = default;

    const Declaration*  lookup(StringRef id) const
    {
        const auto it = m_impl.find(id);
        return it == m_impl.end() ? nullptr : &*it;
    }

    InsertResult        insert(Declaration &&declaration)
    {
        auto [it, inserted] = m_impl.insert(std::move(declaration));
        return std::make_pair(success(inserted), *it);
    }
    template<class... Args>
    InsertResult        emplace(StringRef id, Args&&... args)
    {
        const auto hint = m_impl.lower_bound(id);
        if (hint != m_impl.end() && hint->getId() == id) {
            return std::make_pair(failure(), *hint);
        }

        return std::make_pair(
            success(),
            *m_impl.emplace_hint(hint, id, std::forward<Args>(args)...)
        );
    }

private:
    using Storage       = std::set<Declaration, std::less<void>>;
    Storage             m_impl;
};

} // namespace mlir::cfdlang::detail
