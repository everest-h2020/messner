/// Declares the EKL definitions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/StringMap.h"

#include <type_traits>

namespace llvm {

template<>
struct PointerLikeTypeTraits<mlir::SymbolOpInterface> {
    static inline const void *getAsVoidPointer(mlir::SymbolOpInterface value)
    {
        return value.getAsOpaquePointer();
    }
    static inline mlir::SymbolOpInterface getFromVoidPointer(void *ptr)
    {
        return mlir::SymbolOpInterface::getFromOpaquePointer(ptr);
    }

    static constexpr int NumLowBitsAvailable =
        PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
};

template<>
struct PointerLikeTypeTraits<mlir::ekl::LiteralAttr> {
    static inline const void *getAsVoidPointer(mlir::ekl::LiteralAttr value)
    {
        return value.getAsOpaquePointer();
    }
    static inline mlir::ekl::LiteralAttr getFromVoidPointer(void *ptr)
    {
        return cast<mlir::ekl::LiteralAttr>(
            mlir::ekl::LiteralAttr::getFromOpaquePointer(ptr));
    }

    static constexpr int NumLowBitsAvailable =
        PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
};

} // namespace llvm

namespace mlir::ekl {

//===----------------------------------------------------------------------===//
// Definition
//===----------------------------------------------------------------------===//

using DefinitionUnion =
    llvm::PointerUnion<SymbolOpInterface, Expression, LiteralAttr, Type>;

class Definition : public DefinitionUnion {
public:
    enum class Kind { Symbol, Expression, Constant, Type };

    /*implicit*/ Definition(SymbolOpInterface symbol);
    /*implicit*/ Definition(NameLoc name, Expression expr)
            : DefinitionUnion(expr),
              m_name(name)
    {}
    /*implicit*/ Definition(NameLoc name, LiteralAttr constant)
            : DefinitionUnion(constant),
              m_name(name)
    {}
    /*implicit*/ Definition(NameLoc name, Type type)
            : DefinitionUnion(type),
              m_name(name)
    {}

    [[nodiscard]] NameLoc getLoc() const { return m_name; }
    [[nodiscard]] StringAttr getNameAttr() const { return getLoc().getName(); }
    [[nodiscard]] StringRef getName() const { return getNameAttr().getValue(); }
    [[nodiscard]] Kind getKind() const;

    /*implicit*/ operator Location() const { return getLoc(); }

private:
    NameLoc m_name;
};

inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &out, Definition::Kind kind)
{
    switch (kind) {
    case Definition::Kind::Symbol:     return out << "symbol";
    case Definition::Kind::Expression: return out << "expression";
    case Definition::Kind::Constant:   return out << "constant";
    case Definition::Kind::Type:       return out << "type";
    }
    return out;
}

inline Diagnostic &operator<<(Diagnostic &out, Definition::Kind kind)
{
    switch (kind) {
    case Definition::Kind::Symbol:     return out << "symbol";
    case Definition::Kind::Expression: return out << "expression";
    case Definition::Kind::Constant:   return out << "constant";
    case Definition::Kind::Type:       return out << "type";
    }
    return out;
}

//===----------------------------------------------------------------------===//
// Scope
//===----------------------------------------------------------------------===//

namespace detail {

using ScopeImpl = llvm::StringMap<Definition>;

template<bool IsConst>
using ScopeImplIterator =
    std::conditional_t<IsConst, ScopeImpl::const_iterator, ScopeImpl::iterator>;

template<bool IsConst>
struct ScopeIterator
        : llvm::iterator_adaptor_base<
              ScopeIterator<IsConst>,
              ScopeImplIterator<IsConst>,
              std::forward_iterator_tag,
              std::conditional_t<IsConst, const Definition, Definition>> {
    using base = llvm::iterator_adaptor_base<
        ScopeIterator<IsConst>,
        ScopeImplIterator<IsConst>,
        std::forward_iterator_tag,
        std::conditional_t<IsConst, const Definition, Definition>>;
    explicit ScopeIterator(ScopeImplIterator<IsConst> wrapped) : base(wrapped)
    {}

    auto &operator*() const { return base::wrapped()->second; }
};

} // namespace detail

class Scope {
    [[nodiscard]] static auto *lookup(auto &impl, StringRef name)
    {
        const auto it = impl.find(name);
        return it != impl.end() ? &it->second : nullptr;
    }

public:
    using value_type      = Definition;
    using size_type       = detail::ScopeImpl::size_type;
    using pointer         = Definition *;
    using const_pointer   = const Definition *;
    using reference       = Definition &;
    using const_reference = const Definition &;
    using iterator        = detail::ScopeIterator<false>;
    using const_iterator  = detail::ScopeIterator<true>;

    explicit Scope(llvm::SMLoc startLoc) : m_startLoc(startLoc), m_impl() {}

    llvm::SMLoc getStartLoc() const { return m_startLoc; }

    [[nodiscard]] const_pointer lookup(StringRef name) const
    {
        return lookup(m_impl, name);
    }
    [[nodiscard]] pointer lookup(StringRef name)
    {
        return lookup(m_impl, name);
    }

    std::pair<pointer, bool> insert(Definition def)
    {
        const auto [it, ok] = m_impl.try_emplace(def.getName(), def);
        return std::make_pair(&it->second, ok);
    }

    reference shadow(Definition def)
    {
        const auto [present, ok] = insert(def);
        if (!ok) *present = def;
        return *present;
    }

    [[nodiscard]] bool empty() const { return m_impl.empty(); }
    [[nodiscard]] size_type size() const { return m_impl.size(); }
    [[nodiscard]] const_iterator begin() const
    {
        return const_iterator(m_impl.begin());
    }
    [[nodiscard]] iterator begin() { return iterator(m_impl.begin()); }
    [[nodiscard]] const_iterator end() const
    {
        return const_iterator(m_impl.end());
    }
    [[nodiscard]] iterator end() { return iterator(m_impl.end()); }

private:
    llvm::SMLoc m_startLoc;
    detail::ScopeImpl m_impl;
};

//===----------------------------------------------------------------------===//
// Scopes
//===----------------------------------------------------------------------===//

class Scopes {
    [[nodiscard]] static auto *lookup(auto &impl, StringRef name)
    {
        for (auto it = impl.rbegin(); it != impl.rend(); ++it)
            if (const auto result = it->lookup(name)) return result;
        return decltype(&*impl.back().begin()){};
    }

public:
    using impl_type       = llvm::SmallVector<Scope>;
    using value_type      = impl_type::value_type;
    using size_type       = impl_type::size_type;
    using pointer         = impl_type::pointer;
    using const_pointer   = impl_type::const_pointer;
    using reference       = impl_type::reference;
    using const_reference = impl_type::const_reference;
    using iterator        = impl_type::reverse_iterator;
    using const_iterator  = impl_type::const_reverse_iterator;

    /*implicit*/ Scopes() = default;

    [[nodiscard]] const Definition *lookup(StringRef name) const
    {
        return lookup(m_impl, name);
    }
    [[nodiscard]] Definition *lookup(StringRef name)
    {
        return lookup(m_impl, name);
    }

    std::pair<Definition *, bool> define(Definition def);
    std::pair<Definition *, bool> shadowOuter(Definition def);
    Definition &shadow(Definition def);

    Scope &push(llvm::SMLoc startLoc);
    void pop();

    const Scope &peek(unsigned up = 0) const
    {
        assert(up < size());
        return *std::next(begin(), up);
    }
    Scope &peek(unsigned up = 0)
    {
        assert(up < size());
        return *std::next(begin(), up);
    }

    void clear() { m_impl.clear(); }

    [[nodiscard]] bool empty() const { return m_impl.empty(); }
    [[nodiscard]] size_type size() const { return m_impl.size(); }
    [[nodiscard]] const_iterator begin() const { return m_impl.rbegin(); }
    [[nodiscard]] iterator begin() { return m_impl.rbegin(); }
    [[nodiscard]] const_iterator end() const { return m_impl.rend(); }
    [[nodiscard]] iterator end() { return m_impl.rend(); }

    [[nodiscard]] const_reference front() const { return m_impl.back(); }
    [[nodiscard]] reference front() { return m_impl.back(); }
    [[nodiscard]] const_reference back() const { return m_impl.front(); }
    [[nodiscard]] reference back() { return m_impl.front(); }

private:
    llvm::SmallVector<Scope> m_impl;
};

} // namespace mlir::ekl
