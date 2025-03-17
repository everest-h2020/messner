/// Implements the EKL definitions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "Definitions.h"

#include "llvm/Support/Debug.h"

#include <cassert>
#include <mlir/IR/SymbolTable.h>

#define DEBUG_TYPE "ekl-parser"

using namespace mlir;
using namespace mlir::ekl;

[[nodiscard]] static NameLoc getLoc(SymbolOpInterface symbol)
{
    assert(symbol);

    if (const auto nameLoc = llvm::dyn_cast<NameLoc>(symbol.getLoc()))
        return nameLoc;

    return NameLoc::get(symbol.getNameAttr(), symbol.getLoc());
}

//===----------------------------------------------------------------------===//
// Definition implementation
//===----------------------------------------------------------------------===//

Definition::Definition(SymbolOpInterface symbol)
        : DefinitionUnion(symbol),
          m_name(::getLoc(symbol))
{}

Definition::Kind Definition::getKind() const
{
    const auto &self = static_cast<const DefinitionUnion &>(*this);
    if (llvm::isa<SymbolOpInterface>(self)) return Kind::Symbol;
    if (llvm::isa<Expression>(self)) return Kind::Expression;
    if (llvm::isa<LiteralAttr>(self)) return Kind::Constant;
    if (llvm::isa<Type>(self)) return Kind::Type;

    llvm_unreachable("invalid Definition");
}

//===----------------------------------------------------------------------===//
// Scopes implementation
//===----------------------------------------------------------------------===//

std::pair<Definition *, bool> Scopes::define(Definition def)
{
    LLVM_DEBUG(
        llvm::dbgs() << "[Scopes] define(" << def.getName() << ", "
                     << def.getKind() << ")\n");

    if (const auto sym = lookup(def.getName())) {
        // Name is already defined.
        return std::make_pair(sym, false);
    }

    // Name is inserted into the current scope.
    const auto result = m_impl.back().insert(def);
    assert(result.second);
    return result;
}

std::pair<Definition *, bool> Scopes::shadowOuter(Definition def)
{
    LLVM_DEBUG(
        llvm::dbgs() << "[Scopes] shadowOuter(" << def.getName() << ", "
                     << def.getKind() << ")\n");

    // Insert into the current scope, which automatically shadows outer
    // scopes, but does not shadow the current scope.
    return m_impl.back().insert(def);
}

Definition &Scopes::shadow(Definition def)
{
    LLVM_DEBUG(
        llvm::dbgs() << "[Scopes] shadow(" << def.getName() << ", "
                     << def.getKind() << ")\n");

    // Shadow in the current scope.
    return m_impl.back().shadow(def);
}

Scope &Scopes::push(llvm::SMLoc startLoc)
{
    LLVM_DEBUG(
        llvm::dbgs()
        << "[Scopes] push("
        << std::bit_cast<std::uint64_t>(startLoc.getPointer()) << ")\n");

    return m_impl.emplace_back(startLoc);
}

void Scopes::pop()
{
    LLVM_DEBUG(
        llvm::dbgs() << "[Scopes] pop("
                     << std::bit_cast<std::uint64_t>(
                            m_impl.back().getStartLoc().getPointer())
                     << ")\n");

    m_impl.pop_back();
}
