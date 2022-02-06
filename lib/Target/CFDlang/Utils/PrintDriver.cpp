/** Implements the print driver.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/Utils/PrintDriver.h"

#include <cassert>

using namespace mlir;
using namespace mlir::cfdlang;

namespace mlir::cfdlang::detail {

void PrintDriver::print(ProgramOp op)
{
    // Print a header.
    m_out << "// MLIR cfdlang dialect export of program ";
    m_out << op.getName().getValueOr("<unnamed>");
    m_out << "\n";

    // Declare all symbols.
    auto decls = op.getBody()->getOps<DeclarationOp>();
    for (auto decl : decls) {
        declare(decl);
    }

    // Leave a blank line in between.
    m_out << "\n";

    // Emit the code.
    auto defs = op.getBody()->getOps<DefinitionOp>();
    for (auto def : defs) {
        print(def);
    }

    // Trailing newline!
    m_out << "\n";
}

void PrintDriver::declare(DeclarationOp op)
{
    // Determine the kind.
    auto kind = DeclarationKind::Variable;
    if (isa<InputOp>(op)) {
        kind = DeclarationKind::Input;
    } else if (isa<OutputOp>(op)) {
        kind = DeclarationKind::Output;
    }

    // Remember the symbol.
    auto [ok, sym] = m_symbols.emplace(
        op.getName(),
        ImportLocation(),
        kind,
        op.getAtomType()
    );
    assert(succeeded(ok));

    // Print the declaration.
    m_out << "var ";
    switch (kind) {
        case DeclarationKind::Input: m_out << "input "; break;
        case DeclarationKind::Output: m_out << "output "; break;
        default: break;
    }
    m_out << op.getName() << " : [";
    llvm::interleave(op.getAtomShape(), m_out, " ");
    m_out << "]\n";
}

void PrintDriver::print(DefinitionOp op)
{
    // Lookup the symbol.
    auto sym = m_symbols.lookup(op.getName());
    assert(sym);

    // Print the lhs.
    m_out << op.getName() << " = ";
    // Print the rhs.
    m_prec.clear();
    print(op.getAtom().getDefiningOp<AtomOp>());
    m_out << "\n";
}

void PrintDriver::print(AtomOp op)
{
    if (auto concrete = dyn_cast<EvalOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<AddOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<SubOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<MulOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<DivOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<ProductOp>(op.getOperation())) {
        print(concrete);
        return;
    }
    if (auto concrete = dyn_cast<ContractOp>(op.getOperation())) {
        print(concrete);
        return;
    }

    llvm_unreachable("AtomOp");
}

void PrintDriver::print(EvalOp op)
{
    auto sym = m_symbols.lookup(op.name().getLeafReference());
    assert(sym);
    m_out << sym->getId();
}

void PrintDriver::print(ContractOp op)
{
    beginExpr(-1);
    print(op.operand().getDefiningOp<AtomOp>());
    m_out << " . [";
    auto indices = op.indicesAttr().getAsValueRange();
    for (auto it = indices.begin(); it != indices.end(); ++it) {
        m_out << '[' << *it++ - 1 << ' ' << *it - 1 << ']';
    }
    m_out << ']';
    endExpr();
}

} // namespace mlir::cfdlang::detail
