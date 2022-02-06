/** Declares the print driver class.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "mlir/Concepts/Translation.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
// NOTE: Could use this if indentation was required.
// #include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/CFDlang/Utils/Symbols.h"
#include "mlir/IR/Builders.h"

#include <cassert>

namespace mlir::cfdlang::detail {

class PrintDriver {
public:
    explicit            PrintDriver(llvm::raw_ostream &out)
    : m_out(out), m_symbols()
    {}

    void                print(ProgramOp op);

    void                declare(DeclarationOp op);

    void                print(DefinitionOp op);
    void                print(AtomOp op);

private:
    void                beginExpr(int prec)
    {
        if (!m_prec.empty() && prec < m_prec.back()) {
            m_out << "(";
        }
        m_prec.push_back(prec);
    }
    void                print(EvalOp op);
    void                printBinop(Operation* op, int prec, char sym)
    {
        beginExpr(prec);
        print(op->getOperand(0).getDefiningOp<AtomOp>());
        m_out << ' ' << sym << ' ';
        print(op->getOperand(1).getDefiningOp<AtomOp>());
        endExpr();
    }
    void                print(AddOp op) { printBinop(op, 0, '+'); }
    void                print(SubOp op) { printBinop(op, 0, '-'); }
    void                print(MulOp op) { printBinop(op, 1, '*'); }
    void                print(DivOp op) { printBinop(op, 1, '/'); }
    void                print(ProductOp op) { printBinop(op, 2, '#'); }
    void                print(ContractOp op);
    void                endExpr()
    {
        assert(!m_prec.empty());
        auto prec = m_prec.back();
        m_prec.pop_back();
        if (!m_prec.empty() && prec < m_prec.back()) {
            m_out << ")";
        }
    }

    llvm::raw_ostream   &m_out;
    SymbolTable         m_symbols;
    std::vector<int>    m_prec;
};

} // namespace mlir::cfdlang::detail
