/** Implements the CFDlang dialect operations.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Dialect/TeIL/Concepts/Shape.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cfdlang;

//===- Custom directives --------------------------------------------------===//

// DimensionSize    ::= Integer | '?'
static void printDimSize(OpAsmPrinter &p, dim_size_t dimSize)
{
    if (ShapedType::isDynamic(dimSize)) {
        p << '?';
    } else {
        p << dimSize;
    }
}
static ParseResult parseDimSize(OpAsmParser &p, dim_size_t& dimSize)
{
    if (!p.parseOptionalQuestion()) {
        dimSize = teil::dynamic_size;
        return success();
    }

    return p.parseInteger(dimSize);
}

// AtomShape        ::= '[' ( DimSize { ',' DimSize } ) ']'
static void printAtomShape(OpAsmPrinter &p, shape_t atomShape)
{
    p << '[';

    auto it = atomShape.begin();
    if (it != atomShape.end()) {
        printDimSize(p, *it);
        for (++it; it != atomShape.end(); ++it) {
            p << ' ';
            printDimSize(p, *it);
        }
    }

    p << ']';
}
static ParseResult parseAtomShape(
    OpAsmParser &p,
    teil::ShapeBuilder &atomShape
)
{
    if (p.parseLSquare()) return failure();

    while (p.parseOptionalRSquare())
    {
        if (parseDimSize(p, atomShape.emplace_back())) return failure();
    }

    return success();
}

// AtomType         ::= AtomShape
static void printAtomType(OpAsmPrinter &p, Operation*, Type atomType)
{
    printAtomShape(p, atomType.cast<AtomType>().getShape());
}
static ParseResult parseAtomType(OpAsmParser &p, AtomType &atomType)
{
    teil::ShapeStorage atomShape;
    if (parseAtomShape(p, atomShape)) return failure();

    atomType = AtomType::get(p.getBuilder().getContext(), atomShape);
    return success();
}
static ParseResult parseAtomType(OpAsmParser &p, Type &atomType)
{
    AtomType temp;
    if (succeeded(parseAtomType(p, temp))) {
        atomType = temp;
        return success();
    }

    return failure();
}

// AtomTypeAttr     ::= AtomType
static void printAtomTypeAttr(OpAsmPrinter &p, Operation* op, TypeAttr attr)
{
    printAtomType(p, op, attr.getValue());
}
static ParseResult parseAtomTypeAttr(OpAsmParser &p, TypeAttr &attr)
{
    Type atomType;
    if (parseAtomType(p, atomType)) return failure();

    attr = TypeAttr::get(atomType);
    return success();
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::build(OpBuilder&, OperationState&) {}

//===----------------------------------------------------------------------===//
// EvalOp
//===----------------------------------------------------------------------===//

LogicalResult EvalOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // Find the matching declaration.
    auto declaration = symbolTable.lookupNearestSymbolFrom<DeclarationOp>(
        *this,
        name()
    );

    if (!declaration) {
        return emitOpError()
            << "'" << name() << "' does not reference a valid symbol";
    }

    // Check that our use is consistent with the declared type.
    if (declaration.getAtomType() != getResult().getType()) {
        return emitOpError()
            << "result type " << getResult().getType()
            << " does not match declared type " << declaration.getAtomType()
            << " of symbol @" << name();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ArithmeticOp
//===----------------------------------------------------------------------===//

static void printArithmeticOp(OpAsmPrinter &p, Operation *op)
{
    p << op->getName();
    p.printOperands(op->getOperands());
    p << " : ";
    printAtomType(p, op, op->getResult(0).getType());
    p.printOptionalAttrDict(op->getAttrs());
}
static ParseResult parseArithmeticOp(
    OpAsmParser &p,
    OperationState &result
)
{
    SmallVector<OpAsmParser::OperandType> operands;
    if (p.parseOperandList(operands)) return failure();
    if (p.parseColon()) return failure();
    AtomType atomType;
    if (parseAtomType(p, atomType)) return failure();
    result.types.push_back(atomType);
    if (p.parseOptionalAttrDict(result.attributes)) return failure();
    return p.resolveOperands(operands, atomType, result.operands);
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CFDlang/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CFDlangDialect
//===----------------------------------------------------------------------===//

void CFDlangDialect::registerOps()
{
    addOperations<
        #define GET_OP_LIST
        #include "mlir/Dialect/CFDlang/IR/Ops.cpp.inc"
    >();
}
