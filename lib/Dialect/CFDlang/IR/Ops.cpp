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

    // Check that the symbol was declared.
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
// AtomOp
//===----------------------------------------------------------------------===//

static void printAtomOp(OpAsmPrinter &p, Operation *op)
{
    p << op->getName() << " ";

    // operands `:`
    p.printOperands(op->getOperands());
    p << " : ";

    // custom<AtomType>(type(operands))
    auto it = op->operand_begin();
    if (it != op->operand_end()) {
        printAtomType(p, op, (*it++).getType());
        for (; it != op->operand_end(); ++it) {
            p << ", ";
            printAtomType(p, op, (*it).getType());
        }
    }

    // attr-dict
    p.printOptionalAttrDict(op->getAttrs());
}
template<class ConcreteOp>
static ParseResult parseAtomOp(
    OpAsmParser &p,
    OperationState &result
)
{
    // operands `:`
    SmallVector<OpAsmParser::OperandType> operands;
    if (p.parseOperandList(operands)) return failure();
    if (p.parseColon()) return failure();

    // custom<AtomType>(type(operands))
    SmallVector<AtomType> operandTypes;
    for (auto it = operands.begin(); ;) {
        if (parseAtomType(p, operandTypes.emplace_back())) return failure();
        if (p.resolveOperand(*it, operandTypes.back(), result.operands))
            return failure();
        if (++it == operands.end()) break;
        if (p.parseComma()) return failure();
    }

    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // Infer result type.
    using AtomOpModel = cfdlang::detail::AtomOpInterfaceTraits::Model<ConcreteOp>;
    if (
        auto atomType = AtomOpModel::inferAtomType(
            p.getBuilder().getContext(),
            result.location,
            result.operands,
            DictionaryAttr::get(p.getBuilder().getContext(), result.attributes),
            result.regions
        )
    ) {
        result.types.push_back(atomType);
        return success();
    }
    return p.emitError(p.getNameLoc()) << "incompatible operand types";
}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

LogicalResult ProductOp::inferAtomShape(
    MLIRContext*,
    Optional<Location>,
    ValueRange operands,
    DictionaryAttr,
    RegionRange,
    teil::ShapeBuilder &atomShape
)
{
    // Concatenate shapes.
    for (auto op : operands) {
        auto atom = op.dyn_cast<Atom>();
        if (!atom) return failure();
        const auto shape = atom.getShape();
        atomShape.append(shape.begin(), shape.end());
    }

    return success();
}

FailureOr<teil::AtomSize> ProductOp::reifyAtomSize(OpBuilder &builder)
{
    // TODO: Implement.
    return failure();
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ContractOp op)
{
    p << op.getOperationName() << ' ';

    // $operand `:` custom<AtomType>($operand)
    p << op.getOperand() << " : ";
    printAtomType(p, op, op.getOperand().getType());

    // `indices` custom<NatPairs>($indices)
    p << " indices ";
    auto indices = op.indicesAttr().getAsValueRange();
    for (auto it = indices.begin(); it != indices.end(); ++it) {
        p << '[' << *it++ << ' ' << *it << ']';
    }

    // attr-dict
    p.printOptionalAttrDict(op->getAttrs(), {"indices"});
}
static ParseResult parseContractOp(OpAsmParser &p, OperationState &result)
{
    // $operand `:`
    OpAsmParser::OperandType operand;
    if (p.parseOperand(operand)) return failure();
    if (p.parseColon()) return failure();

    // custom<AtomType>($operand)
    AtomType operandType;
    if (parseAtomType(p, operandType)) return failure();
    if (p.resolveOperand(operand, operandType, result.operands))
        return failure();

    // `indices` custom<NatPairs>($indices)
    if (p.parseKeyword("indices")) return failure();
    SmallVector<natural_t> indices;
    while (!p.parseOptionalLSquare()) {
        if (p.parseInteger(indices.emplace_back())) return failure();
        if (p.parseInteger(indices.emplace_back())) return failure();
        if (p.parseRSquare()) return failure();
    }
    result.addAttribute(
        "indices",
        teil::NatArrayAttr::get(p.getBuilder().getContext(), indices)
    );

    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // Infer result type.
    using AtomOpModel = cfdlang::detail::AtomOpInterfaceTraits::Model<ContractOp>;
    if (
        auto atomType = AtomOpModel::inferAtomType(
            p.getBuilder().getContext(),
            result.location,
            result.operands,
            DictionaryAttr::get(p.getBuilder().getContext(), result.attributes),
            result.regions
        )
    ) {
        result.types.push_back(atomType);
        return success();
    }
    return p.emitError(p.getNameLoc()) << "invalid reduction indices";
}

LogicalResult ContractOp::inferAtomShape(
    MLIRContext*,
    Optional<Location>,
    ValueRange operands,
    DictionaryAttr attributes,
    RegionRange,
    teil::ShapeBuilder &atomShape
)
{
    auto operand = operands.front().dyn_cast<Atom>();
    if (!operand) return failure();
    auto indicesAttr = attributes.getAs<teil::NatArrayAttr>("indices");
    if (!indicesAttr) return failure();
    auto indices = indicesAttr.getAsValueRange();

    // Initialize result shape from operand shape.
    atomShape.assign(
        operand.getShape().begin(),
        operand.getShape().end()
    );

    // Check indices and set all pairs' dimensions to 0.
    for (auto it = indices.begin(); it != indices.end(); ++it) {
        const auto l = *it++ - 1;
        if (it == indices.end()) return failure();
        const auto r = *it - 1;

        if (l == r || l >= atomShape.size() || r >= atomShape.size())
            return failure();

        auto &l_dim = atomShape[l], &r_dim = atomShape[r];
        if (
            l_dim != r_dim
            && l_dim != teil::dynamic_size
            && r_dim != teil::dynamic_size
        ) {
            return failure();
        }

        l_dim = r_dim = 0;
    }

    // Remove all 0 dimensions.
    atomShape.erase(
        std::remove_if(
            atomShape.begin(),
            atomShape.end(),
            [](auto x){ return x == 0; }
        ),
        atomShape.end()
    );

    return success();
}

FailureOr<teil::AtomSize> ContractOp::reifyAtomSize(OpBuilder &builder)
{
    // TODO: Implement.
    return failure();
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
