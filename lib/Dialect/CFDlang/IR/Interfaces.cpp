/** Implements the CFDlang dialect interfaces.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/CFDlang/IR/Interfaces.h"

#include "llvm/Support/Debug.h"
#include "mlir/IR/OpImplementation.h"

#define DEBUG_TYPE "cfdlang-interfaces"

using namespace mlir;
using namespace mlir::cfdlang;

LogicalResult mlir::cfdlang::interface_defaults::verifyDefinitionOp(
    Operation *op
)
{
    auto definition = cast<DefinitionOp>(op);

    // Ensure that the provided value actually matches the declaration.
    auto atom = definition.getAtom();
    if (definition.getAtomType() != atom.getType()) {
        return op->emitOpError()
            << "result type " << atom.getType()
            << " does not match the declared type " << definition.getAtomType()
            << " of the symbol @" << definition.getName();
    }

    return success();
}

LogicalResult mlir::cfdlang::interface_defaults::inferAtomShape(
    ValueRange operands,
    teil::ShapeBuilder &atomShape
)
{
    LLVM_DEBUG(
        llvm::dbgs() << "mlir::cfdlang::interface_defaults::inferAtomShape({";
        llvm::interleaveComma(operands, llvm::dbgs());
        llvm::dbgs() << "}, _) -> ";
    );

    // Range of inferred shapes for all Atom opreands.
    const auto inferred = llvm::map_range(
        llvm::make_filter_range(
            operands,
            [](Value value) -> bool { return value.isa<Atom>(); }
        ),
        [](Value value) -> shape_t { return value.cast<Atom>().getShape(); }
    );

    if (inferred.empty()) {
        // Unable to infer shape without Atom operands.
        LLVM_DEBUG(llvm::dbgs() << "failure (no Atom operands)\n");
        return failure();
    }

    // Initialize the result shape.
    auto first = *inferred.begin();
    atomShape.assign(first.begin(), first.end());

    // Fold other shapes into result.
    for (auto part : llvm::drop_begin(inferred)) {
        if (failed(teil::fold(atomShape, part))) {
            // Inferred conflicting shapes.
            LLVM_DEBUG(llvm::dbgs() << "failure (conflicting shapes)\n");
            return failure();
        }
    }

    LLVM_DEBUG(
        llvm::dbgs() << "{\n";
        llvm::interleaveComma(atomShape, llvm::dbgs());
        llvm::dbgs() << "}\n";
    );
    return success();
}

FailureOr<teil::AtomSize> mlir::cfdlang::interface_defaults::reifyAtomSize(
    OpBuilder &builder,
    Operation* op
)
{
    // TODO: Implement.
    return failure();
}

LogicalResult mlir::cfdlang::interface_defaults::reifyResultShapes(
    teil::AtomSize atomSize,
    OpBuilder &builder,
    Location loc,
    ReifiedRankedShapedTypeDims &reifiedReturnShapes
)
{
    LLVM_DEBUG(
        llvm::dbgs() << "mlir::cfdlang::interface_defaults::reifyResultShapes({{";
        llvm::interleaveComma(atomSize.getShape(), llvm::dbgs());
        llvm::dbgs() << "}, {";
        llvm::interleaveComma(atomSize.getValues(), llvm::dbgs());
        llvm::dbgs() << "}}) -> ";
    );

    if (failed(atomSize.reify(builder, loc))) {
        LLVM_DEBUG(
            llvm::dbgs() << "failure (unable to reify atomSize)\n";
        );
        return failure();
    }

    reifiedReturnShapes.emplace_back(
        atomSize.getValues().begin(),
        atomSize.getValues().end()
    );
    LLVM_DEBUG(
        llvm::dbgs() << "{";
        llvm::interleaveComma(atomSize.getValues(), llvm::dbgs());
        llvm::dbgs() << "}\n";
    );
    return success();
}

//===- Generated implementation -------------------------------------------===//

//#include "mlir/Dialect/CFDlang/IR/AttrInterfaces.cpp.inc"
#include "mlir/Dialect/CFDlang/IR/OpInterfaces.cpp.inc"
//#include "mlir/Dialect/CFDlang/IR/TypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
