/// Implements the ImplicitCast helper.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace {

using namespace mlir;
using namespace mlir::ekl;

struct ImplicitCast {
protected:
    static LogicalResult rewriteOperands(
        PatternRewriter &rewriter,
        OperandRange operands,
        auto rewriteOperand)
    {
        if (operands.empty()) return failure();
        const auto op  = operands.getBase()->getOwner();
        const auto loc = op->getLoc();

        // Visit all operands with the rewriter.
        SmallVector<Value> newOperands(operands);
        auto update = false;
        for (auto &operand : newOperands)
            update |= succeeded(rewriteOperand(rewriter, loc, operand));

        if (!update) return failure();

        // Perform a bulk update on the target operation.
        rewriter.updateRootInPlace(op, [&]() {
            op->setOperands(
                operands.getBeginOperandIndex(),
                operands.size(),
                newOperands);
        });
        return success();
    }

    static LogicalResult
    unifyOperands(PatternRewriter &rewriter, OperandRange operands, Type type)
    {
        return rewriteOperands(
            rewriter,
            operands,
            [&](OpBuilder &builder, Location loc, Value &operand) {
                const auto opTy = llvm::cast<ExpressionType>(operand.getType())
                                      .getTypeBound();
                if (opTy == type) return failure();

                operand = builder.create<UnifyOp>(loc, operand, type);
                return success();
            });
    }

    static LogicalResult broadcastOperands(
        PatternRewriter &rewriter,
        OperandRange operands,
        ExtentRange extents)
    {
        const auto extentsAttr =
            BroadcastOp::getSignedExtentsAttr(rewriter.getContext(), extents);
        return rewriteOperands(
            rewriter,
            operands,
            [&](OpBuilder &builder, Location loc, Value &operand) {
                const auto opTy = llvm::cast<ExpressionType>(operand.getType())
                                      .getTypeBound();
                if (opTy && *getExtents(opTy) == extents) return failure();

                operand = builder.create<BroadcastOp>(
                    loc,
                    operand,
                    extentsAttr,
                    ArrayType::get(getScalarType(opTy), extents));
                return success();
            });
    }

    static LogicalResult broadcastAndUnifyOperands(
        PatternRewriter &rewriter,
        OperandRange operands,
        BroadcastType type)
    {
        auto updated = false;
        if (!llvm::isa<ScalarType>(type))
            updated |= succeeded(
                broadcastOperands(rewriter, operands, type.getExtents()));
        updated |= succeeded(unifyOperands(rewriter, operands, type));
        return success(updated);
    }
};

} // namespace
