/// Implements canonicalization and folding for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"
#include "messner/Dialect/EKL/Enums.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/EKL.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"

#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <optional>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// IntroOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult IntroOp::fold(FoldAdaptor adaptor)
{
    // If the input value is a compatible LiteralAttr, it is materialized by the
    // dialect. Otherwise, it will be passed along by the folder, but there is
    // no guarantee this op will be deleted.
    return adaptor.getValue();
}

//===----------------------------------------------------------------------===//
// EvalOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult EvalOp::fold(FoldAdaptor adaptor)
{
    if (!isFullyTyped()) return {};

    if (auto intro = getOperand().getDefiningOp<IntroOp>()) {
        if (intro.getOperand().getType() == getResult().getType()) {
            // eval(intro(x : T) : T) = x
            return intro.getOperand();
        }
    }

    // Since the result type of the op is not an ExpressionType, the dialect
    // constant materializer will not be able to materialize any attribute
    // returned by this operation.
    return adaptor.getExpression();
}

//===----------------------------------------------------------------------===//
// StaticOp implementation
//===----------------------------------------------------------------------===//

LogicalResult StaticOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &)
{
    // Remove the initializer attribute of a write-only local variable.
    if (getInitializerAttr() && !isPublic() && isOwned() && !isReadable()) {
        removeInitializerAttr();
        return success();
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// IfOp implementation
//===----------------------------------------------------------------------===//

LogicalResult IfOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &results)
{
    // Folding only applies to fully typed if expressions.
    if (!getResult() || !isFullyTyped()) return failure();

    // Eliminate the operation if both branches yield the same value.
    const auto thenValue = getThenExpression();
    if (thenValue == getElseExpression()) {
        results.emplace_back(thenValue);
        return success();
    }

    return failure();
}

namespace {

struct InlineIf : OpRewritePattern<IfOp> {
    using OpRewritePattern<IfOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "requires concrete types");

        // Determine if there is a statically known code path.
        BoolAttr condition;
        if (!matchPattern(op.getCondition(), m_Constant(&condition)))
            return rewriter.notifyMatchFailure(
                op,
                "requires statically known condition");

        // Inline the statically known code path before the IfOp.
        auto branch =
            condition.getValue() ? op.getThenBranch() : op.getElseBranch();
        auto yield = llvm::cast<YieldOp>(branch->getTerminator());
        rewriter.inlineBlockBefore(branch, op);

        // Since the yield type may be a subtype of the result type, unify.
        rewriter.setInsertionPoint(yield);
        const auto unify = rewriter.create<UnifyOp>(
            yield->getLoc(),
            yield.getExpression(),
            op.getType(0));

        // Erase the extraneous terminator and replace the IfOp.
        rewriter.eraseOp(yield);
        rewriter.replaceOp(op, unify);
        return success();
    }
};

struct RewriteIfToStatement : OpRewritePattern<IfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(IfOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isExpression())
            return rewriter.notifyMatchFailure(op, "requires if expression");
        if (!op.getResult().use_empty())
            return rewriter.notifyMatchFailure(op, "requires unused result");

        // Since we can't erase a result, we need to create a new IfOp.
        auto result = rewriter.create<IfOp>(op.getLoc(), op.getCondition());

        // Move the branches via inlining.
        rewriter.inlineBlockBefore(
            op.getThenBranch(),
            result.getThenBranch(),
            result.getThenBranch()->end());
        rewriter.inlineBlockBefore(
            op.getElseBranch(),
            result.getElseBranch(),
            result.getElseBranch()->end());

        // Erase the extraneous terminators and the old IfOp.
        rewriter.eraseOp(result.getThenBranch()->getTerminator());
        rewriter.eraseOp(result.getElseBranch()->getTerminator());
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void IfOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<InlineIf, RewriteIfToStatement>(context);
}

//===----------------------------------------------------------------------===//
// SubscriptOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult SubscriptOp::fold(FoldAdaptor adaptor)
{
    // Only applies to speculatable subscripts.
    if (!isSpeculatable(*this)) return {};

    // Fold away subscripts into broadcasted scalars.
    if (auto bcast = getArray().getDefiningOp<BroadcastOp>(); bcast)
        if (llvm::isa_and_present<ScalarType>(
                bcast.getOperand().getType().getTypeBound()))
            return bcast.getOperand();

    // Fold away no-op subscripts.
    if (getSubscripts().empty() && getType() == getArray().getType())
        return getArray();

    // Must have constant array.
    const auto array =
        llvm::dyn_cast_if_present<ekl::ArrayAttr>(adaptor.getArray());
    if (!array) return {};

    const auto bounds = array.getType().getExtents();
    if (adaptor.getSubscripts().size() > bounds.size()) return {};

    // Must be constant index values only.
    SmallVector<extent_t> indices;
    for (auto [attr, bound] :
         llvm::zip_first(adaptor.getSubscripts(), bounds)) {
        const auto indexAttr = llvm::dyn_cast_if_present<ekl::IndexAttr>(attr);
        if (!indexAttr || indexAttr.getValue() >= bound) return {};
        indices.push_back(indexAttr.getValue());
    }

    // Perform the subscript operation.
    return array.subscript(indices);
}

namespace {

struct ExpandEllipsisSubscript : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "requires concrete types");

        // Find the ellipsis operand, if any.
        const auto it = llvm::find_if(op.getSubscripts(), [](Value value) {
            return llvm::isa<EllipsisType>(
                llvm::cast<Expression>(value).getType().getTypeBound());
        });
        if (it == op.getSubscripts().end())
            return rewriter.notifyMatchFailure(op, "requires ellipsis");

        // Calculate the number of identities it should expand to.
        const auto extentExprTy = ExpressionType::get(
            rewriter.getContext(),
            ExtentType::get(rewriter.getContext()));
        // How many extents do we need to index?
        const auto numExtents =
            llvm::cast<ArrayType>(op.getArray().getType().getTypeBound())
                .getNumExtents();
        // How many extents did we insert?
        const auto numExpanded = std::size_t(
            llvm::count(op.getSubscripts().getTypes(), extentExprTy));
        const auto expand =
            numExtents - op.getSubscripts().size() + 1UL + numExpanded;

        // Create an instance of the identity literal. We use the location of
        // the ellipsis literal to track where they came from.
        auto idLiteral = rewriter.create<LiteralOp>(
            (*it).getLoc(),
            IdentityAttr::get(rewriter.getContext()));
        SmallVector<Value> identities(expand, idLiteral.getResult());

        // Replace the single ellipsis operand with that many identity literals.
        rewriter.modifyOpInPlace(op, [&]() {
            op->setOperands(
                op.getSubscripts().getBeginOperandIndex(),
                1U,
                identities);
        });
        return success();
    }
};

struct MergeSubscripts : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp suffix, PatternRewriter &rewriter) const final
    {
        // Find the subscript prefix.
        auto prefix = suffix.getArray().getDefiningOp<SubscriptOp>();
        if (!prefix)
            return rewriter.notifyMatchFailure(
                suffix,
                "requires prefix subscript");
        if (!suffix.isFullyTyped() || !prefix.isFullyTyped())
            return rewriter.notifyMatchFailure(
                suffix,
                "requires concrete types");

        // Ensure concatenation would not yield too many ellipses.
        const auto countEllipses = [](SubscriptOp op) {
            const auto isEllipsis = [](Type type) {
                return llvm::isa_and_present<EllipsisType>(getTypeBound(type));
            };
            return llvm::count_if(op.getSubscripts().getTypes(), isEllipsis);
        };
        if (countEllipses(prefix) + countEllipses(suffix) > 1)
            return rewriter.notifyMatchFailure(
                suffix,
                "requires at most one ellipsis");

        // Create a new subscript by concatenating the prefix with suffix.
        const auto indices = llvm::to_vector(llvm::concat<Value>(
            prefix.getSubscripts(),
            suffix.getSubscripts()));
        rewriter.replaceOpWithNewOp<SubscriptOp>(
            suffix,
            prefix.getArray(),
            indices,
            suffix.getType().getTypeBound());
        return success();
    }
};

struct InlineBroadcastSubscript : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "requires concrete types");
        auto bcast = op.getArray().getDefiningOp<BroadcastOp>();
        if (!bcast)
            return rewriter.notifyMatchFailure(
                op,
                "requires broadcast operand");
        const auto inTy = llvm::dyn_cast_if_present<ArrayType>(
            bcast.getOperand().getType().getTypeBound());
        if (!inTy)
            return rewriter.notifyMatchFailure(op, "requires concrete type");

        // Directly index into the broadcasted operand.
        rewriter.modifyOpInPlace(op, [&]() {
            op.setOperand(0, bcast.getOperand());
        });

        // All size 1 input dimensions must be pinned.
        SmallVector<OpOperand *> pin;
        for (unsigned idx = 0; idx < op.getSubscripts().size(); ++idx)
            if (inTy.getExtent(idx) == 1)
                pin.push_back(&op->getOpOperand(1U + idx));
        if (!pin.empty()) {
            auto literal = rewriter.create<LiteralOp>(
                op.getLoc(),
                ekl::IndexAttr::get(getContext(), 0));
            rewriter.modifyOpInPlace(op, [&]() {
                for (auto opd : pin) opd->set(literal);
            });
        }

        // Fix partial subscripting results.
        if (const auto arrayTy =
                llvm::dyn_cast<ArrayType>(op.getType().getTypeBound());
            arrayTy) {
            rewriter.setInsertionPointAfter(op);
            auto sunken = rewriter.create<BroadcastOp>(
                bcast.getLoc(),
                op.getResult(),
                arrayTy);
            rewriter.replaceAllUsesExcept(op, sunken, sunken);
            rewriter.modifyOpInPlace(op, [&]() {
                op.getResult().setType(ExpressionType::get(
                    getContext(),
                    inTy.cloneWith(
                        inTy.getExtents().take_back(arrayTy.getNumExtents()))));
            });
        }
        return success();
    }
};

struct PinSubscripts : OpRewritePattern<SubscriptOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(SubscriptOp op, PatternRewriter &rewriter) const final
    {
        // Collect trivial 0 indexers.
        SmallVector<OpOperand *> pin;
        for (auto &opd : op->getOpOperands().drop_front()) {
            const auto indexTy = llvm::dyn_cast_if_present<ekl::IndexType>(
                getTypeBound(opd.get()));
            if (!indexTy || indexTy.getUpperBound() != 0) continue;
            if (matchPattern(opd.get(), m_Constant())) continue;
            pin.push_back(&opd);
        }
        if (pin.empty())
            return rewriter.notifyMatchFailure(
                op,
                "requires known 0 subscripts");

        auto literal = rewriter.create<LiteralOp>(
            op.getLoc(),
            ekl::IndexAttr::get(getContext(), 0));
        rewriter.modifyOpInPlace(op, [&]() {
            for (auto opd : pin) opd->set(literal);
        });
        return success();
    }
};

} // namespace

void SubscriptOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<
        ExpandEllipsisSubscript,
        MergeSubscripts,
        InlineBroadcastSubscript,
        PinSubscripts>(context);
}

//===----------------------------------------------------------------------===//
// StackOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult StackOp::fold(FoldAdaptor adaptor)
{
    // Only applies to stacks with known result types.
    const auto arrayTy =
        llvm::cast_if_present<ArrayType>(getType().getTypeBound());
    if (!arrayTy) return {};

    for (auto &op : getOperandsMutable()) {
        // Short-circuit any no-op subscript operands to their arrays.
        if (auto prior = op.get().getDefiningOp<SubscriptOp>();
            prior && prior.getSubscripts().empty())
            op.set(prior.getArray());

        // Short-circuit any 0-dim broadcast operands to their scalars.
        if (auto prior = op.get().getDefiningOp<BroadcastOp>();
            prior
            && llvm::isa_and_present<ScalarType>(
                prior.getOperand().getType().getTypeBound()))
            op.set(prior.getOperand());
    }

    // All operands must be constant.
    if (llvm::count(adaptor.getOperands(), Attribute{}) > 0) return {};

    // Stack easily by exploiting the representation of ekl::ArrayAttr.
    return ekl::ArrayAttr::get(arrayTy, adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// AssocOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult AssocOp::fold(FoldAdaptor)
{
    // Only applies to fully typed assoc expressions.
    if (!isFullyTyped()) return {};

    // If the yielded expression was folded to a scalar, a splat can be derived.
    const auto expr = getMapExpression();
    ScalarAttr value;
    if (!matchPattern(expr, m_Constant(&value))) return {};
    return ekl::ArrayAttr::get(
        llvm::cast<ArrayType>(getType().getTypeBound()),
        value);
}

namespace {

struct InlineAssoc : OpRewritePattern<AssocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(AssocOp op, PatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!arrayTy || arrayTy.getNumExtents() > 0)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete 0-dim array type");

        // Inline the assoc expression body in-place.
        auto yield = llvm::cast<YieldOp>(op.getMap()->getTerminator());
        rewriter.inlineBlockBefore(op.getMap(), op);

        // Replace the assoc expression by unifying to the 0-dim array.
        rewriter.replaceOpWithNewOp<UnifyOp>(op, yield->getOperand(0), arrayTy);

        // Remove the extraneous terminator.
        rewriter.eraseOp(yield);
        return success();
    }
};

struct RewriteAssocToBroadcast : OpRewritePattern<AssocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(AssocOp op, PatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!arrayTy)
            return rewriter.notifyMatchFailure(op, "requires concrete types");
        if (!op.getMapExpression().getParentRegion()->isProperAncestor(
                &op.getMapRegion()))
            return rewriter.notifyMatchFailure(
                op,
                "requires invariant map expression");

        // Convert the atom to the right type.
        const auto atomType = llvm::cast_if_present<BroadcastType>(
            op.getMapExpression().getType().getTypeBound());
        auto atom = rewriter.create<UnifyOp>(
            op.getLoc(),
            op.getMapExpression(),
            atomType.cloneWith(arrayTy.getScalarType()));

        // Keep stacking the atom until it has the right rank.
        auto extents = llvm::to_vector(atomType.getExtents());
        auto array   = atom.getResult();
        while (extents.size() < arrayTy.getNumExtents()) {
            extents.insert(extents.begin(), 1);
            auto stack = rewriter.create<StackOp>(
                op.getLoc(),
                array,
                arrayTy.cloneWith(extents));
            array = stack.getResult();
        }

        // Broadcast it to the result shape.
        rewriter.replaceOpWithNewOp<BroadcastOp>(op, array, arrayTy);
        return success();
    }
};

} // namespace

void AssocOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<InlineAssoc, RewriteAssocToBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// ZipOp implementation
//===----------------------------------------------------------------------===//

namespace {

struct InlineZip : OpRewritePattern<ZipOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        const auto broadcastTy =
            llvm::cast_if_present<BroadcastType>(op.getType().getTypeBound());
        if (!broadcastTy)
            return rewriter.notifyMatchFailure(op, "requires concrete types");

        SmallVector<OpOperand *> decay;
        if (!llvm::all_of(op->getOpOperands(), [&](OpOperand &opd) {
                const auto inTy = getTypeBound(opd.get());
                if (!inTy) return false;
                if (llvm::isa<ScalarType>(inTy)) return true;
                const auto arrayTy = llvm::cast<ArrayType>(inTy);
                if (arrayTy.getNumExtents() != 0) return false;
                decay.push_back(&opd);
                return true;
            }))
            return rewriter.notifyMatchFailure(op, "requires 0-dim operands");

        // Decay 0-dim arrays to scalars.
        for (auto opd : decay) {
            auto subscript = rewriter.create<SubscriptOp>(
                op.getLoc(),
                opd->get(),
                ValueRange{},
                getScalarType(getTypeBound(opd->get())));
            opd->set(subscript);
        }

        // Inline the zip expression body in-place.
        auto yield = llvm::cast<YieldOp>(op.getCombinator()->getTerminator());
        rewriter.inlineBlockBefore(op.getCombinator(), op, op->getOperands());

        // Replace the zip expression by unifying to the result type.
        rewriter.replaceOpWithNewOp<UnifyOp>(
            op,
            yield->getOperand(0),
            broadcastTy);

        // Remove the extraneous terminator.
        rewriter.eraseOp(yield);
        return success();
    }
};

struct RewriteZipToBroadcast : OpRewritePattern<ZipOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::dyn_cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!arrayTy)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete array type");

        // Find the argument that is being forwarded.
        const auto findResult = [&](Value expr) -> FailureOr<Value> {
            if (expr.getParentRegion()->isProperAncestor(
                    &op.getCombinatorRegion()))
                return expr;
            for (auto arg : op.getCombinator()->getArguments())
                if (expr == arg) return op->getOperand(arg.getArgNumber());
            return failure();
        };
        const auto maybeResult = findResult(op.getCombinatorExpression());
        if (failed(maybeResult))
            return rewriter.notifyMatchFailure(op, "requires trivial body");

        const auto invariantTy =
            llvm::cast_if_present<BroadcastType>(getTypeBound(*maybeResult));
        if (!invariantTy)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete invariant type");

        // Broadcast to the right shape and then unify to the right type.
        auto bcast = rewriter.create<BroadcastOp>(
            op.getLoc(),
            *maybeResult,
            arrayTy.cloneWith(invariantTy.getScalarType()));
        rewriter.replaceOpWithNewOp<UnifyOp>(op, bcast.getResult(), arrayTy);
        return success();
    }
};

struct UniqueZipOperands : OpRewritePattern<ZipOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ZipOp op, PatternRewriter &rewriter) const final
    {
        llvm::DenseMap<Value, OpOperand *> uniquer;
        llvm::DenseMap<OpOperand *, OpOperand *> remap;
        for (auto &opd : op->getOpOperands()) {
            const auto [it, added] = uniquer.try_emplace(opd.get(), &opd);
            if (!added) remap.insert(std::make_pair(&opd, it->second));
        }
        if (remap.empty())
            return rewriter.notifyMatchFailure(
                op,
                "requires duplicate operands");

        rewriter.modifyOpInPlace(op, [&]() {
            for (auto [from, to] : remap) {
                const auto oldIdx = from->getOperandNumber();
                rewriter.replaceAllUsesWith(
                    op.getCombinator()->getArgument(oldIdx),
                    op.getCombinator()->getArgument(to->getOperandNumber()));
                op->eraseOperand(oldIdx);
                op.getCombinator()->eraseArgument(oldIdx);
            }
        });
        return success();
    }
};

} // namespace

void ZipOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<InlineZip, RewriteZipToBroadcast, UniqueZipOperands>(context);
}

//===----------------------------------------------------------------------===//
// ConstexprOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult ConstexprOp::fold(FoldAdaptor)
{
    // Only applies to fully typed constant expressions.
    if (!isFullyTyped()) return {};

    // Fold to the constant expression value.
    LiteralAttr literal;
    if (matchPattern(getExpression(), m_Constant(&literal))) return literal;
    return {};
}

//===----------------------------------------------------------------------===//
// UnifyOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult UnifyOp::fold(FoldAdaptor adaptor)
{
    // Only applies to fully typed casts.
    // NOTE: The CastOpInterface folder already folds away no-op casts.
    if (!isFullyTyped()) return {};

    // The subtype relation is transitive, so unification casts are as well.
    if (auto prior = getOperand().getDefiningOp<UnifyOp>(); prior) {
        setOperand(prior.getOperand());
        return getResult();
    }

    // Implicitly short-circuit any scalar decay from a no-op subscript.
    if (auto prior = getOperand().getDefiningOp<SubscriptOp>();
        prior && prior.getSubscripts().empty()
        && llvm::isa<ArrayType>(getType().getTypeBound())) {
        setOperand(prior.getArray());
        return getResult();
    }

    // Attributes are covariant in the IR, no unification happens. The
    // materializer will produce a LiteralOp with a different type.
    return adaptor.getOperand();
}

namespace {

template<class Cast>
struct HoistCastBeforeBroadcast : OpRewritePattern<Cast> {
    using OpRewritePattern<Cast>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(Cast op, PatternRewriter &rewriter) const final
    {
        const auto outTy =
            llvm::dyn_cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!outTy)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete array type");
        auto bcast = op.getOperand().template getDefiningOp<BroadcastOp>();
        if (!bcast)
            return rewriter.notifyMatchFailure(
                op,
                "requires broadcast operand");
        const auto inTy = llvm::dyn_cast_if_present<BroadcastType>(
            bcast.getOperand().getType().getTypeBound());
        if (!inTy)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete broadcast input type");

        // Create a sunken BroadcastOp and change the order of casts.
        auto sink =
            rewriter.create<BroadcastOp>(bcast.getLoc(), op.getResult(), outTy);
        rewriter.replaceAllUsesExcept(op, sink, sink);
        rewriter.modifyOpInPlace(op, [&]() {
            op.setOperand(bcast.getOperand());
            op.getResult().setType(ExpressionType::get(
                rewriter.getContext(),
                inTy.cloneWith(outTy.getScalarType())));
        });
        return success();
    }
};

struct RewriteUnifyToBroadcast : OpRewritePattern<UnifyOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(UnifyOp op, PatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::dyn_cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!arrayTy || arrayTy.getNumExtents() > 0)
            return rewriter.notifyMatchFailure(op, "requires 0-dim array type");
        if (!llvm::isa<ScalarType>(op.getOperand().getType().getTypeBound()))
            return rewriter.notifyMatchFailure(op, "requires scalar operand");

        auto bcast =
            rewriter.create<BroadcastOp>(op.getLoc(), op.getResult(), arrayTy);
        rewriter.replaceAllUsesExcept(op, bcast, bcast);
        rewriter.modifyOpInPlace(op, [&]() {
            op.getResult().setType(
                ExpressionType::get(getContext(), arrayTy.getScalarType()));
        });
        return success();
    }
};

} // namespace

void UnifyOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<HoistCastBeforeBroadcast<UnifyOp>, RewriteUnifyToBroadcast>(
        context);
}

//===----------------------------------------------------------------------===//
// BroadcastOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor)
{
    // Only applies to fully typed casts.
    // NOTE: The CastOpInterface folder already folds away no-op casts.
    if (!isFullyTyped()) return {};

    // The compatibility relation is transitive, so broadcasting is as well.
    if (auto prior = getOperand().getDefiningOp<BroadcastOp>(); prior) {
        setOperand(prior.getOperand());
        return getResult();
    }

    // Only applies to constant operands.
    if (!adaptor.getOperand()) return {};
    const auto resultTy = llvm::cast<ArrayType>(getType().getTypeBound());

    return llvm::TypeSwitch<Attribute, Attribute>(adaptor.getOperand())
        .Case([&](ScalarAttr scalar) {
            return ArrayAttr::get(resultTy, {scalar});
        })
        .Case([&](ekl::ArrayAttr array) {
            return array.broadcastTo(resultTy.getExtents());
        })
        .Default([](auto) -> Attribute { return {}; });
}

//===----------------------------------------------------------------------===//
// CoerceOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult CoerceOp::fold(FoldAdaptor adaptor)
{
    // Only applies to fully typed casts.
    // NOTE: The CastOpInterface folder already folds away no-op casts.
    if (!isSpeculatable(*this)) return {};

    // Only applies to constant operands.
    if (!adaptor.getOperand()) return {};

    return llvm::TypeSwitch<Type, Attribute>(getType().getTypeBound())
        .Case([&](ekl::ArrayType arrayTy) {
            return coerce(
                llvm::cast<ekl::ArrayAttr>(adaptor.getOperand()),
                arrayTy.getScalarType());
        })
        .Case([&](ScalarType scalarTy) {
            return coerce(
                llvm::cast<ekl::ScalarAttr>(adaptor.getOperand()),
                scalarTy);
        })
        .Default([](auto) -> Attribute { return {}; });
}

namespace {

struct RewriteCoerceToUnify : OpRewritePattern<CoerceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(CoerceOp op, PatternRewriter &rewriter) const final
    {
        if (!op.isFullyTyped())
            return rewriter.notifyMatchFailure(op, "requires concrete types");

        const auto inTy  = op.getOperand().getType().getTypeBound();
        const auto outTy = op.getType().getTypeBound();
        if (!isSubtype(inTy, outTy))
            return rewriter.notifyMatchFailure(op, "requires subtype");

        rewriter.replaceOpWithNewOp<UnifyOp>(op, op.getOperand(), outTy);
        return success();
    }
};

} // namespace

void CoerceOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<HoistCastBeforeBroadcast<CoerceOp>, RewriteCoerceToUnify>(
        context);
}

//===----------------------------------------------------------------------===//
// ChoiceOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult ChoiceOp::fold(FoldAdaptor)
{
    // not c ? a : b = c ? b : a
    if (auto lnot = getSelector().getDefiningOp<LogicalNotOp>(); lnot) {
        (*this)->setOperands(
            {lnot.getOperand(), getAlternatives()[1], getAlternatives()[0]});
        return getResult();
    }

    // TODO: Implement general folding.
    return {};
}

namespace {

struct RewriteChoiceToBroadcast : OpRewritePattern<ChoiceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ChoiceOp op, PatternRewriter &rewriter) const final
    {
        if (!llvm::isa<ScalarType>(op.getSelector().getType().getTypeBound()))
            return rewriter.notifyMatchFailure(op, "requires scalar selector");
        const auto resultTy =
            llvm::cast_if_present<BroadcastType>(op.getType().getTypeBound());
        if (!resultTy)
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete result type");

        const auto fold = [&](Value alternative) {
            // Since we only apply to scalar selectors, no extension can happen
            // on the alternatives.
            assert(
                resultTy.getExtents().size()
                == llvm::cast<BroadcastType>(getTypeBound(alternative))
                       .getExtents()
                       .size());

            if (llvm::isa<ScalarType>(resultTy)) {
                rewriter.replaceOp(op, alternative);
                return;
            }

            rewriter.replaceOpWithNewOp<BroadcastOp>(
                op,
                alternative,
                llvm::cast<ArrayType>(resultTy));
        };

        // Try to fold a constant selector.
        ScalarAttr selector;
        if (matchPattern(op.getSelector(), m_Constant(&selector))) {
            if (const auto i1 = llvm::dyn_cast<BoolAttr>(selector); i1) {
                fold(op.getAlternatives()[i1.getValue()]);
                return success();
            }
            if (const auto index = llvm::dyn_cast<IndexAttr>(selector); index) {
                fold(op.getAlternatives()[index.getValue()]);
                return success();
            }
        }

        // Try to fold a constant alternative.
        auto alternative = op.getAlternatives().front();
        if (static_cast<size_t>(llvm::count(op.getAlternatives(), alternative))
            == op.getAlternatives().size()) {
            fold(alternative);
            return success();
        }

        return rewriter.notifyMatchFailure(op, "requires known alternative");
    }
};

struct DecayChoiceToScalar : OpRewritePattern<ChoiceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ChoiceOp op, PatternRewriter &rewriter) const final
    {
        const auto arrayTy =
            llvm::dyn_cast_if_present<ArrayType>(op.getType().getTypeBound());
        if (!arrayTy || arrayTy.getNumExtents() != 0)
            return rewriter.notifyMatchFailure(
                op,
                "requires 0-dim result array");

        rewriter.setInsertionPointAfter(op);
        auto bcast =
            rewriter.create<BroadcastOp>(op.getLoc(), op.getResult(), arrayTy);
        rewriter.replaceAllUsesExcept(op, bcast, bcast);
        rewriter.modifyOpInPlace(op, [&]() {
            op.getResult().setType(
                ExpressionType::get(getContext(), arrayTy.getScalarType()));
        });
        return success();
    }
};

} // namespace

void ChoiceOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<RewriteChoiceToBroadcast, DecayChoiceToScalar>(context);
}

//===----------------------------------------------------------------------===//
// CompareOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult CompareOp::fold(FoldAdaptor adaptor)
{
    // Only applies to comparison with a concrete result type.
    const auto resultTy =
        llvm::cast_if_present<BroadcastType>(getType().getTypeBound());
    if (!resultTy) return {};

    const auto makeSplat = [&](bool value) -> Attribute {
        auto scalar = BoolAttr::get(getContext(), value);
        if (llvm::isa<ScalarType>(resultTy)) return scalar;
        return ArrayAttr::get(scalar, resultTy.getExtents());
    };

    auto kind = getKind();

    // Fold trivial comparisons.
    if (getLhs() == getRhs()) {
        switch (kind) {
        case RelationKind::Equivalent:
        case RelationKind::GreaterOrEqual:
        case RelationKind::LessOrEqual:
            // Trivial tautology.
            return makeSplat(true);

        case RelationKind::Antivalent:
        case RelationKind::GreaterThan:
        case RelationKind::LessThan:
            // Trivial contradiction.
            return makeSplat(false);
        }
    }

    // Find some static knowledge.
    auto lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    if (!lhs) {
        using std::swap;
        swap(lhs, rhs);
        kind = flip(kind);
    }
    // We require some static knowledge.
    if (!lhs) return {};

    // TODO: Implement general folding.
    return {};
}

namespace {

struct NegateComparison : OpRewritePattern<CompareOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(CompareOp op, PatternRewriter &rewriter) const final
    {
        if (!op->hasOneUse())
            return rewriter.notifyMatchFailure(op, "requires single use");
        auto lnot = llvm::dyn_cast<LogicalNotOp>(*op->user_begin());
        if (!lnot)
            return rewriter.notifyMatchFailure(op, "requires logical not user");

        rewriter.modifyOpInPlace(op, [&]() {
            op.setKind(negate(op.getKind()));
        });
        rewriter.replaceOp(lnot, op);
        return success();
    }
};

} // namespace

void CompareOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<NegateComparison>(context);
}

//===----------------------------------------------------------------------===//
// Logical operator implementation
//===----------------------------------------------------------------------===//

OpFoldResult LogicalNotOp::fold(FoldAdaptor)
{
    // not not a = a
    if (auto source = getOperand().getDefiningOp<LogicalNotOp>(); source)
        return source.getOperand();

    return {};
}

OpFoldResult LogicalOrOp::fold(FoldAdaptor)
{
    // a \/ a = a
    if (getLhs() == getRhs()) return getLhs();

    // TODO: Implement general folding.
    return {};
}

OpFoldResult LogicalAndOp::fold(FoldAdaptor)
{
    // a /\ a = a
    if (getLhs() == getRhs()) return getLhs();

    // TODO: Implement general folding.
    return {};
}

//===----------------------------------------------------------------------===//
// Arithmetic operator implementation
//===----------------------------------------------------------------------===//

OpFoldResult NegateOp::fold(FoldAdaptor)
{
    // - - a = a
    if (auto source = getOperand().getDefiningOp<NegateOp>(); source)
        return source.getOperand();

    // TODO: Implement general folding.
    return {};
}

template<class Fn>
[[nodiscard]]
static Attribute foldIndexArith(Fn &fn, Attribute lhs, Attribute rhs)
{
    const auto lhsAttr = llvm::dyn_cast_if_present<ekl::IndexAttr>(lhs);
    const auto rhsAttr = llvm::dyn_cast_if_present<ekl::IndexAttr>(rhs);
    if (!lhsAttr || !rhsAttr) return {};

    if (const auto maybeResult = fn(lhsAttr.getValue(), rhsAttr.getValue());
        maybeResult)
        return ekl::IndexAttr::get(lhs.getContext(), *maybeResult);
    return {};
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        extent_t result = 0;
        if (__builtin_add_overflow(lhs, rhs, &result)) return std::nullopt;
        return result;
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

OpFoldResult SubtractOp::fold(FoldAdaptor adaptor)
{
    // Only applies to concretely typed subtractions.
    const auto resultTy =
        llvm::cast_if_present<BroadcastType>(getType().getTypeBound());
    if (!resultTy) return {};

    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        extent_t result = 0;
        if (__builtin_sub_overflow(lhs, rhs, &result)) return std::nullopt;
        return result;
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

OpFoldResult MultiplyOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        extent_t result = 0;
        if (__builtin_mul_overflow(lhs, rhs, &result)) return std::nullopt;
        return result;
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

OpFoldResult DivideOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        if (rhs == 0) return std::nullopt;
        return lhs / rhs;
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

OpFoldResult RemainderOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        if (rhs == 0) return std::nullopt;
        return lhs % rhs;
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

//===----------------------------------------------------------------------===//
// MinOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult MinOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        return std::min(lhs, rhs);
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}

//===----------------------------------------------------------------------===//
// MaxOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult MaxOp::fold(FoldAdaptor adaptor)
{
    const auto indexFn = [](extent_t lhs,
                            extent_t rhs) -> std::optional<extent_t> {
        return std::max(lhs, rhs);
    };

    // TODO: Implement general folding.
    return foldIndexArith(indexFn, adaptor.getLhs(), adaptor.getRhs());
}
