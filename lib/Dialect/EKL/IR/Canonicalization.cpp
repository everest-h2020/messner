/// Implements canonicalization and folding for the EKL dialect.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/Analysis/AbstractTypeChecker.h"
#include "messner/Dialect/EKL/Enums.h"
#include "messner/Dialect/EKL/IR/Attributes.h"
#include "messner/Dialect/EKL/IR/Ops.h"
#include "messner/Dialect/EKL/IR/TypeUtils.h"
#include "messner/Dialect/EKL/IR/Types.h"

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

OpFoldResult IntroOp::fold(IntroOp::FoldAdaptor adaptor)
{
    // If the input value is a compatible LiteralAttr, it is materialized by the
    // dialect. Otherwise, it will be passed along by the folder, but there is
    // no guarantee this op will be deleted.
    return adaptor.getValue();
}

//===----------------------------------------------------------------------===//
// EvalOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult EvalOp::fold(EvalOp::FoldAdaptor adaptor)
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

OpFoldResult SubscriptOp::fold(SubscriptOp::FoldAdaptor adaptor)
{
    // Only applies to speculatable subscripts.
    if (!isSpeculatable(*this)) return {};

    // Fold away empty subscripts.
    if (getSubscripts().empty()) return getArray();

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

} // namespace

void SubscriptOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<MergeSubscripts>(context);
}

//===----------------------------------------------------------------------===//
// StackOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult StackOp::fold(StackOp::FoldAdaptor adaptor)
{
    // Only applies to stacks with known result types.
    const auto arrayTy =
        llvm::dyn_cast_if_present<ArrayType>(getType().getTypeBound());
    if (!arrayTy) return {};

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

} // namespace

void AssocOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<InlineAssoc>(context);
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
        if (!broadcastTy || !broadcastTy.getExtents().empty())
            return rewriter.notifyMatchFailure(
                op,
                "requires concrete 0-dim result type");

        // Inline the zip expression body in-place.
        auto yield = llvm::cast<YieldOp>(op.getCombinator()->getTerminator());
        rewriter.inlineBlockBefore(op.getCombinator(), op, op->getOperands());

        // Replace the zip expression by unifying to the 0-dim result type.
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
        const auto findResult = [&](Value expr) -> FailureOr<unsigned> {
            for (auto arg : op.getCombinator()->getArguments())
                if (expr == arg) return arg.getArgNumber();
            return failure();
        };
        const auto maybeResult = findResult(op.getCombinatorExpression());
        if (failed(maybeResult))
            return rewriter.notifyMatchFailure(op, "requires trivial body");

        // Replace the zip expression with a simple BroadcastOp.
        rewriter.replaceOpWithNewOp<BroadcastOp>(
            op,
            op->getOperand(*maybeResult),
            arrayTy);
        return success();
    }
};

} // namespace

void ZipOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<InlineZip, RewriteZipToBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// ConstexprOp implementation
//===----------------------------------------------------------------------===//

OpFoldResult ConstexprOp::fold(ConstexprOp::FoldAdaptor)
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

    // Attributes are covariant in the IR, no unification happens. The
    // materializer will produce a LiteralOp with a different type.
    return adaptor.getOperand();
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

[[nodiscard]] static ekl::IntegerAttr
coerce(ScalarAttr input, ekl::IntegerType output)
{
    return llvm::TypeSwitch<ScalarAttr, ekl::IntegerAttr>(input)
        .Case([&](NumberAttr attr) {
            auto value = attr.getValue();
            value.roundTowardsZero();
            return ::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(value.getMantissa(), true)),
                output);
        })
        .Case([&](ekl::IntegerAttr attr) {
            auto value    = attr.getValue();
            auto adjValue = attr.getType().isSigned()
                              ? value.sextOrTrunc(output.getWidth())
                              : value.zextOrTrunc(output.getWidth());
            return ekl::IntegerAttr::get(
                input.getContext(),
                llvm::APSInt(adjValue, output.isUnsigned()));
        })
        .Case([&](FloatAttr attr) {
            llvm::APSInt result(output.getWidth(), output.isUnsigned());
            bool isExact;
            attr.getValue().convertToInteger(
                result,
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &isExact);
            return ekl::IntegerAttr::get(attr.getContext(), result);
        })
        .Case([&](ekl::IndexAttr attr) {
            return ::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(llvm::APInt(64U, attr.getValue()), true)),
                output);
        })
        .Default(ekl::IntegerAttr{});
}

[[nodiscard]] static FloatAttr coerce(ScalarAttr input, FloatType output)
{
    return llvm::TypeSwitch<ScalarAttr, FloatAttr>(input)
        .Case([&](NumberAttr attr) {
            return FloatAttr::get(
                output,
                attr.getValue().toAPFloatWithRounding(
                    const_cast<llvm::fltSemantics &>(
                        output.getFloatSemantics())));
        })
        .Case([&](ekl::IntegerAttr attr) {
            llvm::APFloat value(output.getFloatSemantics());
            value.convertFromAPInt(
                attr.getValue(),
                attr.getType().isSigned(),
                llvm::APFloat::roundingMode::NearestTiesToEven);
            return FloatAttr::get(output, value);
        })
        .Case([&](FloatAttr attr) {
            auto value = attr.getValue();
            bool losesInfo;
            value.convert(
                output.getFloatSemantics(),
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &losesInfo);
            return FloatAttr::get(output, value);
        })
        .Case([&](ekl::IndexAttr attr) {
            return FloatAttr::get(output, static_cast<double>(attr.getValue()));
        })
        .Default(FloatAttr{});
}

[[nodiscard]] static ekl::IndexAttr
coerce(ScalarAttr input, ekl::IndexType output)
{
    return llvm::TypeSwitch<ScalarAttr, ekl::IndexAttr>(input)
        .Case([&](NumberAttr attr) {
            auto value = attr.getValue();
            value.roundTowardsZero();
            return ::coerce(
                ekl::IntegerAttr::get(
                    input.getContext(),
                    llvm::APSInt(value.getMantissa(), true)),
                output);
        })
        .Case([&](ekl::IntegerAttr attr) {
            auto value = attr.getValue();
            if (value.getActiveBits() > 64U) return ekl::IndexAttr{};
            const auto intValue = value.getZExtValue();
            if (intValue > output.getUpperBound()) return ekl::IndexAttr{};
            return ekl::IndexAttr::get(input.getContext(), intValue);
        })
        .Case([&](FloatAttr attr) {
            llvm::APSInt intValue(64U, true);
            bool isExact;
            attr.getValue().convertToInteger(
                intValue,
                llvm::APFloat::roundingMode::NearestTiesToEven,
                &isExact);
            return ekl::IndexAttr::get(
                input.getContext(),
                intValue.getZExtValue());
        })
        .Case([&](ekl::IndexAttr attr) {
            if (attr.getValue() > output.getUpperBound())
                return ekl::IndexAttr{};
            return attr;
        })
        .Default(ekl::IndexAttr{});
}

[[nodiscard]] static ScalarAttr coerce(ScalarAttr input, ScalarType output)
{
    return llvm::TypeSwitch<ScalarType, ScalarAttr>(output)
        .Case([&](ekl::IntegerType type) { return ::coerce(input, type); })
        .Case([&](FloatType type) { return ::coerce(input, type); })
        .Case([&](ekl::IndexType type) { return ::coerce(input, type); })
        .Default(ScalarAttr{});
}

[[nodiscard]] static ekl::ArrayAttr
coerce(ekl::ArrayAttr input, ScalarType output)
{
    SmallVector<Attribute> stack(input.getStack().getValue());
    for (auto &attr : stack) {
        attr =
            llvm::TypeSwitch<Attribute, Attribute>(attr)
                .Case(
                    [&](ekl::ArrayAttr array) { return coerce(array, output); })
                .Case([&](ScalarAttr scalar) { return coerce(scalar, output); })
                .Default([](auto) -> Attribute { return {}; });

        if (!attr) return {};
    }

    return ekl::ArrayAttr::get(input.getArrayType().cloneWith(output), stack);
}

OpFoldResult CoerceOp::fold(CoerceOp::FoldAdaptor adaptor)
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
    results.add<RewriteCoerceToUnify>(context);
}

//===----------------------------------------------------------------------===//
// ChoiceOp implementation
//===----------------------------------------------------------------------===//

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

} // namespace

void ChoiceOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext *context)
{
    results.add<RewriteChoiceToBroadcast>(context);
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
