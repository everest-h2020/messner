/// Implements the EKL dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Base.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ekl;

//===- Generated implementation -------------------------------------------===//

#include "messner/Dialect/EKL/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

struct EKLAsmInterface : OpAsmDialectInterface {
    using OpAsmDialectInterface::OpAsmDialectInterface;

    static constexpr std::size_t kInlineArrayLength = 9;

    AliasResult getAlias(Attribute attr, raw_ostream &os) const override final
    {
        // Direct the printer to outline arrays that are longer than the limit
        // to a non-deferred alias.
        return llvm::TypeSwitch<Attribute, AliasResult>(attr)
            .Case([&](ekl::ArrayAttr arrayAttr) {
                if (arrayAttr.getStack().size() <= kInlineArrayLength)
                    return AliasResult::NoAlias;

                os << "array";
                return AliasResult::OverridableAlias;
            })
            .Case([&](InitializerAttr initAttr) {
                if (static_cast<std::size_t>(initAttr.getFlattened().size())
                    <= kInlineArrayLength)
                    return AliasResult::NoAlias;

                os << "init";
                return AliasResult::OverridableAlias;
            })
            .Default(AliasResult::NoAlias);
    }
};

struct EKLInlinerInterface : DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &)
        const override final
    {
        return true;
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// EKLDialect
//===----------------------------------------------------------------------===//

Operation *EKLDialect::materializeConstant(
    OpBuilder &builder,
    Attribute attr,
    Type type,
    Location location)
{
    // LiteralOp only materializes expression values.
    if (const auto exprTy = llvm::dyn_cast<ExpressionType>(type)) {
        if (const auto literalAttr = llvm::dyn_cast<LiteralAttr>(attr)) {
            if (exprTy.getTypeBound() != literalAttr.getType()) return nullptr;

            return builder.create<LiteralOp>(location, literalAttr);
        }
    }

    return nullptr;
}

void EKLDialect::initialize()
{
    registerAttributes();
    registerOps();
    registerTypes();

    addInterfaces<EKLAsmInterface, EKLInlinerInterface>();
}
