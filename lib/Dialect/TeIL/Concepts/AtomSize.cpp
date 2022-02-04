/** Implements the TeIL atom size concept.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Dialect/TeIL/Concepts/AtomSize.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/TeIL/IR/Base.h"

#define DEBUG_TYPE "teil-atom-size"

using namespace mlir;
using namespace mlir::teil;

LogicalResult AtomSize::fold(ArrayRef<DimSizeAttr> args)
{
    LLVM_DEBUG(
        llvm::dbgs() << "mlir::teil::AtomSize::fold({";
        llvm::interleaveComma(getShape(), llvm::dbgs());
        llvm::dbgs() << "} with {";
        llvm::interleaveComma(args, llvm::dbgs());
        llvm::dbgs() << "} -> {";
    );

    auto result = false;

    auto arg = args.begin();
    for (
        auto [sz, val] = std::make_pair(m_shape.begin(), m_values.begin());
        sz != m_shape.end();
        ++sz,++val
    ) {
        if (*sz != dynamic_size) {
            // Size is already static.
            continue;
        }

        assert(arg != args.end());
        if (auto constant = *arg++) {
            // Fold the constant into the shape.
            *sz = constant.getValue();
            result |= true;
            continue;
        }
    }
    assert(arg == args.end());

    LLVM_DEBUG(
        llvm::interleaveComma(getShape(), llvm::dbgs());
        llvm::dbgs() << "}\n";
    );
    return success(result);
}

LogicalResult AtomSize::reify(OpBuilder& builder, Location loc)
{
    auto dialect = builder.getContext()->getLoadedDialect<TeILDialect>();
    auto result = true;

    for (
        auto [sz, val] = std::make_pair(m_shape.begin(), m_values.begin());
        sz != m_shape.end();
        ++sz,++val
    ) {
        if (*sz == dynamic_size) {
            if (!*val) {
                // Size is dynamic but we don't have a value for it, reification
                // can never complete successfully.
                result &= false;
            }

            // Size is already dynamic.
            continue;
        }

        // Size is static and can be reified.
        const auto attr = builder.getIndexAttr(*sz);
        *val = dialect
            ->materializeConstant(builder, attr, attr.getType(), loc)
            ->getResult(0)
            .cast<DimSize>();
    }

    // Indicate what verify() will return now.
    return success(result);
}
