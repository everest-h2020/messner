#include "llvm/Support/Debug.h"
#include "mlir/Dialect/CFDlang/IR/Ops.h"
#include "mlir/Dialect/CFDlang/Passes.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::cfdlang;

#define DEBUG_TYPE "cfdlang-contr-fact"

namespace {

class ContractionFactorizationPass
        : public ContractionFactorizationBase<ContractionFactorizationPass> {
public:
    virtual void runOnOperation() override
    {
        getOperation().getBody()->walk([this](Operation *op) {
            if (auto contract = dyn_cast<ContractOp>(op)) {
                work.push_back(contract);
            }
        });

        // Collect all contractions that shall be processed.
        //auto candidates = getOperation().getBody()->getOps<ContractOp>();
        //work.assign(candidates.begin(), candidates.end());

        // Process all candidates until the list is empty.
        while (!work.empty()) {
            visit(work.pop_back_val());
        }
    }

private:
    SmallVector<ContractOp, 16> work;

    void visit(ContractOp op)
    {
        // TODO: OpRewritePattern!

        LLVM_DEBUG(llvm::dbgs() << "visit(" << op << ")\n");

        // First, match against the (L # R) . [indices] pattern.
        auto prod = op.operand().getDefiningOp<ProductOp>();
        if (!prod) {
            // This candidate is discarded.
            return;
        }

        LLVM_DEBUG(llvm::dbgs() << "prod = " << prod << "\n");

        // Pattern is matched.
        auto L = prod.lhs().cast<Atom>(), R = prod.rhs().cast<Atom>();
        auto indices = op.indicesAttr().getAsValueRange();

        // Prepare lists for the rerwite indices, splitting at pivot.
        SmallVector<natural_t> L_indices, R_indices, LR_indices;
        const auto pivot = L.getRank();
        // Prepare a map for the result indices after rewriting.
        auto resultIndices = to_vector(
            llvm::iota_range<natural_t>(1, pivot + R.getRank(), true)
        );

        LLVM_DEBUG(
            llvm::dbgs() << "resultIndices = ";
            llvm::interleaveComma(resultIndices, llvm::dbgs());
            llvm::dbgs() << "\n";
        );

        // Loop over all contraction indices.
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            const auto x = *it++, y = *it;
            // Decide to which factor the indices belong.
            const auto u = x <= pivot, v = y <= pivot;
            if (u != v) {
                // The indices belong to different factors.
                LR_indices.push_back(x);
                LR_indices.push_back(y);
                continue;
            }

            if (u) {
                // The indices belong to L.
                L_indices.push_back(x);
                L_indices.push_back(y);
            } else {
                // The indices belong to R.
                R_indices.push_back(x - pivot);
                R_indices.push_back(y - pivot);
            }

            // The indices will no longer be contracted over in the result.
            resultIndices.erase(
                std::remove_if(
                    resultIndices.begin(),
                    resultIndices.end(),
                    [=](auto i) { return i == x || i == y; }
                ),
                resultIndices.end()
            );
        }

        LLVM_DEBUG(
            llvm::dbgs() << "L_indices = ";
            llvm::interleaveComma(L_indices, llvm::dbgs());
            llvm::dbgs() << "\nR_indices = ";
            llvm::interleaveComma(R_indices, llvm::dbgs());
            llvm::dbgs() << "\nLR_indices = ";
            llvm::interleaveComma(LR_indices, llvm::dbgs());
            llvm::dbgs() << "\nresultIndices = ";
            llvm::interleaveComma(resultIndices, llvm::dbgs());
            llvm::dbgs() << "\n";
        );

        OpBuilder builder(op);
        bool changed = false;

        if (!L_indices.empty()) {
            // The left factor can be rewritten.
            auto rewrite = builder.create<ContractOp>(
                op.getLoc(),
                L,
                L_indices
            );
            // Add the new contraction to the work list.
            work.push_back(rewrite);
            // Update the left atom.
            L = rewrite.result().cast<Atom>();
            changed = true;
        }
        if (!R_indices.empty()) {
            // The right factor can be rewritten.
            auto rewrite = builder.create<ContractOp>(
                op.getLoc(),
                R,
                R_indices
            );
            // Add the new contraction to the work list.
            work.push_back(rewrite);
            // Update the left atom.
            R = rewrite.result().cast<Atom>();
            changed = true;
        }

        if (!changed) {
            // No modifications made.
            return;
        }

        // Create a new product from changed operands.
        prod = builder.create<ProductOp>(
            op.getLoc(),
            L,
            R
        );

        if (LR_indices.empty()) {
            // The contraction was removed and is now a product.
            op.replaceAllUsesWith(prod.getResult());
            return;
        }

        // Process the LR_indices using the resultIndices map.
        for (auto &i : LR_indices) {
            i = std::distance(
                resultIndices.begin(),
                std::find(resultIndices.begin(), resultIndices.end(), i)
            ) + 1;
        }

        LLVM_DEBUG(
            llvm::dbgs() << "LR_indices = ";
            llvm::interleaveComma(LR_indices, llvm::dbgs());
            llvm::dbgs() << "\n";
        );

        // Create the rewritten contraction, which is irreducible.
        auto rewrite = builder.create<ContractOp>(
            op.getLoc(),
            prod.getResult().cast<Atom>(),
            LR_indices
        );
        // Replace the old contraction.
        op.replaceAllUsesWith(rewrite.getResult());
    }
};

} // namespace <anonymous>

std::unique_ptr<OperationPass<ModuleOp>> mlir::cfdlang::createContractionFactorizationPass() {
    return std::make_unique<ContractionFactorizationPass>();
}
