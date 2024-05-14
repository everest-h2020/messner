/// Declares the DiagHandler.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Diagnostics.h"

#include <memory>

namespace llvm {

class SourceMgr;

} // namespace llvm

namespace mlir::ekl {

/// Implements a ScopedDiagnosticHandler that handles rich locations.
///
/// Re-implements the SourceMgrDiagnosticHandler so that it accepts
/// SourceLocationAttr to allow for better diagnostics.
struct DiagHandler : ScopedDiagnosticHandler {
    /// Initializes and installs a DiagHandler on @p context .
    ///
    /// @pre    `context && sourceMgr`
    ///
    /// @param              context     MLIRContext.
    /// @param              sourceMgr   llvm::SourceMgr.
    explicit DiagHandler(
        MLIRContext *context,
        std::shared_ptr<llvm::SourceMgr> sourceMgr);

private:
    LogicalResult emitDiagnostic(Diagnostic &diag) const;

    std::shared_ptr<llvm::SourceMgr> m_sourceMgr;
};

} // namespace mlir::ekl
