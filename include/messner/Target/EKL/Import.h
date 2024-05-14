/// Declares the main entry point for importing EKL sources into MLIR.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::ekl {

/// Imports an EKL program from @p sourceMgr and type checks it.
///
/// The caller is expected to install a DiagHandler prior to calling this
/// function so that proper diagnostics will be emitted to the user.
///
/// @param              context     MLIRContext.
/// @param              sourceMgr   llvm::SourceMgr.
///
/// @retval nullptr     Failed to import, verify or type check.
/// @retval ProgramOp   The imported, verified and type checked program.
[[nodiscard]] OwningOpRef<ProgramOp> importAndTypeCheck(
    MLIRContext *context,
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr);

/// Registers the EKL import target.
void registerImport();

} // namespace mlir::ekl
