/// Declares the main entry point for importing EKL sources into MLIR.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::ekl {

// /// Provides a shared context for lexing and parsing.
// struct ImportContext {
//     /// Initializes an ImportContext.
//     ///
//     /// @param              context     MLIRContext.
//     /// @param              sourceMgr   llvm::SourceMgr.
//     /// @param              errorLimit  Number of errors to tolerate.
//     explicit ImportContext(
//         MLIRContext *context,
//         std::shared_ptr<llvm::SourceMgr> sourceMgr,
//         unsigned errorLimit = 5);

//     // Closes the ImportContext.
//     ~ImportContext();

//     /// Obtains an MLIR Location for an ImportLocation.
//     ///
//     /// The obtained Location is not round-trippable through MLIR export,
//     /// meaning that after serialization, its value is no longer equivalent
//     /// to the original. This is because MLIR location attributes can't be
//     /// extended such that the AsmPrinter will handle them correctly.
//     [[nodiscard]] Location getLocation(ImportLocation location);

//     /// Determines whether processing should continue after an error.
//     ///
//     /// The caller is responsible for emitting an error diagnostic first,
//     then
//     /// calling this method and observing its return value.
//     ///
//     /// @retval failure     Too many errors were consumed, abort.
//     /// @retval success     Continue processing.
//     LogicalResult consumeError(ImportLocation location)
//     {
//         m_hasErrors = true;
//         if (m_errorLimit == 0) {
//             emitError(location, "too many errors, aborting");
//             return failure();
//         }

//         --m_errorLimit;
//         return success();
//     }
//     /// Determines whether processing should continue after @p diagnostic .
//     ///
//     /// @retval failure     Too many errors were consumed, abort.
//     /// @retval success     Continue processing.
//     LogicalResult consumeError(InFlightDiagnostic diagnostic)
//     {
//         if (const auto diag = diagnostic.getUnderlyingDiagnostic())
//             return consumeError(diag->getLocation());
//         return success();
//     }

//     /// Gets the MLIRContext.
//     MLIRContext *getContext() const { return m_context; }
//     /// Gets the llvm::SourceMgr.
//     llvm::SourceMgr &getSourceMgr() const { return *m_sourceMgr; }
//     /// Gets the main source file memory buffer.
//     llvm::StringRef getBuffer() const { return m_buffer; }
//     /// Obtains the current remaining error limit.
//     unsigned getErrorLimit() const { return m_errorLimit; }
//     /// Determines whether errors were consumed by this context.
//     bool hasErrors() const { return m_hasErrors; }

//     /// @copydoc getContext()
//     /*implicit*/ operator MLIRContext *() const { return getContext(); };

// private:
//     llvm::SMLoc getSMLoc(FileLineColLoc location) const;
//     llvm::SMRange getSMRange(LocationAttr location) const;
//     LogicalResult emitDiagnostic(Diagnostic &diag) const;

//     MLIRContext *m_context;
//     DiagnosticEngine::HandlerID m_handlerId;
//     std::shared_ptr<llvm::SourceMgr> m_sourceMgr;
//     llvm::StringRef m_buffer;
//     unsigned m_errorLimit;
//     bool m_hasErrors;
// };

/// Registers the EKL import target.
void registerImport();

} // namespace mlir::ekl
