/** Declares common utilities for implementing translation targets.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

#include <string>

namespace mlir::concepts {

//===----------------------------------------------------------------------===//
// Import
//===----------------------------------------------------------------------===//
//
// Import is the process that accepts some input source given via a
// llvm::SourceMgr and produces MLIR, returned as an owning reference to an op.
//

/** Function prototype for an MLIR importer. */
template<class OpTy>
using ImportFn = OwningOpRef<OpTy>(MLIRContext*, llvm::SourceMgr&);

/** Provides a flex compatible location type that also tracks llvm::SMLoc. */
// TODO: Make a location attribute for this.
struct ImportLocation {
    /*implicit*/    ImportLocation() = default;
    /*implicit*/    ImportLocation(const ImportLocation&) = default;

    llvm::SMLoc     asSMLoc() const { return location; }
    /*implicit*/    operator llvm::SMLoc() const { return asSMLoc(); }

    llvm::SMLoc     location;
    std::string     *filename;
    unsigned        line;
    unsigned        column;
};

/** Provides a Bison compatible location type that also tracks llvm::SMLoc. */
// TODO: Make a location attribute for this.
struct ImportRange {
    /*implicit*/    ImportRange() = default;
    /*implicit*/    ImportRange(const ImportRange&) = default;
    /*implicit*/    ImportRange(ImportLocation begin, ImportLocation end)
    : begin(begin), end(end)
    {}
    /*implicit*/    ImportRange(ImportLocation at)
    : ImportRange(at, at)
    {}

    llvm::SMRange   asSMRange() const { return {begin, end}; }
    /*implicit*/    operator llvm::SMRange() const { return asSMRange(); }

    ImportLocation  begin;
    ImportLocation  end;
};

/** Provides a helper context for an import translation action. */
class ImportContext {
    // TODO: Ideally, we'd have our own DiagHandler that can use the fact we are
    //       able to attach SMLoc to our ImportLocations.
    using DiagHandler   = SourceMgrDiagnosticHandler;

public:
    /** Initializes a new ImportContext. */
    explicit            ImportContext(
        MLIRContext *context,
        llvm::SourceMgr &source
    )
    : m_context(context),
      m_source(source),
      m_diag_handler(source, context)
    {}

    /** Gets the MLIRContext. */
    MLIRContext*        getContext() const { return m_context; }
    /** Gets the llvm::SourceMgr. */
    llvm::SourceMgr&    getSource() const { return m_source; }

    /** Given an ImportLocation, obtain an mlir::Location. */
    Location            getLocation(
        ImportLocation location,
        const Twine &name = {}
    ) const
    {
        // Initialize the location.
        Location result = FileLineColLoc::get(
            getContext(),
            *location.filename,
            location.line,
            location.column
        );

        if (!name.isTriviallyEmpty()) {
            // Wrap the location in a NameLoc.
            result = NameLoc::get(
                Identifier::get(name, getContext()),
                result
            );
        }

        return result;
    }

    /** Gets the filename of the main file. */
    StringRef      getFilename()
    {
        // Get the SM buffer.
        const auto fileId = getSource().getMainFileID();
        auto buffer = getSource().getMemoryBuffer(fileId);

        // Obtain the filename.
        auto filename = buffer->getBufferIdentifier();
        auto I = filename.find_last_of("/\\");
        I = (I == filename.size()) ? 0 : (I + 1);
        filename = filename.substr(I);
        return filename;
    }

    //===------------------------------------------------------------------===//
    // Diagnostics
    //===------------------------------------------------------------------===//
    //
    // Importers may produce diagnostics during lexing, parsing and building.
    // Technically, all these diagnostics can be aggregated with the MLIR diag-
    // nostic handling, provided the locations can be encoded.
    //
    // TODO: Since the SourceMgrDiagnosticHandler delegates to SourceMgr anyway,
    //       but needs to recover SMLoc, which we may have, we skip it here.
    //       This isn't good design but works for now.
    //

    InFlightDiagnostic  emitError(Location location)
    {
        return mlir::emitError(location);
    }
    void                emitError(ImportLocation location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Error, message);
    }
    void                emitError(ImportRange location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Error, message);
    }
    InFlightDiagnostic  emitWarning(Location location)
    {
        return mlir::emitWarning(location);
    }
    void                emitWarning(ImportLocation location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Warning, message);
    }
    void                emitWarning(ImportRange location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Warning, message);
    }
    InFlightDiagnostic  emitRemark(Location location)
    {
        return mlir::emitRemark(location);
    }
    void                emitRemark(ImportLocation location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Remark, message);
    }
    void                emitRemark(ImportRange location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Remark, message);
    }
    void                emitNote(ImportLocation location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Note, message);
    }
    void                emitNote(ImportRange location, const Twine &message)
    {
        emit(location, llvm::SourceMgr::DiagKind::DK_Note, message);
    }

private:
    void                emit(
        ImportLocation location,
        llvm::SourceMgr::DiagKind kind,
        const Twine &message
    )
    {
        // TODO: This is what the SourceMgrDiagHandler does anyway, but not
        //       as cleanly implemented. See above.
        m_source.PrintMessage(location, kind, message);
    }
    void                emit(
        ImportRange location,
        llvm::SourceMgr::DiagKind kind,
        const Twine &message
    )
    {
        // TODO: This is what the SourceMgrDiagHandler does anyway, but not
        //       as cleanly implemented. See above.
        m_source.PrintMessage(
            location.begin,
            kind,
            message,
            llvm::makeArrayRef(location.asSMRange())
        );
    }

    MLIRContext         *m_context;
    llvm::SourceMgr     &m_source;
    DiagHandler         m_diag_handler;
};

//===----------------------------------------------------------------------===//
// Export
//===----------------------------------------------------------------------===//
//
// Export is the process that accepts some MLIR input in the form of an op and
// writes to a raw_ostream.
//

/** Function prototype for an MLIR exporter. */
template<class OpTy>
using ExportFn = LogicalResult(OpTy op, raw_ostream&);

// TODO: ExportContext? How does diag work there without memory buffer?

} // namespace mlir::concepts
