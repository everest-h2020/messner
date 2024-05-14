/// Implements the DiagHandler.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/DiagHandler.h"

#include "messner/Dialect/EKL/IR/EKL.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::ekl;

/// Converts a DiagnosticSeverity enum into a llvm::SourceMgr::DiagKind enum.
[[nodiscard]] static llvm::SourceMgr::DiagKind
getDiagKind(DiagnosticSeverity kind)
{
    switch (kind) {
    case DiagnosticSeverity::Error:   return llvm::SourceMgr::DK_Error;
    case DiagnosticSeverity::Warning: return llvm::SourceMgr::DK_Warning;
    case DiagnosticSeverity::Remark:  return llvm::SourceMgr::DK_Remark;
    case DiagnosticSeverity::Note:    return llvm::SourceMgr::DK_Note;
    }
    llvm_unreachable("unknown DiagnosticSeverity");
}

/// Finds the next CallSiteLoc caller nested below @p loc , if any.
[[nodiscard]] static LocationAttr getCaller(LocationAttr loc)
{
    return llvm::TypeSwitch<LocationAttr, LocationAttr>(loc)
        .Case([](CallSiteLoc callSite) { return callSite.getCaller(); })
        .Case([](NameLoc name) { return getCaller(name.getChildLoc()); })
        .Case([](FusedLoc fused) {
            for (auto callSite :
                 llvm::map_range(fused.getLocations(), getCaller))
                if (callSite) return callSite;
            return LocationAttr{};
        })
        .Default(LocationAttr{});
}

/// Builds a trace from @p loc and its call hierarchy.
[[nodiscard]] static SmallVector<Location> getTrace(LocationAttr loc)
{
    SmallVector<Location> result;
    for (; loc; loc = getCaller(loc)) result.push_back(loc);
    return result;
}

/// Ensures @p filename is mapped in @p sourceMgr .
static int ensureMapped(llvm::SourceMgr &sourceMgr, StringRef filename)
{
    std::optional<int> bufferIndex;
    for (auto idx : llvm::iota_range<int>(1, sourceMgr.getNumBuffers(), false))
        if (filename == sourceMgr.getMemoryBuffer(idx)->getBufferIdentifier()) {
            bufferIndex = idx;
            break;
        }
    if (!bufferIndex) {
        std::string ignore;
        bufferIndex = sourceMgr.AddIncludeFile(filename.str(), {}, ignore);
    }

    return *bufferIndex;
}

/// Ensures @p filename is mapped in @p sourceMgr and gets a location in it.
[[nodiscard]] static llvm::SMRange ensureMapped(
    llvm::SourceMgr &sourceMgr,
    StringRef filename,
    unsigned line,
    unsigned column)
{
    if (line == 0 || column == 0) return {};
    const auto start = sourceMgr.FindLocForLineAndColumn(
        ensureMapped(sourceMgr, filename),
        line,
        column);
    return llvm::SMRange(start, start);
}

/// Ensures @p loc is mapped in @p sourceMgr and gets it.
[[nodiscard]] static llvm::SMRange
ensureMapped(llvm::SourceMgr &sourceMgr, FileLineColLoc loc)
{
    if (!loc) return {};
    return ensureMapped(
        sourceMgr,
        loc.getFilename().getValue(),
        loc.getLine(),
        loc.getColumn());
}

/// @copydoc ensureMapped(llvm::SourceMgr &, FileLineColLoc)
[[nodiscard]] static llvm::SMRange
ensureMapped(llvm::SourceMgr &sourceMgr, SourceLocationAttr loc)
{
    if (!loc) return {};
    const auto bufferIdx = ensureMapped(sourceMgr, loc.getFilename());
    return llvm::SMRange(
        sourceMgr.FindLocForLineAndColumn(
            bufferIdx,
            loc.getStartLine(),
            loc.getStartColumn()),
        sourceMgr.FindLocForLineAndColumn(
            bufferIdx,
            loc.getEndLine(),
            loc.getEndColumn()));
}

/// @copydoc ensureMapped(llvm::SourceMgr &, FileLineColLoc)
[[nodiscard]] static llvm::SMRange
ensureMapped(llvm::SourceMgr &sourceMgr, LocationAttr loc)
{
    if (!loc) return {};
    return TypeSwitch<LocationAttr, llvm::SMRange>(loc)
        .Case([&](CallSiteLoc callSite) {
            return ensureMapped(sourceMgr, callSite.getCallee());
        })
        .Case([&](FileLineColLoc fileLineCol) {
            return ensureMapped(sourceMgr, fileLineCol);
        })
        .Case([&](FusedLoc fusedLoc) {
            for (auto child : fusedLoc.getLocations()) {
                const auto childLoc = ensureMapped(sourceMgr, child);
                if (childLoc.isValid()) return childLoc;
            }
            return llvm::SMRange{};
        })
        .Case([&](NameLoc nameLoc) {
            return ensureMapped(sourceMgr, nameLoc.getChildLoc());
        })
        .Case([&](OpaqueLoc opaqueLoc) {
            if (const auto sourceLoc =
                    SourceLocationAttr::fromLocation(opaqueLoc))
                return ensureMapped(sourceMgr, sourceLoc);
            return ensureMapped(sourceMgr, opaqueLoc.getFallbackLocation());
        })
        .Case([](UnknownLoc) { return llvm::SMRange{}; });
}

/// Emits a diagnostic using @p sourceMgr .
static void emit(
    llvm::SourceMgr &sourceMgr,
    DiagnosticSeverity severity,
    llvm::SMRange loc,
    const llvm::Twine &msg)
{
    ArrayRef<llvm::SMRange> ranges = {loc};
    if (loc.Start == loc.End) ranges = {};

    sourceMgr.PrintMessage(loc.Start, getDiagKind(severity), msg, ranges);
}

//===----------------------------------------------------------------------===//
// DiagHandler implementation
//===----------------------------------------------------------------------===//

DiagHandler::DiagHandler(
    MLIRContext *context,
    std::shared_ptr<llvm::SourceMgr> sourceMgr)
        : ScopedDiagnosticHandler(context),
          m_sourceMgr(std::move(sourceMgr))
{
    assert(context && m_sourceMgr);

    setHandler([&](Diagnostic &diag) { return emitDiagnostic(diag); });
}

LogicalResult DiagHandler::emitDiagnostic(Diagnostic &diag) const
{
    const auto trace = getTrace(diag.getLocation());

    emit(
        *m_sourceMgr,
        diag.getSeverity(),
        ensureMapped(*m_sourceMgr, trace.front()),
        diag.str());

    for (auto caller : ArrayRef<Location>(trace).drop_front()) {
        const auto loc = ensureMapped(*m_sourceMgr, caller);
        if (!loc.isValid()) continue;
        emit(*m_sourceMgr, DiagnosticSeverity::Note, loc, "called from");
    }

    auto loc = trace.front();
    for (auto &note : diag.getNotes()) {
        if (note.getLocation() != loc) {
            loc = note.getLocation();
            emit(
                *m_sourceMgr,
                DiagnosticSeverity::Note,
                ensureMapped(*m_sourceMgr, loc),
                note.str());
            continue;
        }

        emit(*m_sourceMgr, DiagnosticSeverity::Note, {}, note.str());
    }

    return success();
}
