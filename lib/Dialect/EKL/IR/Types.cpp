/// Implements the EKL dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "messner/Dialect/EKL/IR/Types.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"

#include <bit>
#include <numeric>

using namespace mlir;
using namespace mlir::ekl;

//===----------------------------------------------------------------------===//
// Custom directives
//===----------------------------------------------------------------------===//

ParseResult
mlir::ekl::parseExtents(AsmParser &parser, SmallVectorImpl<extent_t> &extents)
{
    return parser.parseCommaSeparatedList(
        AsmParser::Delimiter::OptionalSquare,
        [&]() -> ParseResult {
            return parser.parseInteger(extents.emplace_back());
        });
}

void mlir::ekl::printExtents(AsmPrinter &printer, ExtentRange extents)
{
    if (extents.empty()) return;

    printer << "[";
    llvm::interleaveComma(extents, printer);
    printer << "]";
}

/// Parses an ArrayType without the angeled brackets.
///
/// This parser implements the following grammar:
///
/// ```
/// array-type ::= type [ `[` [ int { `,` int } ] `]` ]
/// ```
///
/// @param  [in]        parser  AsmParser.
/// @param  [out]       arrayTy ArrayType.
///
/// @return ParseResult.
static ParseResult parseArrayType(AsmParser &parser, ArrayType &arrayTy)
{
    const auto loc = parser.getCurrentLocation();
    ScalarType scalarTy;
    SmallVector<extent_t> extents;
    if (parser.parseType(scalarTy) || parseExtents(parser, extents))
        return failure();

    arrayTy = ArrayType::getChecked(
        [&]() { return parser.emitError(loc); },
        scalarTy,
        extents);
    return success();
}

/// Prints an ArrayType without the angeled brackets.
///
/// This printer implements the following grammar:
///
/// ```
/// array-type ::= type [ `[` [ int { `,` int } ] `]` ]
/// ```
///
/// @param  [in]        printer AsmPrinter.
/// @param              arrayTy ArrayType.
static void printArrayType(AsmPrinter &printer, ArrayType arrayTy)
{
    printer << arrayTy.getScalarType();
    printExtents(printer, arrayTy.getExtents());
}

/// Parses an optional ReferenceKind value.
///
/// This parser implements the following grammar:
///
/// ```
/// ref-kind ::= [ (string | `in` | `out` | `inout`) ]
/// ```
///
/// @param  [in]        parser  AsmParser.
/// @param  [out]       kind    ReferenceKind.
///
/// @return ParseResult.
static ParseResult parseReferenceKind(AsmParser &parser, ReferenceKind &kind)
{
    const auto loc = parser.getCurrentLocation();

    StringRef keyword;
    std::string buffer;
    if (parser.parseOptionalKeyword(&keyword, {"in", "out", "inout"})) {
        if (parser.parseOptionalString(&buffer)) {
            kind = ReferenceKind::In;
            return success();
        }
        keyword = buffer;
    }

    if (const auto parsed = symbolizeReferenceKind(keyword)) {
        kind = *parsed;
        return success();
    }

    return parser.emitError(loc, "expected reference kind");
}

/// Prints an optional ReferenceKind value.
///
/// This printer implements the following grammar:
///
/// ```
/// ref-kind ::= [ (`out` | `inout`) ` ` ]
/// ```
///
/// @param  [in]        printer AsmPrinter.
/// @param              kind    ReferenceKind.
static void printReferenceKind(AsmPrinter &printer, ReferenceKind kind)
{
    if (kind == ReferenceKind::In) return;

    printer << stringifyReferenceKind(kind) << " ";
}

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "messner/Dialect/EKL/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IndexType implementation
//===----------------------------------------------------------------------===//

unsigned ekl::IndexType::getBitWidth() const
{
    if (getUpperBound() == 0) return 0U;
    return 64U - std::countl_zero(getUpperBound());
}

//===----------------------------------------------------------------------===//
// ArrayType implementation
//===----------------------------------------------------------------------===//

/// Implements flattening ArrayType construction.
///
/// @pre    `llvm::isa_and_present<ScalarType, ArrayType>(scalarOrArrayType)`
[[nodiscard]] static ArrayType
getArrayType(auto &&get, Type scalarOrArrayType, ExtentRange extents)
{
    return llvm::TypeSwitch<Type, ArrayType>(scalarOrArrayType)
        .Case([&](ArrayType arrayTy) {
            const auto newExtents = concat(arrayTy.getExtents(), extents);
            return get(arrayTy.getScalarType(), newExtents);
        })
        .Case([&](ScalarType scalarTy) { return get(scalarTy, extents); });
}

ArrayType ArrayType::get(Type scalarOrArrayType, ExtentRange extents)
{
    return getArrayType(
        [](ScalarType scalarTy, ExtentRange extents) {
            return ArrayType::get(scalarTy, extents);
        },
        scalarOrArrayType,
        extents);
}

ArrayType ArrayType::getChecked(
    function_ref<InFlightDiagnostic()> emitError,
    Type scalarOrArrayType,
    ExtentRange extents)
{
    return getArrayType(
        [&](ScalarType scalarTy, ExtentRange extents) {
            return ArrayType::getChecked(emitError, scalarTy, extents);
        },
        scalarOrArrayType,
        extents);
}

LogicalResult ArrayType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ScalarType scalarType,
    ExtentRange extents)
{
    if (!scalarType) return emitError() << "expected scalar type";
    if (hasNoElements(extents)) return emitError() << "extents can't be zero";

    return success();
}

ArrayType ArrayType::cloneWith(Type scalarOrArrayType) const
{
    return llvm::TypeSwitch<Type, ArrayType>(scalarOrArrayType)
        .Case([&](ArrayType arrayTy) {
            const auto newExtents = concat(getExtents(), arrayTy.getExtents());
            return get(arrayTy.getScalarType(), newExtents);
        })
        .Case([&](ScalarType scalarTy) { return cloneWith(scalarTy); });
}

//===----------------------------------------------------------------------===//
// ABIScalarType
//===----------------------------------------------------------------------===//

bool ABIScalarType::classof(FloatType type)
{
    return LLVM::isCompatibleFloatingPointType(type);
}

Type ABIScalarType::getLLVMType() const
{
    return llvm::TypeSwitch<Type, Type>(*this)
        .Case([](ekl::IntegerType intTy) {
            return IntegerType::get(intTy.getContext(), intTy.getWidth());
        })
        .Case([](FloatType floatTy) { return floatTy; })
        .Case([](BoolType boolTy) { return boolTy; });
}

//===----------------------------------------------------------------------===//
// ABIType implementation
//===----------------------------------------------------------------------===//

bool ABIType::classof(ArrayType type)
{
    return llvm::isa<ABIScalarType>(type.getScalarType())
        && llvm::all_of(type.getExtents(), [](uint64_t x) {
               return x <= std::numeric_limits<unsigned>::max();
           });
}

Type ABIType::getLLVMType() const
{
    return llvm::TypeSwitch<Type, Type>(*this)
        .Case([](ABIScalarType scalarTy) { return scalarTy.getLLVMType(); })
        .Case([](ArrayType arrayTy) {
            auto result = llvm::cast<ABIScalarType>(arrayTy.getScalarType())
                              .getLLVMType();
            for (auto ext : llvm::reverse(arrayTy.getExtents()))
                result = LLVM::LLVMArrayType::get(result, ext);
            return result;
        })
        .Case([](ABIReferenceType referenceTy) {
            return referenceTy.getLLVMType();
        });
}

//===----------------------------------------------------------------------===//
// ABIReferenceType implementation
//===----------------------------------------------------------------------===//

Type ABIReferenceType::getLLVMType() const
{
    return LLVM::LLVMPointerType::get(getContext());
}

//===----------------------------------------------------------------------===//
// EKLDialect
//===----------------------------------------------------------------------===//

Type EKLDialect::parseType(DialectAsmParser &parser) const
{
    if (!parser.parseOptionalColon())
        return IdentityType::get(parser.getContext());
    if (!parser.parseOptionalStar())
        return ExtentType::get(parser.getContext());
    if (!parser.parseOptionalEllipsis())
        return EllipsisType::get(parser.getContext());
    if (!parser.parseOptionalQuestion())
        return ErrorType::get(parser.getContext());

    StringRef keyword;
    Type result;
    const auto maybe = generatedTypeParser(parser, &keyword, result);
    if (maybe.has_value()) {
        if (maybe.value()) return nullptr;
        return result;
    }

    if (keyword.consume_front("_")) {
        uint64_t value;
        if (keyword.consumeInteger(10, value) || !keyword.empty()) {
            parser.emitError(parser.getNameLoc(), "expected index literal");
            return nullptr;
        }

        return IndexType::get(parser.getContext(), value);
    }

    parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
    return nullptr;
}

void EKLDialect::printType(Type type, DialectAsmPrinter &os) const
{
    if (llvm::isa<IdentityType>(type)) {
        os << ":";
        return;
    }
    if (llvm::isa<ExtentType>(type)) {
        os << "*";
        return;
    }
    if (llvm::isa<EllipsisType>(type)) {
        os << "...";
        return;
    }
    if (llvm::isa<ErrorType>(type)) {
        os << "?";
        return;
    }
    if (const auto indexTy = llvm::dyn_cast<IndexType>(type)) {
        if (!indexTy.isUnbounded()) {
            os << "_" << indexTy.getUpperBound();
            return;
        }
    }

    const auto ok = generatedTypePrinter(type, os);
    assert(succeeded(ok));
}

void EKLDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "messner/Dialect/EKL/IR/Types.cpp.inc"
        >();
}
