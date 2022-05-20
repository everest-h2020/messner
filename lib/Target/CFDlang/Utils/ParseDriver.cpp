/** Implements the parser driver.
 *
 * @file
 * @author      Karl F. A. Friebel (karl.friebel@tu-dresden.de)
 *
 */

#include "mlir/Target/CFDlang/Utils/ParseDriver.h"

#include "Tokenizer.h"

using namespace mlir;
using namespace mlir::cfdlang;

namespace mlir::cfdlang::detail {

ParseDriver::ParseDriver(ImportContext &context)
: m_context(context)
{
    // Get the buffer of the file to parse.
    const auto fileId = context.getSource().getMainFileID();
    const auto buffer = context.getSource().getMemoryBuffer(fileId)
        ->getBuffer();

    // Initialize the tokenizer.
    m_tokenizer = new Tokenizer(
        context,
        reflex::Input(buffer.data(), buffer.size())
    );
    m_tokenizer->filename = context.getFilename();
}

ParseDriver::~ParseDriver()
{
    // Destroy the tokenizer.
    delete m_tokenizer;
}

LogicalResult ParseDriver::parse()
{
    Parser parser(*this);
    return success(parser.parse() == 0);
}

void ParseDriver::program()
{
    // Obtain the start location,
    ImportLocation startLocation;
    startLocation.filename = &m_tokenizer->filename;
    startLocation.line = 0;
    startLocation.column = 0;
    startLocation.location = llvm::SMLoc::getFromPointer(
        m_tokenizer->in().cstring()
    );

    // Initialize the ProgramBuilder.
    m_state.reset();
    m_state.emplace(getContext(), startLocation);
}
bool ParseDriver::stmt_begin(ImportRange location, StringRef id)
{
    // Lookup the declaration for id.
    const auto declaration = result().getSymbols().lookup(id);
    if (!declaration) {
        error(location,
            "symbol with name '" + id + "' was not declared"
        );
        return false;
    }
    if (!declaration->canHaveDefinition()) {
        error(location,
            "declaration with name '" + id + "' cannot have a definition'"
        );
        return false;
    }
    if (declaration->isDefined()) {
        warning(location,
            "redeclared atom '" + id + "'"
        );
        note(declaration->getDefinition().getValue(),
            "previous definition was here"
        );
    }

    // Create the definition block.
    const auto loc = getContext().getLocation(location.begin, id);
    Block *body;
    switch (declaration->getKind())
    {
        case DeclarationKind::Variable:
            body = &result().getBuilder()
                .create<DefineOp>(loc, id, declaration->getType())
                .body()
                .front();
            break;
        case DeclarationKind::Output:
            body = &result().getBuilder()
                .create<OutputOp>(loc, id, declaration->getType())
                .body()
                .front();
            break;
        default:
            llvm_unreachable("DeclarationKind");
    }

    // Step into the definition.
    result().getBuilder().setInsertionPointToStart(body);
    return true;
}
bool ParseDriver::stmt_end(ImportRange location, StringRef id, AtomOp value)
{
    // Get the declaration.
    const auto declaration = result().getSymbols().lookup(id);
    assert(declaration);
    if (declaration->getType() != value.getAtomType()) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "type mismatch: value of type [";
        llvm::interleave(value.getAtomShape(), os, " ");
        os << "] cannot bind to declaration of type [";
        llvm::interleave(declaration->getType().getShape(), os, " ");
        os << ']';
        error(location, os.str());
        note(declaration->getLocation(), "see declaration here");
        return false;
    }
    declaration->define(location.begin);

    // Emit the YieldOp.
    result().getBuilder().create<YieldOp>(
        getContext().getLocation(location.begin),
        value.getAtom()
    );

    // Step out of the definition.
    result().getBuilder().setInsertionPointAfter(
        result().getBuilder().getBlock()->getParentOp()
    );
    return true;
}

bool ParseDriver::decl(
    ImportRange location,
    StringRef id,
    AtomType type,
    DeclarationKind kind
)
{
    // Emplace the symbol.
    auto [ok, sym] = result().getSymbols().emplace(id, location, kind, type);
    if (failed(ok)) {
        error(location, "symbol '" + id + "' redeclared");
        note(sym.getLocation(), "previous declaration is here");
        return false;
    }

    if (kind == DeclarationKind::Input) {
        // Emit the input declaration.
        result().getBuilder().create<InputOp>(
            getContext().getLocation(location.begin, id),
            id,
            type
        );
    }

    return true;
}

AtomType ParseDriver::type_expr(ImportRange location, StringRef id)
{
    // Get the declaration.
    const auto declaration = result().getSymbols().lookup(id);
    if (!declaration) {
        error(location, "unknown type identifier '" + id + "'");
        return nullptr;
    }
    if (!declaration->isType()) {
        error(location, "symbol '" + id + "' does not name a type");
        note(declaration->getLocation(), "see declaration here");
        return nullptr;
    }

    // Return the type.
    return declaration->getType();
}
AtomType ParseDriver::type_expr(ImportRange location, shape_t shape)
{
    // Check the shape.
    if (teil::isTriviallyEmpty(shape)) {
        error(location, "atom shape cannot be trivially empty");
        return nullptr;
    }

    // Build the type.
    return AtomType::get(getContext().getContext(), shape);
}

AtomOp ParseDriver::eval(ImportRange location, StringRef id)
{
    // Get the declaration.
    const auto declaration = result().getSymbols().lookup(id);
    if (!declaration) {
        error(location, "unknown atom identifier '" + id + "'");
        return nullptr;
    }
    if (!declaration->isAtom()) {
        error(location, "symbol '" + id + "' does not name an atom");
        note(declaration->getLocation(), "see declaration here");
        return nullptr;
    }

    // Emit the AtomOp.
    return result().getBuilder().create<EvalOp>(
        getContext().getLocation(location.begin),
        declaration->getType(),
        SymbolRefAttr::get(result().getBuilder().getContext(), id)
    );
}

template<class BinOp>
static AtomOp binop(
    ProgramBuilder &builder,
    ImportRange location,
    AtomOp lhs,
    AtomOp rhs
)
{
    if (!teil::are_compatible(lhs.getAtomShape(), rhs.getAtomShape())) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "incompatible shapes [";
        llvm::interleave(lhs.getAtomShape(), os, " ");
        os << "] and [";
        llvm::interleave(rhs.getAtomShape(), os, " ");
        os << ']';
        builder.getContext().emitError(location, os.str());
        return nullptr;
    }

    return builder.getBuilder().create<BinOp>(
        builder.getContext().getLocation(location.begin),
        lhs.getAtom(),
        rhs.getAtom()
    );
}

AtomOp ParseDriver::add(ImportRange location, AtomOp lhs, AtomOp rhs)
{
    return binop<AddOp>(result(), location, lhs, rhs);
}
AtomOp ParseDriver::sub(ImportRange location, AtomOp lhs, AtomOp rhs)
{
    return binop<SubOp>(result(), location, lhs, rhs);
}
AtomOp ParseDriver::mul(ImportRange location, AtomOp lhs, AtomOp rhs)
{
    return binop<MulOp>(result(), location, lhs, rhs);
}
AtomOp ParseDriver::div(ImportRange location, AtomOp lhs, AtomOp rhs)
{
    return binop<DivOp>(result(), location, lhs, rhs);
}
AtomOp ParseDriver::prod(ImportRange location, AtomOp lhs, AtomOp rhs)
{
    return result().getBuilder().create<ProductOp>(
        getContext().getLocation(location.begin),
        lhs.getAtom(),
        rhs.getAtom()
    );
}
AtomOp ParseDriver::cont(
    ImportRange location,
    AtomOp op,
    NatList indices
)
{
    for (auto &i : indices) { ++i; }

    auto idxAttr = teil::NatArrayAttr::get(getContext().getContext(), indices);
    NamedAttrList attrs;
    attrs.append(
        result().getBuilder()
            .getNamedAttr("indices",  idxAttr)
    );

    teil::ShapeStorage shape;
    if (
        failed(
            ContractOp::inferAtomShape(
                getContext().getContext(),
                {},
                {op.getAtom()},
                result().getBuilder().getDictionaryAttr(attrs),
                {},
                shape
            )
        )
    ) {
        error(location, "invalid reduction indices");
        return nullptr;
    }

    return result().getBuilder().create<ContractOp>(
        getContext().getLocation(location.begin),
        op.getAtom(),
        idxAttr
    );
}

} // namespace mlir::cfdlang::detail
