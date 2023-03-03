//===- Types.td - TeIL type declarations -------------------*- tablegen -*-===//
//
// TODO: Add license.
//
//===----------------------------------------------------------------------===//
//
// This is the definitions file for the TeIL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef TEIL_TYPES
#define TEIL_TYPES

include "mlir/Dialect/TeIL/IR/Base.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Number type
//===----------------------------------------------------------------------===//

def TeIL_Number : TeIL_Type<"Number", [MemRefElementTypeInterface]> {
    let mnemonic = "num";
    let summary = "Generic unspecified number type";
    let description = [{
        `teil.num` acts as a generic number type that can be used as a stand-in
        in places where a concrete number representation has not been fixed yet.

        To disambiguate between multiple instances of this type, each type can
        hold a numeric equivalence class identifier. If no such identifier is
        specified, it is assumed to be zero. As a result, the following types
        are semantically equivalent:

        - `!teil.num` = `!teil.num<0>`
    }];

    let parameters = (ins
        TypeParameter<
            "natural_t",
            "equivalence class identifier"
        >:$equivalenceClass
    );

    let builders = [
        TypeBuilder<(ins CArg<"natural_t", "0">:$equivalenceClass), [{
            return $_get($_ctxt, equivalenceClass);
        }]>
    ];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;
    code parser = [{
        natural_t equivalenceClass = 0;
        if (succeeded($_parser.parseOptionalLess())) {
            if (
                $_parser.parseInteger(equivalenceClass)
                || $_parser.parseGreater()
            )
                return Type();
        }

        return get($_ctxt, equivalenceClass);
    }];
    code printer = [{
        $_printer << getMnemonic();
        if (const auto equivalenceClass = getEquivalenceClass()) {
            $_printer << "<" << equivalenceClass << ">";
        }
    }];
}

#endif
