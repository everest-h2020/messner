//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Ref dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef REF_ATTRIBUTES
#define REF_ATTRIBUTES

include "messner/Dialect/Ref/IR/Types.td"

def Ref_VolatileAttr : Ref_Attr<"Volatile"> {
    let mnemonic = "volatile";
    let summary = "Marker for memory side effects that model volatility";
    let description = [{
        The `ref.volatile` attibute is attached to a memory side effect of an
        operation to indicate that the particular effect is mandated by
        volatility. Volatility is implemented as follows:

            - Reading

              Reading from a shared reference may observe state changes that are
              caused outside of the program scope. Such changes are modeled by
              a stage 0 volatile write effect.

              Reading from a reference always causes a stage 1 read effect.

              Reading from an impure reference may change state, which imposes
              an ordering between subsequent reads. This is modeled by a stage 2
              volatile write effect. This effect applies to all values of the
              resource.

            - Writing

              Writing to a reference that is not marked as idempotent may have
              different effects based on the underlying state. Such dependencies
              are modeled by a stage 0 volatile read effect. This effect applies
              to all values of the resource.

              Writing to a reference always causes a stage 1 write effect.

        Example:

        ```
        #ref.volatile
        ```
    }];

    let parameters = (ins);
    let assemblyFormat = "";
}

#endif
