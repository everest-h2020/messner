add_mlir_dialect_library(TeILDialect
        Enums.cpp

        IR/Base.cpp
        IR/Interfaces.cpp
        IR/Ops.cpp
        IR/Types.cpp

    DEPENDS
        TeILInterfacesIncGen
        TeILIRIncGen
        TeILEnumsIncGen

    LINK_COMPONENTS
        Core
)
