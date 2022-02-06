add_mlir_dialect_library(CFDlangDialect
        IR/Base.cpp
        IR/Interfaces.cpp
        IR/Ops.cpp
        IR/Types.cpp
        Transforms/ContractionFactorization.cpp

    LINK_COMPONENTS
        Core

    DEPENDS
        CFDlangIRIncGen
        CFDlangInterfacesIncGen
        CFDlangPassesIncGen
)
