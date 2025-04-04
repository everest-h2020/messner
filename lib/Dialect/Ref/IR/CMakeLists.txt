add_mlir_dialect_library(RefDialect
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        RefIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRInferTypeOpInterface
        RefEnums
)
