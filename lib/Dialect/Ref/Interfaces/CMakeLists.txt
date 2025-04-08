add_mlir_library(RefInterfaces
    ReferenceTypeInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSideEffectInterfaces
        RefEnums)
