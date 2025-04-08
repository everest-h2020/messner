add_mlir_library(EKLInterfaces
        ContiguousType.cpp
        SemaOpInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        EKLAnalysis)
