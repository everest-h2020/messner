add_mlir_library(EKLInterfaces
        ContiguousType.cpp
        TypeCheckOpInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        EKLAnalysis
)
