add_mlir_library(EKLInterfaces
        ContiguousType.cpp
        ExpressionOp.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        EKLAnalysis
)
