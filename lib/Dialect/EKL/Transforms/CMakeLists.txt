add_mlir_library(EKLTransforms
        Lower.cpp
        TypeCheck.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLPassesIncGen
    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        EKLAnalysis
        EKLInterfaces
)
