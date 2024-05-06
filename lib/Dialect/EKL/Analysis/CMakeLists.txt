add_mlir_library(EKLAnalysis
        AbstractTypeChecker.cpp
        Casting.cpp
        Extents.cpp
        LocalTypeChecker.cpp
        Number.cpp
        TypeCheckingAdaptor.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLIncGen
)
