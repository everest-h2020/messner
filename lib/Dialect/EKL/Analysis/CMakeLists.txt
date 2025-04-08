add_mlir_library(EKLAnalysis
        Rational.cpp
        Shape.cpp

    ENABLE_AGGREGATION
    DEPENDS

    LINK_LIBS
        MLIRIR)
