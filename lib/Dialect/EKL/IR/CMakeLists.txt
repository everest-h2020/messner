add_mlir_dialect_library(EKLDialect
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MessnerDialectUtils
        EKLEnums
        EKLAnalysis
        EKLInterfaces
        EKLTransforms
)
