add_mlir_dialect_library(EKLDialect
        Attributes.cpp
        Base.cpp
        DiagHandler.cpp
        Ops.cpp
        Traits.cpp
        Types.cpp

    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRArithDialect
        MLIRLLVMDialect
        MLIRMathDialect
        MLIRTensorDialect
        EKLEnums
        EKLAnalysis
        EKLInterfaces
        EKLTransforms
)
