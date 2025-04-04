add_mlir_conversion_library(EKLToLinalg
        EKLToLinalg.cpp

    DEPENDS
        EVPConversionPassesIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRPass
        MLIRArithDialect
        MLIRBufferizationDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        EKLDialect
)
