add_mlir_conversion_library(EKLToStandard
        EKLToStandard.cpp

    DEPENDS
        EVPConversionPassesIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRPass
        MLIRArithDialect
        MLIRIndexDialect
        MLIRMathDialect
        EKLDialect
)
