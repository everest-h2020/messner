add_mlir_conversion_library(EKLToFunc
        EKLToFunc.cpp

    DEPENDS
        EVPConversionPassesIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRPass
        MLIRFuncDialect
        MLIRBufferizationDialect
        EKLDialect
)
