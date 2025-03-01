add_mlir_conversion_library(EKLToArith
        EKLToArith.cpp

    DEPENDS
        EVPConversionPassesIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRPass
        MLIRArithDialect
        EKLDialect
)
