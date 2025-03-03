add_mlir_conversion_library(EKLToIndex
        EKLToIndex.cpp

    DEPENDS
        EVPConversionPassesIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRPass
        MLIRIndexDialect
        EKLDialect
)
