add_mlir_library(MessnerDialectUtils
        Closures.cpp

    ENABLE_AGGREGATION
    DEPENDS
        MLIRIR
        MLIRParser
)
