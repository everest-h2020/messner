add_mlir_library(RefEnums
        Enums.cpp

    ENABLE_AGGREGATION
    DEPENDS
        RefEnumsIncGen

    LINK_LIBS
        MLIRIR)

add_subdirectory(Interfaces)
add_subdirectory(IR)
