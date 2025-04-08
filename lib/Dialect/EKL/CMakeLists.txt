add_mlir_library(EKLEnums
        Enums.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLEnumsIncGen

    LINK_LIBS
        MLIRIR)

add_subdirectory(Analysis)
add_subdirectory(Interfaces)
add_subdirectory(IR)
