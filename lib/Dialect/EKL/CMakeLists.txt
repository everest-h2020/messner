add_subdirectory(Analysis)
add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_mlir_library(EKLEnums
        Enums.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EKLEnumsIncGen
)
