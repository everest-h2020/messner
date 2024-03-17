################################################################################
# evp-translate
#
# The messner translation driver.
################################################################################

project(evp-translate)

add_executable(${PROJECT_NAME}
    evp-translate.cpp
)

# Link all standard MLIR dialect and translation libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSupport
        ${dialect_libs}
        ${translation_libs}
)
