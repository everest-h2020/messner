add_mlir_translation_library(CFDlangTranslation
    CLI.cpp
    Export.cpp
    Import.cpp

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRTranslation
        CFDlangDialect
        TeILDialect
)
