# TODO: Add license.

################################################################################
# CFDlangTranslation
#
# The CFDlang translation library.
################################################################################

project(CFDlangTranslation)

find_package(BISON 3.0.1 REQUIRED)
find_package(REflex 3.0 REQUIRED)

# Generate the lexer/parser pair.
reflex_target(lexer
	${CMAKE_CURRENT_SOURCE_DIR}/Lexer.lxx
	${CMAKE_CURRENT_BINARY_DIR}/Lexer.cpp
	DEFINES_FILE
		${CMAKE_CURRENT_BINARY_DIR}/Lexer.hpp
)
bison_target(parser
	${CMAKE_CURRENT_SOURCE_DIR}/Parser.yxx
	${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp
	DEFINES_FILE
		${CMAKE_CURRENT_BINARY_DIR}/Parser.hpp
)
add_reflex_bison_dependency(lexer parser)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_mlir_translation_library(${PROJECT_NAME}
        Utils/ParseDriver.cpp
        Utils/PrintDriver.cpp

        CLI.cpp
        Export.cpp
        Import.cpp

        ${REflex_lexer_OUTPUTS}
        ${BISON_parser_OUTPUTS}

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRTranslation
        CFDlangDialect
        TeILDialect
        REflex::libreflex
)
