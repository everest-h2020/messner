################################################################################
# check-messner
#
# The messner regression test project.
################################################################################

project(check-messner)

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    evp-opt
    evp-translate
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the messner regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
