################################################################################
# RefIncGen
#
# Ref dialect generated includes.
################################################################################

add_custom_target(RefIncGen)

mlir_gen_enums(Ref)

add_subdirectory(IR)
