################################################################################
# EKLIncGen
#
# EKL dialect generated includes.
################################################################################

add_custom_target(EKLIncGen)

mlir_gen_enums(EKL)

add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)
