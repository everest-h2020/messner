# Ops.td #######################################################################

set(LLVM_TARGET_DEFINITIONS Ops.td)

mlir_tablegen(Base.h.inc -gen-dialect-decls -dialect=teil)
mlir_tablegen(Base.cpp.inc -gen-dialect-defs -dialect=teil)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=teil)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=teil)
#mlir_tablegen(Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=teil)
#mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=teil)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=teil)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=teil)

add_public_tablegen_target(TeILIRIncGen)

################################################################################

# Interfaces.td  ###############################################################

set(LLVM_TARGET_DEFINITIONS Interfaces.td)

#mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
#mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)
#mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
#mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)

add_public_tablegen_target(TeILInterfacesIncGen)

################################################################################

# Documentation ################################################################

add_mlir_doc(Ops TeILDialect TeIL/ -gen-dialect-doc -dialect=teil)

#add_mlir_doc(Interfaces TeILAttrInterfaces TeIL/ -gen-attr-interface-docs)
add_mlir_doc(Interfaces TeILOpInterfaces TeIL/ -gen-op-interface-docs)
#add_mlir_doc(Interfaces TeILTypeInterfaces TeIL/ -gen-type-interface-docs)

################################################################################
