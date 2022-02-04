# Ops.td #######################################################################

set(LLVM_TARGET_DEFINITIONS Ops.td)

mlir_tablegen(Base.h.inc -gen-dialect-decls -dialect=cfdlang)
mlir_tablegen(Base.cpp.inc -gen-dialect-defs -dialect=cfdlang)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=cfdlang)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=cfdlang)
#mlir_tablegen(Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=cfdlang)
#mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=cfdlang)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=cfdlang)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=cfdlang)

add_public_tablegen_target(CFDlangIRIncGen)

################################################################################

# Interfaces.td  ###############################################################

set(LLVM_TARGET_DEFINITIONS Interfaces.td)

#mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
#mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)
#mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
#mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)

add_public_tablegen_target(CFDlangInterfacesIncGen)

################################################################################

# Documentation ################################################################

add_mlir_doc(Ops CFDlangDialect CFDlang/ -gen-dialect-doc -dialect=cfdlang)

#add_mlir_doc(Interfaces CFDlangAttrInterfaces CFDlang/ -gen-attr-interface-docs)
add_mlir_doc(Interfaces CFDlangOpInterfaces CFDlang/ -gen-op-interface-docs)
#add_mlir_doc(Interfaces CFDlangTypeInterfaces CFDlang/ -gen-type-interface-docs)

################################################################################
