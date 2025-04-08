add_mlir_dialect_library(EKLDialect
        Assembly.cpp
        Attributes.cpp
        Dialect.cpp
        Ops.cpp
        Sema.cpp
        Traits.cpp
        Types.cpp
        TypeSystem.cpp
        TypeUtils.cpp

    DEPENDS
        EKLIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRFunctionInterfaces
        MLIRSideEffectInterfaces
        MLIRTyping
        RefDialect
        RefInterfaces
        EKLAnalysis
        EKLEnums
        EKLInterfaces)
