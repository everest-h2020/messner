// RUN: evp-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @foo_n
    // CHECK-SAME: -> !cfdlang.scalar
    func @foo_n(%0 : !cfdlang.scalar) -> !cfdlang.scalar {
        return %0 : !cfdlang.scalar
    }

    // CHECK-LABEL: func @foo_memref
    func @foo_memref(%0 : memref<2x2x!cfdlang.scalar>) -> !cfdlang.scalar {
        %zero = std.constant 0 : index
        // CHECK: memref.load
        // CHECK-SAME: !cfdlang.scalar
        %1 = memref.load %0[%zero, %zero] : memref<2x2x!cfdlang.scalar>
        return %1 : !cfdlang.scalar
    }
}
