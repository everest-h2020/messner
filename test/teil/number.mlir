// RUN: evp-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @foo_n
    // CHECK-SAME: -> !teil.num
    func @foo_n(%0 : !teil.num) -> !teil.num {
        return %0 : !teil.num
    }

    // CHECK-LABEL: func @foo_n0
    // CHECK-SAME: -> !teil.num
    func @foo_n0(%0 : !teil.num<0>) -> !teil.num {
        return %0 : !teil.num
    }

    // CHECK-LABEL: func @foo_n1
    // CHECK-SAME: -> !teil.num<1>
    func @foo_n1(%0 : !teil.num<1>) -> !teil.num<1> {
        return %0 : !teil.num<1>
    }

    // CHECK-LABEL: func @foo_memref
    func @foo_memref(%0 : memref<2x2x!teil.num<1>>) -> !teil.num<1> {
        %zero = std.constant 0 : index
        // CHECK: memref.load
        // CHECK-SAME: !teil.num<1>
        %1 = memref.load %0[%zero, %zero] : memref<2x2x!teil.num<1>>
        return %1 : !teil.num<1>
    }
}
