// RUN: evp-opt %s | FileCheck %s

// CHECK-LABEL: cfdlang.program @p0 {
cfdlang.program @p0 {
    // CHECK: cfdlang.input @a : [11 11]
    cfdlang.input @a : [11 11]

    // CHECK-LABEL: cfdlang.define @b : [11 11]
    cfdlang.define @b : [11 11] {
        // CHECK: cfdlang.eval @a : [11 11]
        %0 = cfdlang.eval @a : [11 11]

        // CHECK: cfdlang.yield %0 : [11 11]
        cfdlang.yield %0 : [11 11]
    }

    // CHECK-LABEL: cfdlang.output @c : [11 11]
    cfdlang.output @c : [11 11] {
        // CHECK: cfdlang.eval @b : [11 11]
        %0 = cfdlang.eval @b : [11 11]

        // CHECK: cfdlang.yield %0 : [11 11]
        cfdlang.yield %0 : [11 11]
    }
}
