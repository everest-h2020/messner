// RUN: evp-opt %s | FileCheck %s

// CHECK-LABEL: cfdlang.program @p0 {
cfdlang.program @p0 {
    // CHECK: cfdlang.input @a : [11 11]
    cfdlang.input @a : [11 11]

    // CHECK-LABEL: cfdlang.define @b : [11 11]
    cfdlang.define @b : [11 11] {
        // CHECK: %[[A:.+]] = cfdlang.eval @a : [11 11]
        %b = cfdlang.eval @a : [11 11]
        // CHECK: cfdlang.yield %[[A]] : [11 11]
        cfdlang.yield %b : [11 11]
    }

    // CHECK-LABEL: cfdlang.output @c : [11 11]
    cfdlang.output @c : [11 11] {
        // CHECK: %[[B:.+]] = cfdlang.eval @b : [11 11]
        %b = cfdlang.eval @b : [11 11]
        // CHECK: cfdlang.yield %[[B]] : [11 11]
        cfdlang.yield %b : [11 11]
    }
}
