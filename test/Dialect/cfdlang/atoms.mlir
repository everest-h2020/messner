// RUN: evp-opt %s | FileCheck %s

// CHECK-LABEL: cfdlang.program {
cfdlang.program {
    cfdlang.input @a : [3 3]
    cfdlang.input @b : [3 5]
    cfdlang.input @c : [3 ?]
    cfdlang.input @d : [? 3]
    cfdlang.input @e : [? ?]

    // CHECK-LABEL: cfdlang.define @y0
    cfdlang.define @y0 : [3 3] {
        // CHECK-DAG: %[[A:.+]] = cfdlang.eval @a
        %0 = cfdlang.eval @a : [3 3]
        // CHECK-NEXT: cfdlang.add %[[A]], %[[A]] : [3 3], [3 3]
        %1 = cfdlang.add %0, %0 : [3 3], [3 3]
        cfdlang.yield %1 : [3 3]
    }

    // CHECK-LABEL: cfdlang.define @y1
    cfdlang.define @y1 : [3 ? 3 5] {
        // CHECK-DAG: %[[C:.+]] = cfdlang.eval @c
        %0 = cfdlang.eval @c : [3 ?]
        // CHECK-DAG: %[[B:.+]] = cfdlang.eval @b
        %1 = cfdlang.eval @b : [3 5]
        // CHECK-NEXT: cfdlang.prod %[[C]], %[[B]] : [3 ?], [3 5]
        %2 = cfdlang.prod %0, %1 : [3 ?], [3 5]
        cfdlang.yield %2 : [3 ? 3 5]
    }

    cfdlang.define @y2 : [3 5] {
        // CHECK-DAG: %[[B:.+]] = cfdlang.eval @b
        %0 = cfdlang.eval @b : [3 5]
        // CHECK-DAG: %[[C:.+]] = cfdlang.eval @c
        %1 = cfdlang.eval @c : [3 ?]
        // CHECK-DAG: %[[CB:.+]] = cfdlang.prod %[[C]], %[[B]]
        %2 = cfdlang.prod %1, %0 : [3 ?], [3 5]
        // CHECK-DAG: %[[BCB:.+]] = cfdlang.prod %[[B]], %[[CB]]
        %3 = cfdlang.prod %0, %2 : [3 5], [3 ? 3 5]
        // CHECK-NEXT: cfdlang.cont %[[BCB]] : [3 5 3 ? 3 5] indices [1 5][2 4]
        %4 = cfdlang.cont %3 : [3 5 3 ? 3 5] indices [1 5][2 4]
        cfdlang.yield %4 : [3 5]
    }
}
