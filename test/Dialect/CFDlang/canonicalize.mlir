// RUN: evp-opt %s | FileCheck %s

cfdlang.program {
    cfdlang.input @a : [3 3]
    cfdlang.input @b : [3 3]
    cfdlang.input @c : [3 3]

    cfdlang.define @y0 : [3 3] {
        %0 = cfdlang.eval @a : [3 3]
        cfdlang.yield %0 : [3 3]
    }

    cfdlang.output @y1 : [3 3] {
        %0 = cfdlang.eval @a : [3 3]
        %1 = cfdlang.eval @a : [3 3]
        %2 = cfdlang.add %0, %1 : [3 3], [3 3]
        cfdlang.yield %2 : [3 3]
    }

    cfdlang.output @y2 : [3 3] {
        %0 = cfdlang.eval @a : [3 3]
        %1 = cfdlang.eval @b : [3 3]
        %x = cfdlang.prod %0, %1 : [3 3], [3 3]
        %y = cfdlang.cont %x : [3 3 3 3] indices [1 3][2 4]
        %3 = cfdlang.add %0, %0 : [3 3], [3 3]
        cfdlang.yield %3 : [3 3]
    }
}