// RUN: evp-opt %s | FileCheck %s

// CHECK-LABEL: cfdlang.program {
cfdlang.program {

}

// CHECK-LABEL: cfdlang.program @my_prog {
cfdlang.program @my_prog {

}

module {

    // CHECK-LABEL: cfdlang.program @nested_prog {
    cfdlang.program @nested_prog {

    }

}
