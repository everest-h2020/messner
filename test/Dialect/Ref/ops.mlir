// RUN: messner-opt %s | FileCheck %s

// CHECK-LABEL: func.func @read(
// CHECK: %[[ARG0:.+]]: !ref
func.func @read(%arg0: !ref<pr, i32>) -> i32 {
    // CHECK: %[[RET:.+]] = ref.read %[[ARG0]]
    %0 = ref.read %arg0 : !ref<pr, i32>
    // CHECK: return %[[RET]]
    return %0 : i32
}

// CHECK-LABEL: func.func @write(
// CHECK: %[[ARG0:.+]]: i32
// CHECK: %[[ARG1:.+]]: !ref
func.func @write(%arg0: i32, %arg1: !ref<iw, i32>) {
    // CHECK: ref.write %[[ARG0]] to %[[ARG1]]
    ref.write %arg0 to %arg1 : !ref<iw, i32>
    return
}
