// RUN: messner-opt %s | FileCheck %s

// CHECK-LABEL: func.func @read(
// CHECK: %[[ARG0:.+]]: !ref
func.func @read(%arg0: !ref<ro, i32>) -> (i32, i32, i32, i32) {
    // CHECK: %[[RET0:.+]] = ref.read %[[ARG0]]
    %0 = ref.read %arg0 : !ref<ro, i32>
    // CHECK: %[[RET1:.+]] = ref.read %[[ARG0]] volatile
    %1 = ref.read %arg0 volatile : !ref<ro, i32>
    // CHECK: %[[RET2:.+]] = ref.read %[[ARG0]] impure
    %2 = ref.read %arg0 impure : !ref<ro, i32>
    // CHECK: %[[RET3:.+]] = ref.read %[[ARG0]] volatile impure
    %3 = ref.read %arg0 volatile impure : !ref<ro, i32>
    // CHECK: return %[[RET0]], %[[RET1]], %[[RET2]], %[[RET3]]
    return %0, %1, %2, %3 : i32, i32, i32, i32
}

// CHECK-LABEL: func.func @write(
// CHECK: %[[ARG0:.+]]: i32
// CHECK: %[[ARG1:.+]]: !ref
func.func @write(%arg0: i32, %arg1: !ref<unique, i32>) {
    // CHECK: ref.write %[[ARG0]] to %[[ARG1]]
    ref.write %arg0 to %arg1 : !ref<unique, i32>
    // CHECK: ref.write %[[ARG0]] to %[[ARG1]] volatile
    ref.write %arg0 to %arg1 volatile : !ref<unique, i32>
    return
}
