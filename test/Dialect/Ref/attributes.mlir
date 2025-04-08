// RUN: messner-opt %s | FileCheck %s

// CHECK-LABEL: @volatile()
// CHECK-DAG: x = #ref.volatile
func.func private @volatile() attributes {
    x = #ref.volatile
}
