// RUN: messner-opt %s | FileCheck %s

// CHECK: @simple_qualified() -> !ref<r, i64>
func.func private @simple_qualified() -> !ref.type<r, i64>

// CHECK: @simple_unqualified() -> !ref<r, i64>
func.func private @simple_unqualified() -> !ref<r, i64>

// CHECK: @compound_qualified() -> !ref<pr, i64>
func.func private @compound_qualified() -> !ref.type<"r|p", i64>

// CHECK: @compound_unqualified() -> !ref<pr, i64>
func.func private @compound_unqualified() -> !ref<"r|p", i64>
