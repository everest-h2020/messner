// RUN: messner-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{unknown type}}
func.func private @unqualified_no_kind() -> !ref<i64>

// -----

// expected-error@+1 {{invalid reference kind}}
func.func private @unqualified_invalid_kind() -> !ref<"unknown", i64>

// -----

// expected-error@+1 {{expected ','}}
func.func private @unqualified_no_comma() -> !ref<"r|p" i64>

// -----

// expected-error@+1 {{expected non-function type}}
func.func private @unqualified_no_type() -> !ref<"r|p", "i64">
