// RUN: messner-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected integer}}
func.func private @I_empty() -> !ekl.I<>

// -----

// expected-error@+1 {{expected integer}}
func.func private @I_not_integer() -> !ekl.I<a>

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{ScalarType}}
func.func private @array_empty() -> !ekl.array<>

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{ScalarType}}
func.func private @array_no_type() -> !ekl.array<[]>

// -----

// expected-error@+1 {{expected integer}}
func.func private @array_not_extent() -> !ekl.array<f64[a]>

// -----

// expected-error@+1 {{expected in-bounds}}
func.func private @array_zero_extent() -> !ekl.array<f64[0]>

// -----

// expected-error@+1 {{expected in-bounds}}
func.func private @array_unbounded_extent() -> !ekl.array<f64[-1]>
