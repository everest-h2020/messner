// RUN: messner-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected string}}
func.func private @Q_not_int_or_str() attributes { x = #ekl.Q<a> }

// -----

// expected-error@+1 {{expected rational}}
func.func private @Q_inf() attributes { x = #ekl.Q<"inf"> }

// -----

// expected-error@+1 {{expected rational}}
func.func private @Q_nan() attributes { x = #ekl.Q<"nan"> }

// -----

// expected-error@+1 {{expected binary rational}}
func.func private @Q_not_float() attributes { x = #ekl.Q<"1.2.e+4"> }

// -----

// expected-error@+2 {{Index}}
// expected-error@+1 {{expected index value}}
func.func private @I_not_integer() attributes { x = #ekl.I<a> }

// -----

// expected-error@+2 {{Index}}
// expected-error@+1 {{expected index value}}
func.func private @I_plus() attributes { x = #ekl.I<+1> }

// -----

// expected-error@+2 {{Index}}
// expected-error@+1 {{expected integer value}}
func.func private @I_minus_end() attributes { x = #ekl.I<-end> }

// -----

// expected-error@+2 {{Index}}
// expected-error@+1 {{expected index value}}
func.func private @I_end_plus_end() attributes { x = #ekl.I<end+end> }

// -----

// expected-error@+1 {{invalid kind of type}}
func.func private @array_not_array() attributes { x = #ekl.array<[0 : i32]> : i32 }

// -----

// expected-error@+1 {{stack can't be empty}}
func.func private @array_stack_empty() attributes { x = #ekl.array<[]> : !ekl.array<si32[3]> }

// -----

// expected-note@+2 {{'si64' is not a subtype of 'si32'}}
// expected-error@+1 {{type mismatch}}
func.func private @array_splat_not_covariant() attributes { x = #ekl.array<[0 : si64]> : !ekl.array<si32[3]> }

// -----

// expected-note@+2 {{'si64' is not a subtype of 'si32'}}
// expected-error@+1 {{type mismatch}}
func.func private @array_bcast_not_covariant() attributes { x = #ekl.array<[#ekl.array<[0 : si64]> : !ekl.array<si64[2]>]> : !ekl.array<si32[3, 2]> }

// -----

// expected-error@+1 {{expected splat}}
func.func private @array_scalar_not_splat() attributes { x = #ekl.array<[0 : si32, 1 : si32]> : !ekl.array<si32[]> }

// -----

// expected-error@+1 {{expected 1 element}}
func.func private @array_stack_size_mismatch() attributes { x = #ekl.array<[0 : si32, 1 : si32]> : !ekl.array<si32[1]> }

// -----

// expected-note@+2 {{[5] is not broadcastable to [4]}}
// expected-error@+1 {{shape mismatch}}
func.func private @array_bcast_failed() attributes { x = #ekl.array<[#ekl.array<[0 : si32]> : !ekl.array<si32[5]>]> : !ekl.array<si32[3, 4]> }

// -----

// expected-note@+2 {{[1, 3] is not broadcastable to [4, 1]}}
// expected-error@+1 {{shape mismatch}}
func.func private @array_bcast_failed() attributes { x = #ekl.array<[#ekl.array<[0 : si32]> : !ekl.array<si32[1, 3]>]> : !ekl.array<si32[3, 4, 1]> }
