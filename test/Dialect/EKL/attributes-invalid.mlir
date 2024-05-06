// RUN: evp-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected string}}
#not_integer = #ekl.number<1.3991>

// -----

// expected-error@+1 {{expected binary rational}}
#not_double = #ekl.number<"1.3391">

// -----

// expected-error@+1 {{expected binary rational}}
#not_rational = #ekl.number<"1pa">

// -----

// expected-error@+1 {{expected 3 elements, but got 2}}
#array_shape = #ekl.array<[0 : si32, 1 : si32]> : !ekl.array<si32 [3]>

// -----

// expected-error@+1 {{type 'si64' of element #0 is not a subtype of 'si32'}}
#array_type = #ekl.array<[0 : si64, 0 : si32, 0 : si16]> : !ekl.array<si32 [3]>

// -----

// expected-error@+1 {{expected 4 elements, but got 1}}
#init_shape = #ekl.init<array<si32: 0>, [2, 2]>
