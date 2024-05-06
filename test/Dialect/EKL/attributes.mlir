// RUN: evp-opt %s -split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @number_int
// CHECK-SAME: value = #ekl.number<1>
func.func @number_int() attributes {
    value = #ekl.number<1>
    } {
    return
}

// CHECK-LABEL: func.func @number_float
// CHECK-SAME: value1 = #ekl.number<"1.994250e+03">
// CHECK-SAME: value2 = #ekl.number<"1.994250e+03">
func.func @number_float() attributes {
    value1 = #ekl.number<"19942.5e-1">,
    value2 = #ekl.number<"7977P-2">
} {
    return
}

// CHECK-LABEL: func.func @number_rat
// CHECK-SAME: value = #ekl.number<"5558567840082035P-50">
func.func @number_rat() attributes {
    value = #ekl.number<"5558567840082035p-50">
} {
    return
}

// CHECK-LABEL: func.func @index
// CHECK-SAME: value1 = #ekl<_4>
// CHECK-SAME: value2 = #ekl<_3>
func.func @index() attributes {
    value1 = #ekl.index<4>,
    value2 = #ekl<_3>
} {
    return
}

// -----

#short = #ekl.array<[#ekl<_0>, #ekl<_1>, #ekl<_2>]> : !ekl.array<!ekl.index[3]>
// CHECK: #[[ARRAY:.+]] = #ekl.array<[false, false, false, true, true, true, true, false, false, false]> : !ekl.array<i1[10]>
#long = #ekl.array<[false, false, false, true, true, true, true, false, false, false]> : !ekl.array<i1[10]>

// CHECK-LABEL: @array_alias
// CHECK-SAME: value1 = #ekl.array<[#ekl<_0>, #ekl<_1>, #ekl<_2>]> : !ekl.array<!ekl.index[3]>
// CHECK-SAME: value2 = #[[ARRAY]]
func.func @array_alias() attributes {
    value1 = #short,
    value2 = #long
} {
    return
}

#covariant = #ekl.array<[#ekl<_0>, #ekl.number<1>, -3 : si32, 3.0 : f64]> : !ekl.array<!ekl.number[4]>

// CHECK-LABEL: @array_covariant
// CHECK-SAME: value = #ekl.array<[#ekl<_0>, #ekl.number<1>, -3 : si32, 3.000000e+00]> : !ekl.array<!ekl.number[4]>
func.func @array_covariant() attributes {
    value = #covariant
} {
    return
}

// -----

// CHECK: #[[INIT:.+]] = #ekl.init<array<ui16: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, [6, 2]>
#long = #ekl.init<array<ui16: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, [6, 2]>

// CHECK-LABEL: @init_alias
// CHECK-SAME: value1 = #ekl.init<array<si32: 0, 1, 2>, [3]>
// CHECK-SAME: value2 = #[[INIT]]
func.func @init_alias() attributes {
    value1 = #ekl.init<array<si32: 0, 1, 2>, [3]>,
    value2 = #long
} {
    return
}
