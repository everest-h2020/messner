// RUN: messner-opt %s | FileCheck %s

// CHECK-LABEL: @Q()
// CHECK-DAG: one_0 = #ekl.Q<1>
// CHECK-DAG: one_1 = #ekl.Q<1>
// CHECK-DAG: f64_0 = #ekl.Q<"1.28125">
// CHECK-DAG: f64_1 = #ekl.Q<"1.28125">
// CHECK-DAG: pi_0 = #ekl.Q<"3.141592653589793">
// CHECK-DAG: pi_1 = #ekl.Q<"3.141592653589793">
// CHECK-DAG: f64_inf = #ekl.Q<"1P1024">
// CHECK-DAG: large = #ekl.Q<"1P1026">
func.func private @Q() attributes {
    one_0 = #ekl.Q<1>,
    one_1 = #ekl.Q<"4p-2">,
    f64_0 = #ekl.Q<"1.28125">,
    f64_1 = #ekl.Q<"328p-8">,
    pi_0 = #ekl.Q<"3.141592653589793">,
    pi_1 = #ekl.Q<"884279719003555p-48">,
    f64_inf = #ekl.Q<"1p1024">,
    large = #ekl.Q<"2p1025">
}

// CHECK-LABEL: @I()
// CHECK-DAG: zero = #ekl.I<0>
// CHECK-DAG: three = #ekl.I<3>
// CHECK-DAG: minus_three = #ekl.I<-3>
// CHECK-DAG: end_minus_three = #ekl.I<end-3>
// CHECK-DAG: end_plus_three = #ekl.I<end+3>
func.func private @I() attributes {
    zero = #ekl.I<0>,
    three = #ekl.I<3>,
    minus_three = #ekl.I<-3>,
    end_minus_three = #ekl.I<end-3>,
    end_plus_three = #ekl.I<end+3>
}

// CHECK-LABEL: @array()
// CHECK-DAG: short = #ekl.array<[#ekl.I<0>, #ekl.I<1>, #ekl.I<2>]> : !ekl.array<!ekl.I[3]>
// CHECK-DAG: long = #ekl.array<[false, false, false, true, true, true, true, false, false, false]> : !ekl.array<i1[10]>,
// CHECK-DAG: splat = #ekl.array<[false]> : !ekl.array<i1[3]>
// CHECK-DAG: covariant = #ekl.array<[#ekl.Q<1>, -3 : si32, 3.000000e+00]> : !ekl.array<!ekl.Q[3]>
// CHECK-DAG: bcast = #ekl.array<[#ekl.array<[false]> : !ekl.array<i1[3]>, true]> : !ekl.array<i1[2, 3]>
func.func private @array() attributes {
    short = #ekl.array<[#ekl.I<0>, #ekl.I<1>, #ekl.I<2>]> : !ekl.array<!ekl.I[3]>,
    long = #ekl.array<[false, false, false, true, true, true, true, false, false, false]> : !ekl.array<i1[10]>,
    splat = #ekl.array<[false]> : !ekl.array<i1[3]>,
    covariant = #ekl.array<[#ekl.Q<1>, -3 : si32, 3.0 : f64]> : !ekl.array<!ekl.Q[3]>,
    bcast = #ekl.array<[#ekl.array<[false]> : !ekl.array<i1[3]>, true]> : !ekl.array<i1[2, 3]>
}

// CHECK-LABEL: @slice()
// CHECK-DAG: id_0 = #ekl<:>
// CHECK-DAG: id_1 = #ekl<:>
// CHECK-DAG: id_2 = #ekl<:>
// CHECK-DAG: id_3 = #ekl<:>
// CHECK-DAG: id_4 = #ekl<:>
// CHECK-DAG: id_5 = #ekl<:>
// CHECK-DAG: id_6 = #ekl<:>
// CHECK-DAG: take_front = #ekl.slice<:1>
// CHECK-DAG: take_back = #ekl.slice<end-1:>
// CHECK-DAG: drop_back = #ekl.slice<:end-1>
// CHECK-DAG: drop_front = #ekl.slice<1:>
// CHECK-DAG: even_0 = #ekl.slice<::2>
// CHECK-DAG: even_1 = #ekl.slice<::2>
func.func private @slice() attributes {
    id_0 = #ekl<:>,
    id_1 = #ekl.slice<:>,
    id_2 = #ekl.slice<0:>,
    id_3 = #ekl.slice<:end>,
    id_4 = #ekl.slice<0:end>,
    id_5 = #ekl.slice<0:end:1>,
    id_6 = #ekl.slice<-1:end+1>,
    take_front = #ekl.slice<:1>,
    take_back = #ekl.slice<end-1:>,
    drop_back = #ekl.slice<:end-1>,
    drop_front = #ekl.slice<1:>,
    even_0 = #ekl.slice<0::2>,
    even_1 = #ekl.slice<::2>
}

// CHECK-LABEL: @axis()
// CHECK-DAG: qualified = #ekl<*>
// CHECK-DAG: implicit = #ekl<*>
func.func private @axis() attributes {
    qualified = #ekl.axis,
    implicit = #ekl<*>
}

// CHECK-LABEL: @ellipsis()
// CHECK-DAG: qualified = #ekl<...>
// CHECK-DAG: implicit = #ekl<...>
func.func private @ellipsis() attributes {
    qualified = #ekl.ellipsis,
    implicit = #ekl<...>
}

// CHECK-LABEL: @meta()
// CHECK-DAG: host_0 = #ekl.meta<host>
// CHECK-DAG: host_1 = #ekl.meta<host>
// CHECK-DAG: hdl = #ekl.meta<hdl("src/HDL/my_top.v", "my_top")>
func.func private @meta() attributes {
    host_0 = #ekl.meta<host>,
    host_1 = #ekl.meta<host()>,
    hdl = #ekl.meta<hdl("src/HDL/my_top.v", "my_top")>
}
