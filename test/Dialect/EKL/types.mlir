// RUN: messner-opt %s | FileCheck %s

// CHECK-LABEL: @expr()
func.func @expr() {
    // CHECK: to !ekl<?>
    %0 = builtin.unrealized_conversion_cast to !ekl.expr
    // CHECK: to !ekl<?>
    %1 = builtin.unrealized_conversion_cast to !ekl<?>
    return
}

// CHECK-LABEL: @Q()
func.func @Q() {
    // CHECK: to !ekl.Q
    %0 = builtin.unrealized_conversion_cast to !ekl.Q
    return
}

// CHECK-LABEL: @I()
func.func @I() {
    // CHECK: to !ekl.I
    %0 = builtin.unrealized_conversion_cast to !ekl.I
    // CHECK: to !ekl.I
    %1 = builtin.unrealized_conversion_cast to !ekl.I<?>
    // CHECK: to !ekl.I<1>
    %2 = builtin.unrealized_conversion_cast to !ekl.I<1>
    // CHECK: to !ekl.I<-1>
    %3 = builtin.unrealized_conversion_cast to !ekl.I<-1>
    // CHECK: to !ekl.I<-?>
    %4 = builtin.unrealized_conversion_cast to !ekl.I<-?>
    return
}

// CHECK-LABEL: @str()
func.func @str() {
    // CHECK: to !ekl.str
    %0 = builtin.unrealized_conversion_cast to !ekl.str
    return
}

// CHECK-LABEL: @array()
func.func @array() {
    // CHECK: to !ekl.array<f64>
    %0 = builtin.unrealized_conversion_cast to !ekl.array<f64>
    // CHECK: to !ekl.array<f64>
    %1 = builtin.unrealized_conversion_cast to !ekl.array<f64[]>
    // CHECK: to !ekl.array<f64[3, 4]>
    %2 = builtin.unrealized_conversion_cast to !ekl.array<f64[3, 4]>
    return
}

// CHECK-LABEL: @ref()
func.func @ref() {
    // CHECK: to !ekl.ref<ro, f64>
    %0 = builtin.unrealized_conversion_cast to !ekl.ref<ro, f64>
    // CHECK: to !ekl.ref<ro, f64>
    %1 = builtin.unrealized_conversion_cast to !ekl.ref<ro, f64[]>
    // CHECK: to !ekl.ref<ro, f64[3, 4]>
    %2 = builtin.unrealized_conversion_cast to !ekl.ref<ro, f64[3, 4]>
    // CHECK: to !ekl.ref<ro, f64[...]>
    %4 = builtin.unrealized_conversion_cast to !ekl.ref<ro, f64[...]>
    // CHECK: to !ekl.ref<ro, f64[..., 3, 4]>
    %5 = builtin.unrealized_conversion_cast to !ekl.ref<ro, f64[..., 3, 4]>
    return
}

// CHECK-LABEL: @slice()
func.func @slice() {
    // CHECK: to !ekl.slice
    %0 = builtin.unrealized_conversion_cast to !ekl.slice
    // CHECK: to !ekl.slice
    %1 = builtin.unrealized_conversion_cast to !ekl.slice<:>
    // CHECK: to !ekl.slice
    %2 = builtin.unrealized_conversion_cast to !ekl.slice<::1>
    // CHECK: to !ekl.slice<0:>
    %3 = builtin.unrealized_conversion_cast to !ekl.slice<0:>
    // CHECK: to !ekl.slice<:end-1>
    %4 = builtin.unrealized_conversion_cast to !ekl.slice<:end-1>
    // CHECK: to !ekl.slice<0:end-1>
    %5 = builtin.unrealized_conversion_cast to !ekl.slice<0:end-1>
    // CHECK: to !ekl.slice<0:end-1>
    %6 = builtin.unrealized_conversion_cast to !ekl.slice<0:end-1:1>
    // CHECK: to !ekl.slice<1:>
    %7 = builtin.unrealized_conversion_cast to !ekl.slice<1:>
    // CHECK: to !ekl.slice<end-2:>
    %8 = builtin.unrealized_conversion_cast to !ekl.slice<end-2:>
    // CHECK: to !ekl.slice<:end-2>
    %9 = builtin.unrealized_conversion_cast to !ekl.slice<:end-2>
    // CHECK: to !ekl.slice<:1>
    %10 = builtin.unrealized_conversion_cast to !ekl.slice<:1>
    // CHECK: to !ekl.slice<0::2>
    %11 = builtin.unrealized_conversion_cast to !ekl.slice<0::2>
    return
}

// CHECK-LABEL: @axis()
func.func @axis() {
    // CHECK: to !ekl<*>
    %0 = builtin.unrealized_conversion_cast to !ekl.axis
    // CHECK: to !ekl<*>
    %1 = builtin.unrealized_conversion_cast to !ekl<*>
    return
}

// CHECK-LABEL: @ellipsis()
func.func @ellipsis() {
    // CHECK: to !ekl<...>
    %0 = builtin.unrealized_conversion_cast to !ekl.ellipsis
    // CHECK: to !ekl<...>
    %1 = builtin.unrealized_conversion_cast to !ekl<...>
    return
}
