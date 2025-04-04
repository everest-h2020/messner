// RUN: messner-opt %s -split-input-file -verify-diagnostics

func.func @read_not_readable(%arg0: !ref<w, i32>) {
    // expected-error@+1 {{invalid kind of Type}}
    %0 = ref.read %arg0 : !ref<w, i32>
    return
}

// -----

func.func @read_mismatched_value(%arg0: !ref<r, i32>) {
    // expected-error@+2 {{incompatible with return type}}
    // expected-error@+1 {{failed to infer returned types}}
    %0 = "ref.read" (%arg0) : (!ref<r, i32>) -> i64
    return
}

// -----

func.func @write_not_writable(%arg0: i32, %arg1: !ref<r, i32>) {
    // expected-error@+1 {{invalid kind of Type}}
    ref.write %arg0 to %arg1 : !ref<r, i32>
    return
}

// -----

// expected-note@+1 {{prior use here}}
func.func @write_mismatched_value(%arg0: i64, %arg1: !ref<w, i32>) {
    // expected-error@+1 {{expects different type}}
    ref.write %arg0 to %arg1 : !ref<w, i32>
    return
}
