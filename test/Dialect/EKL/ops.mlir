// RUN: messner-opt %s | FileCheck %s

ekl.program {
    kernel @my_kernel(%a: !ekl.ref<ro, f64[...]>, %b: f64, %c: si32) {

    }

    func @my_func(%arg0: f64) -> f64 {
        yield %arg0: f64
    }

    func @sqrt(%arg0: f64) -> f64

    static public @test : !ekl.ref<ro, f64[3]> = <[0.0]>
}
