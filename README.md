![EVEREST logo](img/logo_horiz_positive.png)

This software bundle is part of the [EVEREST][1] project, funded under an [EU grant][2].

# messner

This is the main repository of the `messner` project, which implements the DSL compiler for the EVEREST platform.

## Building

The `messner` project is built using **CMake** (version `3.20` or newer). Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

```sh
# Configure.
cmake -S . -B build \
    -G Ninja \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir

# Build.
cmake --build build
```

The following CMake variables can be configured:

|              Name | Type      | Description |
| ----------------: | :-------- | --- |
| `LLVM_DIR`        | `STRING`  | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `~/tools/llvm-12.0.1/lib/cmake/llvm`*        |
| `MLIR_DIR`        | `STRING`  | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-12.0.1/lib/cmake/mlir`*        |
| `GMP_ROOT`        | `PATH`    | Path prefix of a **GMP** installation or build. |
| `ISL_ROOT`        | `PATH`    | Path prefix of an **ISL** installation or build. |
| `MPFR_ROOT`       | `PATH`    | Path prefix of an **MPFR** installation or build. |

## License

This project is licensed under the ISC license.

---

![EU notice](img/eu_banner.png)

[1]: https://everest-h2020.eu/
[2]: https://cordis.europa.eu/project/id/957269
