# Coil.jl

> An holiday experiment to lower and execute Julia tensor operations to the IREE compiler stack using MLIR.

Coil exports only one function: `Coil.compile(f)` which returns a function leverages [MLIR](https://mlir.llvm.org) and the [IREE](https://github.com/iree-org/iree) compiler stack to produce a (hopefully) faster version of `f`.

```julia
julia> using Coil, Flux

julia> dense = Dense(3, 6, relu)
Dense(3 => 6, relu)  # 24 parameters

julia> compiled_dense = Coil.compile(dense)
#23 (generic function with 1 method)

julia> x = randn(Float32,3,1);

julia> compiled_dense(x)
2.7212882f0

julia> compiled_dense(x)
2.7212882f0

julia> dense(x)
2.7212882f0
```

Other niceties include the `@code_mlir` and `@code_linalg` macros.

```julia
julia> 

```
