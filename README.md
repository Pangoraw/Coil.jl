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
julia> using Coil

julia> f(x) = sum(x)
f (generic function with 1 method)

julia> @code_mlir f([1,2,3])
MModule:
module {
  func.func @f(%arg0: tensor<3xi64>) -> i64 {
    %cst = arith.constant dense<0> : tensor<i64>
    %0 = mhlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
     reducer(%arg1: tensor<i64>, %arg2: tensor<i64>)  {
      %3 = mhlo.add %arg1, %arg2 : tensor<i64>
      mhlo.return %3 : tensor<i64>
    }
    %1 = mhlo.reshape %0 : (tensor<i64>) -> tensor<1xi64>
    %c0_i64 = arith.constant 0 : i64
    %2 = arith.index_cast %c0_i64 : i64 to index
    %extracted = tensor.extract %1[%2] : tensor<1xi64>
    return %extracted : i64
  }
}
```
