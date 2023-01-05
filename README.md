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

julia> f(x) = sum(exp, x)
f (generic function with 1 method)

julia> @code_mlir f(Float32[1., 2., 3.])
MModule:
module {
  func.func @f(%arg0: tensor<3xf32>) -> f32 {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<3xf32>) outs(%cst : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %1 = math.exp %in : f32
        %2 = arith.addf %1, %init : f32
        linalg.yield %2 : f32
      }
    %c0 = arith.constant 0 : index
    %0 = mhlo.reshape %reduced : (tensor<f32>) -> tensor<1xf32>
    %extracted = tensor.extract %0[%c0] : tensor<1xf32>
    return %extracted : f32
  }
}
```
