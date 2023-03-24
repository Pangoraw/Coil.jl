# Coil.jl

> An experimental package to lower and execute Julia tensor operations to the IREE compiler stack using MLIR.

Coil exports only one function: `Coil.compile(f)` which returns a function which leverages [MLIR](https://mlir.llvm.org) and the [IREE](https://github.com/iree-org/iree) compiler stack to produce a (hopefully) faster version of `f`. Goals are the following:

 - Perform whole model analysis and optimizations to fuse and re-order operations across function calls.
 - Fold model hyperparameters by unrolling loops, control flow, etc...
 - Evaluate on different hardware accelerators using the [IREE](https://github.com/iree-org/iree) runtime.

> **Note**
> Note that Coil currently does not meet any of those goals and is also a way for me to learn about MLIR and IREE.

## Example usage

`Coil.compile` *should* return an equivalent and hopefully faster function. Note that like Julia, the function will compile when its first called.

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

## Tracing

To trace functions, Coil leverages [Umlaut.jl](https://github.com/dfdx/Umlaut.jl) which converts functions to linearized tapes. It then replaces lowerable calls from this tape to MLIR operations. Since not all Julia
calls can be replaced to MLIR operation (struct code, io, etc...), the transformation produce a new
tape where only tensor and arithmetic operations are lifted to MLIR dialects.

Consider this input tape of a `Flux.Dense` layer with bias and a relu activation:

```
julia> import Coil.Tracing

julia> dense = Dense(3, 6, relu)
Dense(3 => 6, relu)  # 24 parameters

julia> x = randn(Float32,3,1);

julia> _, tape = Tracing.trace(dense, x; ctx=Tracing.Context(dense));

julia> tape
inp %1::Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}
  inp %2::Matrix{Float32}
  const %3 = fast_act::typeof(NNlib.fast_act)
  %4 = getproperty(%1, :σ)::typeof(relu) 
  const %5 = nothing::Nothing
  const %6 = +::typeof(+)
  %7 = getproperty(%1, :weight)::Matrix{Float32} 
  %8 = *(%7, %2)::Matrix{Float32} 
  %9 = getproperty(%1, :bias)::Vector{Float32} 
  %10 = broadcasted(%6, %8, %9)::Broadcasted{} 
  %11 = broadcasted(%4, %10)::Broadcasted{} 
  %12 = materialize(%11)::Matrix{Float32} 

julia> Tracing.compile_tape(tape, x; verbose=true)
[...]
Tape{Coil.Tracing.Context}
  inp %1::Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}
  inp %2::Matrix{Float32}
  %3 = getproperty(%1, :weight)::Matrix{Float32} 
  %4 = getproperty(%1, :bias)::Vector{Float32} 
  %5 = Call(#= 3 => 1 =#)(%3, %2, %4)::Matrix{Float32} 
```

where the `Call` struct calls into the following generated MLIR function:

```julia
julia> Coil.@code_mlir dense(x)
MModule:
module {
  func.func @Dense(%arg0: tensor<6x3xf32>, %arg1: tensor<3x1xf32>, %arg2: tensor<6xf32>) -> tensor<6x1xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<6x3xf32>, tensor<3x1xf32>) -> tensor<6x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<6xf32>) -> tensor<6x1xf32>
    %2 = mhlo.add %0, %1 : tensor<6x1xf32>
    %3 = tensor.empty() : tensor<6x1xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<6x1xf32>
    %3 = arith.maxf %2, %cst : tensor<6x1xf32>
    return %3 : tensor<6x1xf32>
  }
}
```

### Control flow

Due to its use of [Umlaut.jl](https://github.com/dfdx/Umlaut.jl), all control flow from the input function is taken as is for the first given arguments. This means that loops and conditions are unrolled when applied to the linear tape.

## Building

To build [IREE](https://github.com/iree-org/iree) to be used as a shared library callable from Julia, you need to use a [custom fork](https://github.com/Pangoraw/iree/tree/build_coil):

```bash
git clone https://github.com/Pangoraw/iree
cd iree
git checkout build_coil2
git submodule update --init
cmake -GNinja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_HAL_DRIVER_VULKAN=on \
    -DIREE_TARGET_BACKEND_VULKAN_SPIRV=on \
    -DIREE_ENABLE_LLD=ON
cmake --build ../iree-build --target iree_runtime_runtime_shared
```

This will build the runtime library in the `iree-build/` folder. The runtime library (`lib_runtime_shared_shared`) contains the bytecode interpreter and hardware drivers to run IREE programs.

The compiler library (`libIREECompiler`) containing MLIR and IREE specific passes is downloaded using artifacts from [the official releases](https://github.com/openxla/iree/releases) (Linux x86_64 glibc only) when the package is instantiated.

Later, these libraries will be provided as _jll packages built using [Binary Builder](https://binarybuilder.org).

## Dependencies

This package is tested only on the Julia 1.9 release, therefore a special version of [CompilerPluginTools.jl](https://github.com/JuliaCompilerPlugins/CompilerPluginTools.jl) should be installed (see [CompilerPluginTools.jl#9](https://github.com/JuliaCompilerPlugins/CompilerPluginTools.jl/pull/9)):

```
(Coil) pkg> add https://github.com/JuliaCompilerPlugins/CompilerPluginTools.jl#roger/fix-1.9
```

## References

 - [ONNX.jl](https://github.com/FluxML/ONNX.jl) - Coil takes a very similar approach to ONNX.jl but lowers down to MLIR modules instead of ONNX operations. 
 - [XLA.jl](https://github.com/JuliaTPU/XLA.jl) - XLA lowers from Julia IR down to XLA HLO and can execute to TPU. Interestingly, the tensor shape inference is embedded in Julia's type system whereas Coil uses the runtime values collected during tracing.
