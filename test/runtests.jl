using Test

using Coil
using Coil.IREE
using Coil.MLIR
import Coil: func, arith, mhlo

include("./simple_arith.jl")
include("./nnlib.jl")

@testset "Simple add function" begin
    @testset "Simple add function: $T" for T in (Int8, Int16, Int32, Int64, Float32, Float64)
        func = Coil.compile((a, b) -> a + b)

        for t in 1:2
            r = T(1):T(10)
            a = rand(r)
            b = rand(r)
            c = func(a, b)

            @test c == a + b broken = t == 2 && T == Float64
        end
    end
end

@testset "BufferView" begin
    @testset "BufferView: $N dims" for N in 0:3
        a = randn(Float32, (2 + i for i in 1:N)...)
        N == 0 && (a = fill(a))
        v = BufferView(Coil.Tracing.get_session(), a)

        @test a == v
    end
end

@testset "Tracing: Simple Function" begin
    f(x, y) = x .+ y
    cf = Coil.compile(f)

    a, b = rand(Int32, 1), rand(Int32, 1)
    @test f(a, b) == cf(a,b)

    a, b = rand(Int32, 1), rand(Int32, 1)
    @test f(a, b) ≈ cf(a,b)
end

@testset "Tracing: reshape" begin
  f(x) = reshape(x, 10, 10, 2)
  cf = Coil.compile(f; verbose=false)

  x = randn(Float32, 10 * 10 * 2)
  @test f(x) == cf(x)
  @test_broken f(x) == cf(x)
end
