using Test

using Coil

@testset "Tracing: Simple Arithmetic" begin
    @testset "Tracing: Simple Value Function ($op)" for op in (+, -, *, /)
        @eval f(x, y) = $op(x, y)
        cf = Coil.compile(f)

        a, b = rand(Int32), rand(Int32)

        # IREE VM outputs Float64 as Float32
        out_type = typeof(f(a, b))
        cmp = out_type == Float64 ? Base.:≈ : Base.:(==)
        cast_type = out_type == Float64 ? Float32 : out_type

        @test cmp(cast_type(f(a, b)), cf(a,b))

        a, b = rand(Int32), rand(Int32)
        @test cmp(cast_type(f(a, b)), cf(a,b))
    end
end

@testset "Tracing: Nary operation" begin
    f(x) = +(x...)
    cf = Coil.compile(f; verbose=false)
    x = [1,2,3]

    @test f(x) == cf(x)
    @test f(x) == cf(x)
end

@testset "Tracing: Unary functions" begin
    f(x) = sqrt(x)
    cf = Coil.compile(f; verbose=false)

    x = rand(Float32)
    @test f(x) == cf(x)
    @test f(x) == cf(x)
end

@testset "Tracing: Broadcast of unary functions" begin
    f(x) = cos.(x) .+ 1
    cf = Coil.compile(f; verbose=false)

    x = randn(Float32, 10)
    @test f(x) ≈ cf(x)
    @test f(x) ≈ cf(x)
end

@testset "Tracing: sum $s" for s in ((2,2), (10,))
    f(x) = sum(x)
    cf = Coil.compile(f; verbose=false)

    x = randn(Float32, s...)
    @test f(x) == cf(x)
    @test f(x) ≈ cf(x)
end

@testset "Tracing: sum with dims" begin
    f(x) = sum(x; dims=1)
    cf = Coil.compile(f; verbose=false)

    x = randn(Float32, 5, 6)
    @test cf(x) ≈ cf(x)
end

@testset "Tracing: broadcasting of custom functions" begin
    f(x) = (x + 1) * 2
    g(x) = f.(x)
    cg = Coil.compile(g; verbose=false)

    x = randn(Float32, 10)
    @test g(x) == cg(x)
    @test g(x) == cg(x)
end

@testset "Tracing: fma" begin
    f(a, b, c) = fma(a, b, c)
    cf = Coil.compile(f)

    a = rand(Float32)
    b = rand(Float32)
    c = rand(Float32)

    @test f(a,b,c) == cf(a,b,c)
    @test f(a,b,c) ≈ cf(a,b,c)
end

@testset "Tracing: getindex" begin
    f(x) = x[1]
    cf = Coil.compile(f; verbose=false)

    a = [42,2,3]
    @test cf(a) == f(a)
    @test cf(a) == f(a)
end

@testset "Tracing: binary operators" begin
    f(x, y) = x .+ y
    cf = Coil.compile(f; verbose=false)

    x = randn(Float32, 3, 3)
    y = randn(Float32, 3, 3)
    @test cf(x, y) ≈ cf(x, y)
end
