using Test

import NNlib
import Coil

@testset "NNlib" begin
    @testset "relu" begin
        f(x) = NNlib.relu(x)
        cf = Coil.compile(f; verbose=false)

        @test f(1) == cf(1)
        @test f(-1) == cf(-1)

        T = Float32
        cf = Coil.compile(f; verbose=false)

        x = randn(T)
        @test f(x) == cf(x)
        @test f(x) == cf(x)
        @test isnan(cf(T(NaN)))

        n = 10
        cf = Coil.compile(f; verbose=false)

        x = randn(T, n)
        @test f(x) == cf(x)
        @test f(x) == cf(x)
    end
end
