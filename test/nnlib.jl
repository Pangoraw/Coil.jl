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

    @testset "conv" begin
        w = reshape(Float32[
            2 0
            0 1
        ], 2, 2, 1, 1)
        x = reshape(Float32[
            2 0
            0 1
        ], 2, 2, 1, 1)
        cdims = NNlib.DenseConvDims(x, w)

        cconv = Coil.compile(NNlib.conv)
        @test cconv(x, w, cdims) == cconv(x, w, cdims) == NNlib.conv(x, w, cdims)

        cconv = Coil.compile(NNlib.conv)
        cdims = NNlib.DenseConvDims(x, w; flipkernel=true)
        @test cconv(x, w, cdims) == cconv(x, w, cdims) == NNlib.conv(x, w, cdims)
    end
end
