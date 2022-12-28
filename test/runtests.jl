using Test

using Coil
using Coil.IREE
using Coil.MLIR

function compile_to_bytecode(ctx, module_; mhlo=false)
    pass = MLIR.PassManager(ctx)
    op_pass = MLIR.OpPassManager(pass)
    options = IREE.CompilerOptions([
        "--iree-hal-target-backends=llvm-cpu",
        (mhlo ? ("--iree-input-type=mhlo",) : ())...,
    ])

    IREE.build_vm_pass_pipeline!(op_pass, options)

    MLIR.run(pass, module_)
    return IREE.translate_module_to_vm_bytecode(module_, options)
end

function get_session()
    instance = IREE.Instance()
    device = IREE.Device(instance)

    session_options = IREE.SessionOptions()
    IREE.Session(instance, session_options, device)
end

function get_call(ctx, module_, name)
    bytecode = compile_to_bytecode(ctx, module_)
    session = get_session()

    IREE.append_bytecode!(session, bytecode)

    return IREE.Call(session, name)
end

@testset "Simple add function" begin
    @testset "Simple add function: $T" for T in (Int8, Int16, Int32, Int64, Float32, Float64)
        func = let
            ctx = MLIR.Context()
            Coil.IREE.register_all_dialects!(ctx)

            loc = Location(ctx)
            module_ = MModule(loc)

            modbody = MLIR.get_body(module_)

            mlirtype = MType(ctx, T)
            op = Coil.gen_func(ctx, "add", (mlirtype, mlirtype), (mlirtype,))
            push!(modbody, op)

            get_call(ctx, module_, "module.add")
        end

        for _ in 1:2
            r = T(1):T(10)
            a = rand(r)
            b = rand(r)
            c = func(a, b)

            @test c == a + b broken = T == Float64
        end
    end
end

@testset "BufferView" begin
    @testset "BufferView: $N dims" for N in 0:3
        a = randn(Float32, (2 + i for i in 1:N)...)
        N == 0 && (a = fill(a))
        v = BufferView(get_session(), a)

        @test a == v
    end
end
