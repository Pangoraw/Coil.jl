module Tracing

using Umlaut
using Umlaut: V

import ..Coil
using ..IREE
using ..MLIR
import ..func, ..mhlo

struct SymTensor{T,R}
    dims::NTuple{R,Int}
end

SymTensor(x::AbstractArray{T,R}) where {T,R} = SymTensor{T,R}(size(x))

is_trackable_tensor(x::AbstractArray{<:Number}) = true
is_trackable_tensor(x) = false

struct Context
    f
    shapes::Dict{Variable,SymTensor}
    operands::Dict{Variable,MLIR.Value}
    block_args::Vector{Variable}
    session::IREE.Session
end

Context(f, session=get_session()) = Context(
    f,
    Dict{Variable,SymTensor}(),
    Dict{Variable,MLIR.Value}(),
    [],
    session,
)

function Umlaut.isprimitive(ctx::Context, f, args...)
    if f isa Function && nameof(f) == :conv
        return true
    end

    return Umlaut.isprimitive(Umlaut.BaseCtx(), f, args...)
end

struct CompiledModel
    inputs::Vector{SymTensor}
    model::Any
end

function is_tensor(tape::Tape{Context}, v::Umlaut.Variable)
    v = bound(tape, v)
    haskey(tape.c.shapes, v)
end
is_tensor(_, other) = false

struct OpConfig{F} end

struct CodegenContext
    mlir_ctx::MLIR.Context
    block::Block
    tape::Tape{Context}
end

function add_arg!(cg::CodegenContext, val, operand)
    (; tape) = cg
    ctx = tape.c
    val = bound(tape, val)
    ctx.operands[val] = operand
    push!(ctx.block_args, val)
    val
end

function lower_call!(cg, ::OpConfig{F}, op) where {F}
    @warn "skipping lowering of $(nameof(F))"
    nothing
end
if isdefined(Main, :Flux)
function lower_call!(cg, ::OpConfig{typeof(Main.Flux.NNlib.conv)}, op)
    @info "lowering conv" args = op.args
end
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.materialize)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c
    ctx.operands[V(op)] =
        ctx.operands[first(op.args)]
    nothing
end

function lower_relu!(block, operands)

end

function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.broadcasted)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    if length(op.args) == 2
        func_arg = bound(tape, first(op.args))
        broadcast = bound(tape, last(op.args))
        f = tape[func_arg].val

        if f == identity
            ctx.operands[V(op)] =
                ctx.operands[broadcast]
        elseif nameof(f) == :relu
            operand = ctx.operands[broadcast]
            type = MLIR.get_type(operand)
            unrankedT = MType(mlir_ctx, eltype(type), ())

            inner_block = Block([unrankedT], [Location(mlir_ctx)])

            relu0 = mhlo.relu!(
                inner_block,
                mlir_ctx,
                MLIR.get_argument(inner_block, 1);
                loc=Location(mlir_ctx, op.line),
            )
            push!(inner_block,
                mhlo.return_(
                    mlir_ctx,
                    MLIR.get_result(relu0, 1),
                )
            )
            mapping = mhlo.map(
                mlir_ctx,
                inner_block,
                operand;
                loc=Location(mlir_ctx, op.line),
            )

            ctx.operands[V(op)] =
                MLIR.get_result(mapping, 1)
            push!(block, mapping)
        end

        return
    end

    input_shapes = [size(arg.op.val) for arg in op.args[begin+1:end]]

    out_shape = Base.Broadcast.broadcast_shape(input_shapes...)
    out_element_type = Base.promote_type(
        (eltype(arg.op.val) for arg in op.args[begin+1:end])...
    )

    operands = map(zip(@view(op.args[begin+1:end]), input_shapes)) do (v, shape)
        v = bound(tape, v)
        if !haskey(ctx.operands, v)
            operand =
                MLIR.add_argument!(
                    block,
                    MType(mlir_ctx, v.op.val),
                    Location(mlir_ctx, v.op.line),
                )
            add_arg!(cg, v, operand)
        end
        operand = ctx.operands[v]

        if shape != out_shape
            reshape_op = push!(
                block,
                mhlo.broadcast_in_dim(
                    mlir_ctx, operand, out_shape;
                    loc=Location(mlir_ctx, op.line)
                ),
            )
            operand = MLIR.get_result(reshape_op, 1)
        end

        operand
    end

    func_arg = first(op.args)
    op_type = if func_arg isa Variable
        f = tape[func_arg].val
        if f == +
            "mhlo.add"
        elseif f == -
            "mhlo.substract"
        elseif f == /
            "mhlo.divide"
        elseif f == *
            "mhlo.mul"
        else
            throw("unsupported broadcast function $(f)")
        end
    else
        throw("unsupported broadcast function $(first(op.args).val)")
    end

    state = OperationState(op_type, Location(mlir_ctx, op.line))
    MLIR.add_operands!(state, operands)
    MLIR.add_results!(state, [
        MType(mlir_ctx, MType(mlir_ctx, out_element_type), out_shape,)
    ])

    push!(block, Operation(state))
end
function lower_call!(cg, ::OpConfig{typeof(*)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    if !all(arg -> is_tensor(tape, arg), op.args)
        @info "skipping lowering of *" args = op.args
        return
    end

    operands = map(op.args) do v
        if !(v isa V)
            cst = push!(block, mhlo.constant(mlir_ctx, v))
            return MLIR.get_result(cst, 1)
        end

        v = bound(tape, v)
        if !haskey(ctx.operands, v)
            operand =
                MLIR.add_argument!(
                    block,
                    MType(mlir_ctx, v.op.val),
                    Location(mlir_ctx, v.op.line),
                )
            add_arg!(cg, v, operand)
        end
        ctx.operands[v]
    end

    state = OperationState("mhlo.dot", Location(mlir_ctx, op.line))
    MLIR.add_operands!(state, operands)

    result_type = MType(mlir_ctx, op.val)
    MLIR.add_results!(state, [
        result_type,
    ])

    push!(block, Operation(state))
end

function lower_call!(cg, op)
    out = lower_call!(cg, OpConfig{typeof(op.fn)}(), op)
    if out isa MLIR.Operation
        if MLIR.num_results(out) == 1
            cg.tape.c.operands[bound(cg.tape, V(op))] =
                MLIR.get_result(out, 1)
        elseif MLIR.num_results(out) > 1
            throw("unsupported multi-outputs")
        end
    end
end

function compile_tape(tape, args...)
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))
    typeof_args = args = map(Core.Typeof, args)
    m = which(ctx.f, typeof_args)

    mlir_ctx = MLIR.Context()
    Coil.IREE.register_all_dialects!(mlir_ctx)
    MLIR.get_or_load_dialect(mlir_ctx, "func")
    MLIR.get_or_load_dialect(mlir_ctx, "mhlo")

    mlir_module = MModule(Location(mlir_ctx))
    mlir_module_body = MLIR.get_body(mlir_module)
    mlir_func_state = OperationState(
        "func.func",
        Location(mlir_ctx, string(m.file), m.line),
    )

    # Set of ops which are then used to call the computed model
    usable_inputs = [
        op
        for op in tape
        if op isa Umlaut.Input && !isnothing(op.val) &&
            (op.val isa AbstractArray || op.val isa Number)
    ]
    append!(ctx.block_args, V.(usable_inputs))

    input_types = [
        MType(mlir_ctx, op.val)
        for op in usable_inputs
    ]
    input_locations = [
        Location(mlir_ctx, op.line)
        for op in usable_inputs
    ]
    block = Block(input_types, input_locations)

    for (i, input) in enumerate(usable_inputs)
        ctx.operands[V(input)] =
            MLIR.get_argument(block, i)
    end
    cg_context = CodegenContext(mlir_ctx, block, tape)

    for op in tape
        if (op isa Umlaut.Input ||
            op isa Umlaut.Call ||
            op isa Umlaut.Const) && is_trackable_tensor(op.val)

            ctx.shapes[V(op)] = SymTensor(op.val)
        end

        if op isa Umlaut.Call
            # try
                lower_call!(cg_context, op)
            # catch err
                # @error "failed to lower" exception=(err,catch_backtrace())
            # end
        end
    end

    for i in 1:MLIR.num_arguments(block)
        if i <= length(input_types)
            continue
        end

        push!(input_types,
            MLIR.get_type(MLIR.get_argument(block, i)))
        push!(input_locations, Location(mlir_ctx))
    end

    result_op = tape[tape.result]
    result_types = [
        MType(mlir_ctx, result_op.val)
    ]
    ftype = MType(mlir_ctx, input_types => result_types)

    MLIR.add_attributes!(mlir_func_state, [
        NamedAttribute(mlir_ctx, "sym_name", MLIR.Attribute(mlir_ctx, string(func_name))),
        NamedAttribute(mlir_ctx, "function_type", MLIR.Attribute(ftype)),
    ])

    region = Region()
    push!(region, block)
    MLIR.add_owned_regions!(mlir_func_state, [region])

    push!(block, func.return_(mlir_ctx,
        # [MLIR.get_argument(block, 1)]
        [ctx.operands[bound(tape, tape.result)]]
    ))

    funcop = Operation(mlir_func_state)
    push!(mlir_module_body, funcop)

    display(tape)
    show(IOContext(stdout, :debug => true), funcop)

    graph_call = get_call(mlir_ctx, mlir_module, string("module.", func_name))
    session = graph_call.session

    graph_call_op = Umlaut.mkcall(graph_call, ctx.block_args...; val=result_op.val)
    graph_call_v = push!(tape, graph_call_op)
    tape.result = graph_call_v

    display(tape)

    @info "simplifying"
    simplify_tape!(tape)

    display(tape)

    compiled_func = Umlaut.compile(tape)

    (args...) -> compiled_func(ctx.f, args...)
end

function compile(model)
    compiled_model = nothing
    return (args...) -> begin
        if isnothing(compiled_model)
            # session = get_session()
            # model = iree(session, model)

            ctx = Context(model)

            out, tape = trace(model, args...; ctx)
            compiled_model = compile_tape(tape, args...)

            out
        else
            compiled_model(args...)
        end
    end
end

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
    bytecode = compile_to_bytecode(ctx, module_; mhlo=true)
    session = get_session()

    IREE.append_bytecode!(session, bytecode)

    return IREE.Call(session, name)
end

"""
    simplify_tape!(tape::Tape)

Removes any op that is not needed to compute the tape result.
"""
function simplify_tape!(tape)
    res = tape.result
    resop = tape[bound(tape, res)]

    if resop isa Umlaut.Input || resop isa Umlaut.Const
        filter!(!=(resop), tape.ops)
        return tape
    end

    universe = Set{Umlaut.Variable}([V(resop)])
    found_res = false

    for op in Iterators.reverse(tape.ops)
        if op isa Umlaut.Input
            continue
        end

        if op == resop
            found_res = true
            union!(universe, filter(a -> a isa V, op.args))
        end

        v = V(op)
        if !found_res # delete after result
            deleteat!(tape, v)
            continue
        end

        if v ∉ universe # delete unused
            deleteat!(tape, v)
            continue
        end

        if op isa Umlaut.Call
            union!(universe, filter(a -> a isa V, op.args))
        end
    end

    tape
end

import Adapt, Functors

struct BufferViewAdaptor
    session::IREE.Session
end

function Adapt.adapt_storage(adaptor::BufferViewAdaptor, a::AbstractArray)
    BufferView(adaptor.session, a)
end

iree(session, x) = Functors.fmap(
    x -> Adapt.adapt(BufferViewAdaptor(session), x), x
)

end # module Tracing
