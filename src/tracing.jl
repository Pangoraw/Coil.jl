module Tracing

import NNlib
using Umlaut
using Umlaut: V

import ..Coil
using ..IREE
using ..MLIR
import ..func, ..mhlo, ..arith, ..math, ..tensor, ..tosa, ..linalg, ..index
import .MLIR: julia_type, get_result, get_type

macro loc(ctx)
    source = QuoteNode(__source__)
    quote
        Location($(esc(ctx)), $(source))
    end
end

value(v::Variable) = v.op.val
value(other) = other

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
end

Context(f) = Context(
    f,
    Dict{Variable,SymTensor}(),
    Dict{Variable,MLIR.Value}(),
    [],
)

_not_a_primitive() = nothing

function Umlaut.isprimitive(::Context, f, args...)
    # Check if lower_call! has a specialization
    if which(lower_call!, (Any,OpConfig{typeof(f)},Umlaut.Call)) !=
            which(lower_call!, (Any,OpConfig{typeof(_not_a_primitive)},Any))
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

function to_loc(mlir_ctx, line)
    if line isa String
        NEAR = "near "
        if startswith(line, NEAR)
            line = chopprefix(line, NEAR)
            file, line = split(line, ':')
            line = parse(Int, line)
            Location(mlir_ctx, String(file), line)
        else
            Location(mlir_ctx)
        end
    else
        Location(mlir_ctx, line)
    end
end

function get_arg_operand!(cg, v; cst_op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    if !(v isa V)
        cst = push!(block, cst_op(mlir_ctx, v; loc=@loc(mlir_ctx)))
        return MLIR.get_result(cst, 1)
    end

    v = bound(tape, v)
    if !haskey(ctx.operands, v)
        operand =
            MLIR.add_argument!(
                block,
                MType(mlir_ctx, v.op.val),
                to_loc(mlir_ctx, v.op.line),
            )
        add_arg!(cg, v, operand)
    end

    ctx.operands[v]
end

function get_arg_operands!(cg, op, args=op.args; cst_op=mhlo.constant)
    map(v -> get_arg_operand!(cg, v; cst_op), args)
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
function lower_call!(cg, ::OpConfig{typeof(NNlib.relu)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    operand = get_arg_operand!(cg, only(op.args); cst_op=mhlo.constant)
    type = MLIR.get_type(operand)
    loc = Location(mlir_ctx, op.line)

    T = julia_type(eltype(type))
    cst0 = push!(
        block,
        MLIR.is_tensor(type) ?
            arith.constant(mlir_ctx, zeros(T), type; loc) :
            arith.constant(mlir_ctx, zero(T); loc)
    )

    fop = if MLIR.is_tensor(type)
        tosa.maximum
    elseif T <: AbstractFloat
        arith.maxf
    elseif T <: Signed
        arith.maxsi
    elseif T <: Unsigned
        arith.maxui
    else
        throw("invalid type $T for relu")
    end

    push!(block, fop(mlir_ctx, [
        MLIR.get_result(cst0, 1),
        operand,
    ]; loc))
end
function lower_call!(cg, ::OpConfig{typeof(getindex)}, op)
    (; mlir_ctx, block, tape) = cg

    if !is_tensor(tape, first(op.args)) || length(op.args) != 2
        return
    end

    x = get_arg_operand!(cg, first(op.args); cst_op=mhlo.constant)
    i = get_arg_operand!(cg, last(op.args); cst_op=arith.constant)

    T = julia_type(get_type(i))
    @assert T <: Integer

    cst1 = push!(block, arith.constant(mlir_ctx, one(T)))
    zerobased0 = push!(block, arith.subi(mlir_ctx, [i, get_result(cst1, 1)]))
    index0 = push!(block, arith.index_cast(mlir_ctx, get_result(zerobased0, 1); loc=@loc(mlir_ctx)))
    push!(block, tensor.extract(mlir_ctx, x, get_result(index0, 1)))
end
function lower_call!(cg, ::OpConfig{typeof(sum)}, op)
    (; mlir_ctx, block) = cg

    @assert length(op.args) == 1 "unsupported multi-arguments sum"

    operand = get_arg_operand!(cg, only(op.args); cst_op=mhlo.constant)
    type = get_type(operand)
    shape = size(type)

    # # tosa
    # if ndims(type) != 1
    #     reshape0 = push!(block, tosa.reshape(mlir_ctx, operand, (prod(shape),); loc=@loc(mlir_ctx)))
    #     operand = get_result(reshape0, 1)
    # end

    # sum0 = push!(block, tosa.reduce_sum(mlir_ctx, operand, 1; loc=Location(mlir_ctx, op.line)))
    # cst0 = push!(block, arith.constant(mlir_ctx, 0; loc=@loc(mlir_ctx)))
    # index0 = push!(block, arith.index_cast(mlir_ctx, get_result(cst0, 1); loc=@loc(mlir_ctx)))

    # push!(block, tensor.extract(mlir_ctx, get_result(sum0, 1), get_result(index0, 1); loc=@loc(mlir_ctx)))

    # mhlo
    T = julia_type(eltype(type))
    unrankedT = MType(mlir_ctx, MType(mlir_ctx, T), ())

    cst0 = push!(block, arith.constant(mlir_ctx, zeros(T)))
    inner_block = Block(
        [unrankedT, unrankedT],
        [@loc(mlir_ctx), @loc(mlir_ctx)],
    )
    add0 = push!(inner_block, mhlo.add(mlir_ctx,
        [MLIR.get_argument(inner_block, 1), MLIR.get_argument(inner_block, 2)];
        loc=@loc(mlir_ctx)),
    )
    push!(inner_block, mhlo.return_(mlir_ctx, get_result(add0, 1); loc=@loc(mlir_ctx)))

    reduce0 = push!(block, mhlo.reduce(mlir_ctx, inner_block, [operand, get_result(cst0, 1)], 1:length(shape); loc=Location(mlir_ctx, op.line)))
    reshape2 = push!(block, mhlo.reshape(mlir_ctx, get_result(reduce0, 1), (1,); loc=@loc(mlir_ctx)))

    cst0 = push!(block, arith.constant(mlir_ctx, 0; loc=@loc(mlir_ctx)))
    index0 = push!(block, arith.index_cast(mlir_ctx, get_result(cst0, 1); loc=@loc(mlir_ctx)))

    push!(block, tensor.extract(mlir_ctx, get_result(reshape2, 1), get_result(index0, 1); loc=@loc(mlir_ctx)))
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.conv)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    operands = get_arg_operands!(cg, op, op.args[begin:end-1])

    output_type = MLIR.get_type(first(operands))
    conv0 = push!(block, mhlo.convolution(mlir_ctx, output_type, operands))

    ctx.operands[V(op)] = MLIR.get_result(conv0, 1)
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.materialize)}, op)
    (; tape) = cg
    ctx = tape.c
    ctx.operands[V(op)] =
        ctx.operands[first(op.args)]
    nothing
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.broadcasted)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    if length(op.args) == 2
        func_arg = first(op.args)
        f = func_arg isa Variable ? tape[func_arg].val : func_arg
        operand = get_arg_operand!(cg, last(op.args); cst_op=mhlo.constant)

        if f == identity
            ctx.operands[V(op)] = operand
        elseif f ∈ math_unary_functions
            fname = nameof(f)
            fop = getproperty(math, fname)
            mop = fop(mlir_ctx, operand; loc=Location(mlir_ctx, op.line))
            push!(block, mop)
            ctx.operands[V(op)] = MLIR.get_result(mop, 1)
        else
            type = get_type(operand)
            T = julia_type(eltype(type))
            val = zero(T) # use a placeholder arg

            inner_block = Block(MType[], Location[])

            unary_ctx = Context(f)
            if Umlaut.isprimitive(unary_ctx, f, val)
                unary_tape = Umlaut.Tape(unary_ctx)
                arg0 = push!(unary_tape, Umlaut.Input(val))
                unary_tape.result = push!(unary_tape, mkcall(f, arg0; val=f(val), line=op.line))
            else
                _, unary_tape = trace(f, val; ctx=unary_ctx)
            end

            unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape)
            lower_tape!(unary_cg)

            ret_operand = unary_ctx.operands[unary_tape.result]
            push!(inner_block, linalg.yield(mlir_ctx, ret_operand; loc=@loc(mlir_ctx)))

            empty0 = push!(block, tensor.empty(mlir_ctx, type; loc=@loc(mlir_ctx)))
            mapping = push!(block, linalg.map(
                mlir_ctx,
                inner_block,
                operand, MLIR.get_result(empty0, 1);
                loc=Location(mlir_ctx, op.line)
            ))

            ctx.operands[V(op)] = MLIR.get_result(mapping, 1)
        end

        return
    end

    input_shapes = [size(value(arg)) for arg in op.args[begin+1:end]]

    out_shape = Base.Broadcast.broadcast_shape(input_shapes...)
    operands = get_arg_operands!(cg, op, @view(op.args[begin+1:end]))
    element_types = [
        julia_type(eltype(get_type(operand)))
        for operand in operands
    ]
    out_element_type = Base.promote_type(element_types...)

    operands = map(operands) do operand
        type = MLIR.get_type(operand)
        shape = size(type)

        if julia_type(eltype(type)) != out_element_type && out_element_type <: AbstractFloat
            sitofp0 = push!(block,
                arith.sitofp(
                    mlir_ctx, operand,
                    MType(mlir_ctx, MType(mlir_ctx, out_element_type), shape);
                    loc=@loc(mlir_ctx)
                )
            )
            operand = MLIR.get_result(sitofp0, 1)
        end

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
    f = func_arg isa Variable ? tape[func_arg].val : func_arg
    op_type = if f == +
        :add
    elseif f == -
        :substract
    elseif f == /
        :divide
    elseif f == *
        :mul
    else
        throw("unsupported broadcast function $(f)")
    end

    out_type = MType(mlir_ctx, MType(mlir_ctx, out_element_type), out_shape,)
    fop = getproperty(mhlo, op_type)

    push!(block, fop(mlir_ctx, operands, out_type; loc=Location(mlir_ctx, op.line)))
end

function lower_call!(cg, ::OpConfig{typeof(fma)}, op)
    (; block, tape, mlir_ctx) = cg
    ctx = tape.c

    (a, b, c) = get_arg_operands!(cg, op)
    mul0 = push!(block, gen_arith!(cg, *, [a, b]; loc=Location(mlir_ctx, op.line)))
    add0 = push!(block, gen_arith!(cg, +, [MLIR.get_result(mul0, 1), c]; loc=Location(mlir_ctx, op.line)))

    ctx.operands[V(op)] = MLIR.get_result(add0, 1)

    # ctx.operands[V(op)] = MLIR.get_result(
    #     push!(block, math.fma(mlir_ctx, operands; loc=Location(mlir_ctx, op.line))),
    #     1,
    # )
end

function lower_unary!(cg, op, f)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    fname = nameof(f)
    fop = getproperty(math, fname)
    operand = get_arg_operand!(cg, first(op.args); cst_op=arith.constant)
    mop = fop(mlir_ctx, operand; loc=Location(mlir_ctx, op.line))
    push!(block, mop)

    ctx.operands[V(op)] = MLIR.get_result(mop, 1)
end

const math_unary_functions = (cos, sin, tan, tanh, exp, sqrt)
for f in math_unary_functions
    @eval function lower_call!(cg, ::OpConfig{typeof($f)}, op)
        lower_unary!(cg, op, $f)
    end
end

function lower_arith!(cg, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c
    f = op.fn

    operands = get_arg_operands!(cg, op; cst_op=arith.constant)
    mop = push!(block, gen_arith!(cg, f, operands; loc=Location(mlir_ctx, op.line)))

    ctx.operands[V(op)] =
        MLIR.get_result(mop, 1)
end

function _convertto!(mlir_ctx, block, a, Ta, T)
    if Ta <: Integer && T <: AbstractFloat
        a = get_result(push!(block, arith.sitofp(mlir_ctx, a, MType(mlir_ctx, T); loc=@loc(mlir_ctx))), 1)
    elseif Ta <: AbstractFloat && T <: AbstractFloat && sizeof(Ta) < sizeof(T)
        a = get_result(push!(block, arith.extf(mlir_ctx, a, MType(mlir_ctx, T); loc=@loc(mlir_ctx))), 1)
    else
        a
    end
end

function gen_arith!(cg, f, operands; loc=Location(cg.mlir_ctx))
    (; mlir_ctx, block) = cg

    if length(operands) > 2
        fop = gen_arith!(cg, f, operands[begin:begin+1]; loc)
        for i in 3:length(operands)
            push!(block, fop)
            fop = gen_arith!(cg, f, [MLIR.get_result(fop, 1), operands[i]]; loc)
        end
        return fop
    end
    @assert length(operands) == 2

    is_integer = all(operand -> MLIR.is_integer(MLIR.get_type(operand)), operands)
    if is_integer && f == /
        # cast operands to floats
        operands = map(operands) do operand
            sitofp0 = push!(block, arith.sitofp(mlir_ctx, operand; loc=@loc(mlir_ctx)))
            MLIR.get_result(sitofp0, 1)
        end
        is_integer = false
    end

    a, b = operands
    Ta = julia_type(get_type(a))
    Tb = julia_type(get_type(b))
    T = Base.promote_type(Ta, Tb)

    a = _convertto!(mlir_ctx, block, a, Ta, T)
    b = _convertto!(mlir_ctx, block, b, Tb, T)

    names = Dict{Function,Symbol}([
        (+) => :add,
        (-) => :sub,
        (*) => :mul,
        (/) => :div,
    ])
    t = names[f] == :div && is_integer ? :si :
        is_integer ? :i : :f

    fname = Symbol(names[f], t)
    fop = getproperty(arith, fname)
    fop(mlir_ctx, [a, b]; loc)
end

for f in (+, -, /)
    @eval function lower_call!(cg, ::OpConfig{typeof($f)}, op)
        lower_arith!(cg, op)
    end
end

function lower_call!(cg, ::OpConfig{typeof(*)}, op)
    (; mlir_ctx, block, tape) = cg

    if !all(arg -> is_tensor(tape, arg), op.args)
        lower_arith!(cg, op)
        return
    end

    operands = get_arg_operands!(cg, op)

    state = OperationState("mhlo.dot", Location(mlir_ctx, op.line))
    MLIR.add_operands!(state, operands)

    result_type = MType(mlir_ctx, op.val)
    MLIR.add_results!(state, [
        result_type,
    ])

    push!(block, Operation(state))
end

function lower_call!(cg, ::OpConfig{typeof(reshape)}, op)
    (; mlir_ctx, block) = cg

    operand = get_arg_operand!(cg, first(op.args); cst_op=mhlo.constant)

    current_size = size(MLIR.get_type(operand))
    out_size = value.(@view op.args[begin+1:end]) # const prod reshape sizes

    all_elements = prod(current_size)
    num_elements = prod(d -> d isa Colon ? 1 : d, out_size)
    out_size = [
        d == Colon() ? all_elements ÷ num_elements : d
        for d in out_size
    ]

    push!(
        block,
        mhlo.reshape(
            mlir_ctx, operand, out_size;
            loc=Location(mlir_ctx, op.line)
        ),
    )
end

function lower_call!(cg, op)
    f = op.fn isa Variable ?
        cg.tape[op.fn].val : op.fn
    out = lower_call!(cg, OpConfig{typeof(f)}(), op)
    if out isa MLIR.Operation
        if MLIR.num_results(out) == 1
            cg.tape.c.operands[bound(cg.tape, V(op))] =
                MLIR.get_result(out, 1)
        elseif MLIR.num_results(out) > 1
            throw("unsupported multi-outputs")
        end
    end
end

function compile_to_module(tape, args...; verbose)
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))
    typeof_args = args = map(Core.Typeof, args)
    m = which(ctx.f, typeof_args)

    mlir_ctx = MLIR.Context()
    Coil.IREE.register_all_dialects!(mlir_ctx)
    for dialect in ("func", "mhlo", "arith", "math", "tosa", "linalg", "tensor")
        MLIR.get_or_load_dialect(mlir_ctx, dialect)
    end

    location = Location(mlir_ctx, string(m.file), m.line)
    mlir_module = MModule(mlir_ctx, location)
    mlir_module_body = MLIR.get_body(mlir_module)
    mlir_func_state = OperationState("func.func", location)

    block = Block(MType[], Location[])

    cg_context = CodegenContext(mlir_ctx, block, tape)
    lower_tape!(cg_context)

    result_types = [MLIR.get_type(ctx.operands[bound(tape, tape.result)])]
    input_types = [
        MLIR.get_type(ctx.operands[bound(tape, arg)])
        for arg in ctx.block_args
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
        [ctx.operands[bound(tape, tape.result)]]
    ))

    funcop = Operation(mlir_func_state)
    push!(mlir_module_body, funcop)

    verbose && show(IOContext(stdout, :debug => false), funcop)

    return mlir_module
end

function compile_tape(tape, args...; verbose=haskey(ENV, "COIL_VERBOSE"))
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))

    mlir_module = compile_to_module(tape, args...; verbose)
    mlir_ctx = mlir_module.context

    graph_call = get_call(mlir_ctx, mlir_module, string("module.", func_name))

    result_op = tape[tape.result]
    graph_call_op = Umlaut.mkcall(graph_call, ctx.block_args...; val=result_op.val)
    tape.result = push!(tape, graph_call_op)

    verbose && display(tape)

    verbose && @info "simplifying"
    simplify_tape!(tape)

    verbose && display(tape)

    compiled_func = Umlaut.compile(tape)

    (args...) -> compiled_func(ctx.f, args...)
end

function lower_tape!(cg)
    (; tape) = cg
    ctx = tape.c

    for op in tape
        if (op isa Umlaut.Input ||
            op isa Umlaut.Call ||
            op isa Umlaut.Const) && is_trackable_tensor(op.val)

            ctx.shapes[V(op)] = SymTensor(op.val)
        end

        if op isa Umlaut.Call
            try
                lower_call!(cg, op)
            catch
                # @error "failed to lower" exception=(err,catch_backtrace())
                display(tape)
                rethrow()
            end
        end
    end
end

function compile(model; verbose=false)
    compiled_model = nothing
    return (args...) -> begin
        if isnothing(compiled_model)
            # session = get_session()
            # model = iree(session, model)

            ctx = Context(model)

            out, tape = trace(model, args...; ctx)
            verbose && display(tape)

            compiled_model = compile_tape(tape, args...; verbose)

            out
        else
            compiled_model(args...)
        end
    end
end

macro code_mlir(call)
    Meta.isexpr(call, :call) || throw("expected call Expr")
    quote
        f = $(esc(first(call.args)))
        args = $(esc(Expr(:vect, call.args[begin+1:end]...)))
        ctx = Context(f)
        _, tape = trace(f, args...; ctx)
        compile_to_module(tape, args...; verbose=false)
    end
end

function compile_to_bytecode(ctx, module_; input_type=MLIR.get_input_type(module_))
    input_type ∈ (
        :none,
        :mhlo,
        :xla,
        :tm_tensor,
        :tosa,
    ) || throw("invalid iree input type ($input_type)")

    pass = MLIR.PassManager(ctx)
    op_pass = MLIR.OpPassManager(pass)
    options = IREE.CompilerOptions([
        "--iree-hal-target-backends=llvm-cpu",
        string("--iree-input-type=", input_type),
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
