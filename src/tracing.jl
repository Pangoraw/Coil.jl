"""
The Tracing module contains the code in charge of transforming provided
functions to linearized tapes (using [Umlaut.jl](https://github.com/dfdx/Umlaut.jl))
and then replacing lowerable calls from this tape to MLIR operations. Not all Julia
calls can be replaced to MLIR operation (struct code, io, etc...). Therefore, the transformation produce a new
tape where only tensor and arithmetic operations are lifted to MLIR.

Consider this input tape of `Flux.Dense` layer with bias and a relu activation:

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

where the `Call` struct calls into the following generated function:

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
"""
module Tracing

import NNlib
using Umlaut
using Umlaut: V

import ..Coil

using ..IREE
using ..IREE: Call

using ..MLIR
import .MLIR: julia_type, get_result, get_type

import ..func, ..mhlo, ..arith, ..math, ..tensor, ..tosa, ..linalg

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

const primitives = Set([
    Base.setindex!,
    eltype,
    Core.arrayset,
    getproperty,
    getfield,
    Core.apply_type,
    Tuple,
    DataType,
    NamedTuple,
    fieldtype,
    Umlaut.__new__,
])

function Umlaut.isprimitive(c::Context, f, args...)
    if f == Core.kwcall
        return Umlaut.isprimitive(c, args[2], args[3:end]...)
    end

    f isa Umlaut.Variable ||
        (f isa Function && parentmodule(f) == Core) ||
        f in primitives ||
        f isa DataType ||
        which(lower_call!, (Any, OpConfig{typeof(f)}, Umlaut.Call)) !=
            which(lower_call!, (Any, OpConfig{typeof(_not_a_primitive)}, Umlaut.Call))
end

function is_tensor(tape::Tape{Context}, v::Umlaut.Variable)
    ctx = tape.c
    v = bound(tape, v)
    haskey(ctx.operands, v) &&
        MLIR.is_tensor(get_type(ctx.operands[v]))
    # v = bound(tape, v)
    # haskey(tape.c.shapes, v)
end
is_tensor(_, other) = false

struct OpConfig{F} end

macro op_config_kw(f)
    quote
        OpConfig{<:Union{typeof($(esc(f))),Core.kwftype($(esc(f)))}}
    end
end

struct CodegenContext
    mlir_ctx::MLIR.Context
    block::Block
    tape::Tape{Context}
    verbose::Bool
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

function generic_lower_call(cg, ::OpConfig{F}, op) where {F}
    cg.verbose && @warn "skipping lowering of $(nameof(F))" isprimitive = Umlaut.isprimitive(cg.tape.c, op.fn, op.args...)
    nothing
end

struct KwCall{T} <: Umlaut.AbstractOp
    call::Umlaut.Call{T}
    kwargs::NamedTuple
end

function Base.getproperty(kwcall::KwCall, f::Symbol)
    f ∈ (:kwargs, :call) && return getfield(kwcall, f)
    Base.getproperty(kwcall.call, f)
end

function to_vkwargs(tape, arg)
    b = tape[arg]
    T = b.typ
    names = first(T.parameters)
    n = length(names)
    args = tape[b.args[1]].args
    NamedTuple{names,NTuple{n,Any}}(args)
end

function lower_call!(cg, config::OpConfig, op)
    generic_lower_call(cg, config, op)
    nothing
end
function lower_call!(cg, ::OpConfig{typeof(convert)}, op)
    # TODO: cast float types
    cg.tape.c.operands[V(op)] = get_arg_operand!(cg, op.args[2]; cst_op=arith.constant)
end
function lower_call!(cg, ::OpConfig{typeof(typeassert)}, op)
    cg.tape.c.operands[V(op)] = get_arg_operand!(cg, op.args[1]; cst_op=arith.constant)
end
function lower_call!(cg, ::OpConfig{typeof(Core.kwcall)}, op::Umlaut.Call{T}) where {T}
    f = value(op.args[2])
    kwargs = to_vkwargs(cg.tape, first(op.args))
    op.args = op.args[3:end]
    lower_call!(cg, OpConfig{typeof(f)}(), KwCall{T}(op, kwargs))
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.relu)}, op)
    (; mlir_ctx, block) = cg

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

    fop = if T <: AbstractFloat
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
function lower_call!(cg, config::OpConfig{typeof(getindex)}, op)
    (; mlir_ctx, block) = cg

    if length(op.args) != 2
        @warn "skipping getindex"
        return generic_lower_call(cg, config, op)
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
function lower_call!(cg, ::OpConfig{typeof(map)}, op)
    (; mlir_ctx, block) = cg

    fmap = value(op.args[1])
    # fred = value(op.args[2])
    @assert length(op.args) == 2

    operand = get_arg_operand!(cg, op.args[2]; cst_op=arith.constant)
    type = get_type(operand)

    val = first(value(op.args[2]))

    unary_ctx = Context(fmap)
    _, unary_tape = _trace(fmap, val; ctx=unary_ctx)

    inner_block = Block(MType[], Location[])
    unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape, cg.verbose)

    lower_tape!(unary_cg)

    push!(inner_block, linalg.yield(mlir_ctx, unary_ctx.operands[unary_tape.result]; loc=@loc(mlir_ctx)))

    out0 = push!(block, tensor.empty(mlir_ctx, type; loc=@loc(mlir_ctx)))
    push!(block, linalg.map(mlir_ctx, inner_block, operand, get_result(out0, 1); loc=Location(mlir_ctx, op.line)))
end
function lower_call!(cg, ::OpConfig{typeof(zero)}, op)
    (; mlir_ctx, block) = cg
    T = value(only(op.args))
    push!(block, mhlo.constant(mlir_ctx, op.val; loc=Location(mlir_ctx, op.line)))
end
function lower_call!(cg, ::OpConfig{typeof(reduce)}, op)
    op.fn = reduce
    pushfirst!(op.args, identity)
    lower_call!(cg, OpConfig{typeof(mapreduce)}(), op)
end
function lower_call!(cg, ::OpConfig{typeof(mapreduce)}, op)
    (; mlir_ctx, block) = cg

    fmap = value(op.args[1])
    fred = value(op.args[2])
    @assert length(op.args) == 3 "only mapreduce(f, op, a; init=x) is currently supported"

    fmapreduce = (a, b) -> fred(fmap(a), b)

    operand = get_arg_operand!(cg, op.args[3]; cst_op=mhlo.constant)
    type = get_type(operand)

    dims = op isa KwCall ? get(op.kwargs, :dims, Colon()) : Colon()
    if dims isa Umlaut.Variable
        dims = value(dims)
    end

    if dims isa Number
        dims = (dims,)
    elseif dims isa Colon
        dims = 1:ndims(type)
    end

    T = julia_type(eltype(type))
    output_size = [size(type, i) for i in 1:ndims(type) if !(i in dims)]
    init = if op isa KwCall && haskey(op.kwargs, :init)
        get(op.kwargs, :init)
    else
        zeros(T, output_size...)
    end

    valx = first(value(op.args[3]))
    vali = first(init)

    if init isa Umlaut.Variable
        throw("todo: unsupported dynamic init")
    end

    constant0 = push!(block, arith.constant(mlir_ctx, init; loc=@loc(mlir_ctx)))
    init_operand = get_result(constant0, 1)

    _, unary_tape = _trace(fmapreduce, valx, vali)

    inner_block = Block(MType[], Location[])
    unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape, cg.verbose)

    lower_tape!(unary_cg)
    unary_ctx = unary_tape.c

    push!(inner_block, linalg.yield(
        mlir_ctx, unary_ctx.operands[unary_tape.result]))

    reduce0 = push!(block, linalg.reduce(
        mlir_ctx, inner_block,
        [operand, init_operand],
        dims; loc=Location(mlir_ctx, op.line)
    ))

    if ndims(get_type(init_operand)) == 0
        index0 = push!(block, arith.constant(mlir_ctx, 0, MLIR.IndexType(mlir_ctx); loc=@loc(mlir_ctx)))
        reshape0 = push!(block, mhlo.reshape(mlir_ctx, get_result(reduce0, 1), (1,); loc=@loc(mlir_ctx)))
        push!(block, tensor.extract(mlir_ctx, get_result(reshape0, 1), get_result(index0, 1); loc=@loc(mlir_ctx)))
    else
        shape_with_dims = [i in dims ? 1 : size(type, i) for i in 1:ndims(type)]
        push!(block, mhlo.reshape(mlir_ctx, get_result(reduce0, 1), shape_with_dims; loc=@loc(mlir_ctx)))
    end
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.conv)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    @assert length(op.args) == 3 "lowering NNlib.conv: currently only supported conv(::AbstractArray, ::AbstractArray, ::DenseConvDims)"
    operands = get_arg_operands!(cg, op, op.args[begin:end-1])

    input_type = MLIR.get_type(first(operands))
    weight_type = MLIR.get_type(last(operands))
    W, H, _, B = size(input_type)

    cdims = value(op.args[end])
    @assert cdims isa NNlib.DenseConvDims "lowering NNlib.conv: unsupported convolution type $(typeof(cdims))"

    kw, kh = cdims.kernel_size
    pw, ph, _, _ = cdims.padding
    dw, dh = cdims.dilation
    sw, sh = cdims.stride

    h_out = (H + 2 * ph - dh * (kh - 1) - 1) ÷ sh + 1
    w_out = (W + 2 * pw - dw * (kw - 1) - 1) ÷ sw + 1
    c_out = size(weight_type, 4)
    output_size = (w_out, h_out, c_out, B)
    output_type = MType(mlir_ctx, eltype(input_type), output_size)

    conv0 = push!(block, mhlo.convolution(mlir_ctx, output_type, operands))
    ctx.operands[V(op)] = get_result(conv0, 1)
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.materialize)}, op)
    (; tape) = cg
    ctx = tape.c
    ctx.operands[V(op)] = get_arg_operand!(cg, first(op.args); cst_op=mhlo.constant)
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
        elseif f == NNlib.relu
            type = get_type(operand)
            T = julia_type(eltype(type))
            cst0 = push!(block, arith.constant(mlir_ctx, zeros(T), type; loc=Location(mlir_ctx, op.line)))
            mop = if T <: AbstractFloat
                arith.maxf
            elseif T <: Unsigned
                arith.maxui
            elseif T <: Signed
                arith.maxsi
            else
                throw("cannot compute max on element of type $T")
            end
            max0 = push!(block, mop(mlir_ctx, [operand, get_result(cst0, 1)]; loc=Location(mlir_ctx, op.line)))
            ctx.operands[V(op)] = MLIR.get_result(max0, 1)
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
            _, unary_tape = _trace(f, val; ctx=unary_ctx)

            unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape, cg.verbose)
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

    operands = get_arg_operands!(cg, op, @view(op.args[begin+1:end]))

    input_types = [get_type(operand) for operand in operands]
    input_shapes = [size(type) for type in input_types]
    element_types = [
        julia_type(eltype(input_type))
        for input_type in input_types
    ]
    out_element_type = Base.promote_type(element_types...)
    out_shape = Base.Broadcast.broadcast_shape(input_shapes...)

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
    f = value(op.fn)

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

    is_integer = all(operand -> MLIR.is_integer(eltype(get_type(operand))), operands)
    if is_integer && f == /
        # cast operands to floats
        operands = map(operands) do operand
            sitofp0 = push!(block, arith.sitofp(mlir_ctx, operand; loc=@loc(mlir_ctx)))
            MLIR.get_result(sitofp0, 1)
        end
        is_integer = false
    end

    a, b = operands
    Ta = julia_type(eltype(get_type(a)))
    Tb = julia_type(eltype(get_type(b)))
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

    operands = get_arg_operands!(cg, op)

    if !all(arg -> is_tensor(tape, arg), op.args)
        lower_arith!(cg, op)
        return
    end

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
    IREE.register_all_dialects!(mlir_ctx)
    for dialect in ("builtin", "func", "mhlo", "arith", "math", "tosa", "linalg", "tensor")
        MLIR.get_or_load_dialect(mlir_ctx, dialect)
    end

    location = Location(mlir_ctx, string(m.file), m.line)
    mlir_module = MModule(mlir_ctx, location)
    mlir_module_body = MLIR.get_body(mlir_module)
    mlir_func_state = OperationState("func.func", location)

    block = Block(MType[], Location[])

    cg_context = CodegenContext(mlir_ctx, block, tape, verbose)
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

    verbose && show(IOContext(stdout, :debug => true), funcop)

    return mlir_module
end

function compile_tape(tape, args...; verbose=haskey(ENV, "COIL_VERBOSE"))
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))

    mlir_module = compile_to_module(tape, args...; verbose)
    graph_call = get_call(mlir_module, string("module.", func_name))

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

"""
    _trace(f, args...; ctx)::Umlaut.Tape{Coil.Tracing.Context}

Wraps `Umlaut.trace` but does not recurse into the provided function
if it is a primitive and replace and instead make it such that the function
wraps the primitive instead.
"""
function _trace(f, args...; ctx=Context(f))
    if Umlaut.isprimitive(ctx, f, args...)
        out = f(args...)
        tape = Umlaut.Tape(ctx)
        vargs = map(arg -> push!(tape, Umlaut.Input(arg)), args)
        tape.result = Umlaut.record_primitive!(tape, f, vargs...)
        return out, tape
    end
    Umlaut.trace(f, args...; ctx)
end

"""
  compile(f; verbose::Bool=false, discard_first_outputs::Bool=false)::Function

Returns a compiled version of the provided function which will
lazily be compiled + specialized on its arguments on the first call to this function.

One can use `discard_first_outputs=true` to use return the result of the compiled function even
on the first run where the Julia result is available because it has been computed during tracing.

!!! note
    Coil relies on [Umlaut.jl](https://github.com/dfdx/Umlaut.jl) for tracing the function
    and therefore all control flow constructs are linearized using the arguments provided for
    the first compilation.

## Examples

```julia
julia> import Coil

julia> f(x) = sqrt(sum(x .+ 2))
f (generic function with 1 method)

julia> compiled_f = Coil.compile(f)
#29 (generic function with 1 method)

julia> compiled_f(Float32[1.]) # triggers compilation
1.7320508f0

julia> f(Float32[2.])          # compare with Julia version
2.0f0

julia> compiled_f(Float32[2.]) # runs the compiled version
2.0f0
```
"""
function compile(f; verbose=false, discard_first_outputs=false)
    compiled_f = nothing
    return (args...) -> begin
        if isnothing(compiled_f)
            # session = get_session()
            # f = iree(session, f)

            out, tape = _trace(f, args...)
            verbose && display(tape)

            compiled_f = compile_tape(tape, args...; verbose)

            if discard_first_outputs
                Base.invokelatest(compiled_f, args...)
            else
                out
            end
        else
            compiled_f(args...)
        end
    end
end

"""
    @tape(f(args...))::Tape{Coil.Tracing.Context}

Returns the corresponding Umlaut tape to this function call.
"""
macro tape(call)
    Meta.isexpr(call, :call) || throw("expected call Expr")
    quote
        let
            f = $(esc(first(call.args)))
            args = $(esc(Expr(:vect, call.args[begin+1:end]...)))
            _, tape = _trace(f, args...)
            tape
        end
    end
end

"""
    @code_mlir f(args...)

Returns the "func.func" operation generated from the provided function
and arguments. This helper is useful to investigate the generated code
from `Coil.compile`.
"""
macro code_mlir(call)
    Meta.isexpr(call, :call) || throw("expected call Expr")
    quote
        let
            f = $(esc(first(call.args)))
            args = $(esc(Expr(:vect, call.args[begin+1:end]...)))
            ctx = Context(f)
            _, tape = _trace(f, args...; ctx)
            compile_to_module(tape, args...; verbose=false)
        end
    end
end

"""
    @code_linalg f(args...)

Similar to `@code_mlir` but with with the dialect translation option
enabled "iree-mhlo-to-linalg-on-tensors".
"""
macro code_linalg(call)
    Meta.isexpr(call, :call) || throw("expected call Expr")
    quote
        let
            f = $(esc(first(call.args)))
            args = $(esc(Expr(:vect, call.args[begin+1:end]...)))
            ctx = Context(f)
            _, tape = _trace(f, args...; ctx)
            module_ = compile_to_module(tape, args...; verbose=false)
            lower_to_linalg!(module_)
        end
    end
end

function run_pipeline!(module_, pipeline)
    IREE.register_all_dialects!(module_.context)

    pm = MLIR.PassManager(module_.context)
    MLIR.enable_verifier!(pm)

    op_pass = MLIR.OpPassManager(pm)
    MLIR.add_pipeline!(op_pass, pipeline)

    MLIR.run(pm, module_)
    module_
end

function lower_to_linalg!(module_)
    run_pipeline!(module_, "iree-mhlo-to-linalg-on-tensors")
end

with_debug(val) = with_debug(stdout, val)
with_debug(io::IO, val) = show(IOContext(io, :debug => true), val)

### Compile utils

function compile_to_bytecode(module_; input_type=MLIR.get_input_type(module_))
    input_type ∈ (
        :none,
        :mhlo,
        :xla,
        :tm_tensor,
        :tosa,
    ) || throw("invalid iree input type ($input_type)")

    pm = MLIR.PassManager(module_.context)
    op_pass = MLIR.OpPassManager(pm)
    options = IREE.CompilerOptions([
        "--iree-hal-target-backends=llvm-cpu",
        string("--iree-input-type=", input_type),
    ])

    IREE.build_vm_pass_pipeline!(op_pass, options)

    @show op_pass

    MLIR.run(pm, module_)
    return IREE.translate_module_to_vm_bytecode(module_, options)
end

function get_session()
    instance = IREE.Instance()
    device = IREE.Device(instance)

    session_options = IREE.SessionOptions()
    IREE.Session(instance, session_options, device)
end

function get_call(module_, name)
    bytecode = compile_to_bytecode(module_)
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

### Functors stuff

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
