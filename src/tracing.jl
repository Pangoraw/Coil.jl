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
    %3 = arith.maximumf %2, %cst : tensor<6x1xf32>
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

import ..func, ..stablehlo, ..arith, ..math, ..tensor, ..tosa, ..linalg
import ..Passes

macro loc()
    source = __source__
    file = string(source.file)
    line = source.line
    quote
        Location($file, $line, 0)
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
    size,
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
        MLIR.is_tensor(get_type(ctx.operands[v])) || v.op.typ <: AbstractArray{<:Real}
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
    allow_scalar_args::Bool
end

function to_loc(line)
    if line isa String
        NEAR = "near "
        if startswith(line, NEAR)
            line = chopprefix(line, NEAR)
            file, line = split(line, ':')
            line = parse(Int, line)
            Location(String(file), line)
        else
            Location()
        end
    else
        Location()
    end
end

function get_arg_operand!(cg, v)
    (; block, tape) = cg
    ctx = tape.c

    if !(v isa V)
        cst = push!(block, make_constant(v; loc=@loc()))
        return MLIR.get_result(cst, 1)
    end

    v = bound(tape, v)
    if !haskey(ctx.operands, v)
        operand =
            MLIR.push_argument!(
                block,
                MLIRType(v.op.val),
                to_loc(v.op.line),
            )
        add_arg!(cg, v, operand)
    end

    ctx.operands[v]
end

function get_arg_operands!(cg, op, args=op.args)
    map(v -> get_arg_operand!(cg, v), args)
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

function make_constant(v; loc=Location())
    if v isa AbstractArray
        stablehlo.constant(v)
    else
        arith.constant(v)
    end
end

function to_vkwargs(tape, arg)
    b = tape[arg]
    T = b.typ
    names = first(T.parameters)
    n = length(names)
    args = @static if VERSION >= v"1.10"
        b.args[2:end]
    else
        tape[b.args[1]].args
    end
    NamedTuple{names,NTuple{n,Any}}(args)
end

function lower_call!(cg, config::OpConfig, op)
    generic_lower_call(cg, config, op)
    nothing
end
function lower_call!(cg, ::OpConfig{typeof(convert)}, op)
    # TODO: cast float types
    if !haskey(cg.tape.c.operands, op.args[2])
        return
    end
    cg.tape.c.operands[V(op)] = get_arg_operand!(cg, op.args[2])
end
function lower_call!(cg, ::OpConfig{typeof(typeassert)}, op)
    if !haskey(cg.tape.c.operands, op.args[1])
        return
    end
    cg.tape.c.operands[V(op)] = get_arg_operand!(cg, op.args[1])
end
function lower_call!(cg, ::OpConfig{typeof(Core.kwcall)}, op::Umlaut.Call{T}) where {T}
    f = value(op.args[2])
    kwargs = to_vkwargs(cg.tape, first(op.args))
    op.args = op.args[3:end]
    lower_call!(cg, OpConfig{typeof(f)}(), KwCall{T}(op, kwargs))
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.relu)}, op)
    (; block) = cg

    operand = get_arg_operand!(cg, only(op.args))
    type = MLIR.get_type(operand)
    loc = to_loc(op.line)

    T = julia_type(eltype(type))
    cst0 = push!(
        block,
        MLIR.is_tensor(type) ?
        arith.constant(zeros(T), type; loc) :
        arith.constant(zero(T); loc)
    )

    fop = if T <: AbstractFloat
        arith.maximumf
    elseif T <: Signed
        arith.maxsi
    elseif T <: Unsigned
        arith.maxui
    else
        throw("invalid type $T for relu")
    end

    push!(block, fop([
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

    if !(first(op.args).op isa Umlaut.Input || haskey(cg.tape.c.operands, first(op.args)))
        return
    end

    x = get_arg_operand!(cg, first(op.args))
    i = get_arg_operand!(cg, last(op.args))

    T = julia_type(get_type(i))
    @assert T <: Integer

    cst1 = push!(block, arith.constant(one(T)))
    zerobased0 = push!(block, arith.subi([i, get_result(cst1, 1)]))
    index0 = push!(block, arith.index_cast(get_result(zerobased0, 1); loc=@loc()))

    push!(block, tensor.extract(x, get_result(index0, 1)))
end
function lower_call!(cg, ::OpConfig{typeof(map)}, op)
    (; block) = cg

    fmap = value(op.args[1])
    # fred = value(op.args[2])
    @assert length(op.args) == 2

    if !haskey(cg.tape.c.operands, op.args[2])
        return
    end

    operand = get_arg_operand!(cg, op.args[2])
    type = get_type(operand)

    val = first(value(op.args[2]))

    unary_ctx = Context(fmap)
    _, unary_tape = _trace(fmap, val; ctx=unary_ctx)

    inner_block = Block(MLIRType[], Location[])
    unary_cg = CodegenContext(cg.mlir_ctx, inner_block, unary_tape, cg.verbose, true)

    lower_tape!(unary_cg)

    push!(inner_block, linalg.yield(unary_ctx.operands[unary_tape.result]; loc=@loc()))

    out0 = push!(block, tensor.empty(type; loc=@loc()))
    push!(block, linalg.map_as_generic(inner_block, operand, get_result(out0, 1); loc=to_loc(op.line)))
end
# function lower_call!(cg, ::OpConfig{typeof(zero)}, op)
#     (; mlir_ctx, block) = cg
#     push!(block, stablehlo.constant(op.val; loc=Location(mlir_ctx, op.line)))
# end
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

    operand = get_arg_operand!(cg, op.args[3])
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

    constant0 = push!(block, arith.constant(init; loc=@loc()))
    init_operand = get_result(constant0, 1)

    _, unary_tape = _trace(fmapreduce, valx, vali)

    inner_block = Block(MLIRType[], Location[])
    unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape, cg.verbose, true)

    lower_tape!(unary_cg)
    unary_ctx = unary_tape.c

    push!(inner_block, linalg.yield(
        unary_ctx.operands[unary_tape.result])
    )

    reduce0 = push!(block, linalg.reduce(
        inner_block,
        [operand, init_operand],
        dims; loc=to_loc(op.line)
    ))

    if ndims(get_type(init_operand)) == 0
        index0 = push!(block, arith.constant(0, MLIR.IndexType(); loc=@loc()))
        reshape0 = push!(block, stablehlo.reshape(get_result(reduce0, 1), (1,); loc=@loc()))
        push!(block, tensor.extract(get_result(reshape0, 1), get_result(index0, 1); loc=@loc()))
    else
        shape_with_dims = [i in dims ? 1 : size(type, i) for i in 1:ndims(type)]
        push!(block, stablehlo.reshape(get_result(reduce0, 1), shape_with_dims; loc=@loc()))
    end
end

function lower_call!(cg, ::OpConfig{typeof(NNlib.meanpool)}, op)
    (; mlir_ctx, block) = cg
    reduced = get_result(lower_reduce_window!(cg, stablehlo.add, op))
    output_type = get_type(reduced)

    inW, inH, _, _ = size(get_type(get_arg_operand!(cg, op.args[1])))
    outW, outH, _, _ = output_size = size(output_type)

    win_W = inW ÷ outW
    win_H = inH ÷ outH
    factor = win_W * win_H

    cst = get_result(push!(block, arith.constant(fill(julia_type(eltype(output_type))(factor)); loc=to_loc(op.line))))
    cst = get_result(push!(block, stablehlo.broadcast_in_dim(cst, output_size; loc=to_loc(op.line))))
    push!(block, stablehlo.divide([reduced, cst]; loc=to_loc(op.line)))
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.maxpool)}, op)
    lower_reduce_window!(cg, stablehlo.maximum, op)
end

function lower_reduce_window!(cg, reduce_op, op)
    (; mlir_ctx, block) = cg

    operand, dims = op.args
    operand = get_arg_operand!(cg, operand)
    dims = value(dims)

    @assert dims isa NNlib.PoolDims "unsupported NNlib.maxpool call"

    # kernel_size::NTuple{K, Int}
    # channels_in::Int

    # stride::NTuple{S, Int}
    # padding::NTuple{P, Int}
    # dilation::NTuple{D, Int}

    input_type = get_type(operand)
    W, H, C, B = size(input_type)
    T = eltype(input_type)

    window_dimensions = [NNlib.kernel_size(dims)..., 1, 1]
    window_strides = [NNlib.stride(dims)..., 1, 1]
    base_dilations = [1, 1, 1, 1]
    window_dilations = [NNlib.dilation(dims)..., 1, 1]
    padding = [NNlib.padding(dims)..., 0, 0][begin:4]
    padding = repeat(padding; inner=(1, 2))

    kw, kh = NNlib.kernel_size(dims)
    pw, ph, _, _ = NNlib.padding(dims)
    dw, dh = NNlib.dilation(dims)
    sw, sh = NNlib.stride(dims)
    h_out = (H + 2 * ph - dh * (kh - 1) - 1) ÷ sh + 1
    w_out = (W + 2 * pw - dw * (kw - 1) - 1) ÷ sw + 1

    output_size = (w_out, h_out, C, B)
    output_type = MLIRType(T, output_size)

    inner_block = let
        unranked_tensor_type = MLIRType(T, ())
        inner_block = MLIR.Block(
            [unranked_tensor_type, unranked_tensor_type],
            [Location(mlir_ctx, op.line), Location(mlir_ctx, op.line)],
        )
        val = MLIR.get_argument(inner_block, 1)
        init = MLIR.get_argument(inner_block, 2)
        maxop = push!(inner_block, reduce_op(mlir_ctx, [val, init]; loc=@loc()))
        out = get_result(maxop, 1)
        push!(inner_block, stablehlo.return_(mlir_ctx, out; loc=@loc()))
        inner_block
    end
    region = MLIR.Region()
    push!(region, inner_block)

    initop = push!(block, arith.constant(
        mlir_ctx,
        zeros(MLIR.julia_type(T),);
        loc=@loc()
    ))

    init_values = get_result(initop, 1)
    operands = [operand, init_values]

    push!(block, stablehlo.reduce_window(
        operands, region,
        window_dimensions,
        window_strides,
        base_dilations,
        window_dilations,
        padding,
        output_type;
        loc=to_loc(op.line)
    ))
end
function lower_call!(cg, ::OpConfig{typeof(NNlib.conv)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    @assert length(op.args) == 3 "lowering NNlib.conv: currently only supported conv(::AbstractArray, ::AbstractArray, ::DenseConvDims)"
    operands = get_arg_operands!(cg, op, op.args[begin:end-1])

    xop, wop = operands

    input_type = MLIR.get_type(xop)
    weight_type = MLIR.get_type(wop)
    @assert ndims(input_type) == 4 "NNlib.conv: only 2d convolution is currently supported"
    W, H, _, B = size(input_type)

    cdims = value(op.args[end])
    @assert cdims isa NNlib.DenseConvDims "lowering NNlib.conv: unsupported convolution type $(typeof(cdims))"

    kw, kh = NNlib.kernel_size(cdims)
    pw, ph, _, _ = NNlib.padding(cdims)
    dw, dh = NNlib.dilation(cdims)
    sw, sh = NNlib.stride(cdims)

    h_out = (H + 2 * ph - dh * (kh - 1) - 1) ÷ sh + 1
    w_out = (W + 2 * pw - dw * (kw - 1) - 1) ÷ sw + 1
    c_out = size(weight_type, 4)
    output_size = (w_out, h_out, c_out, B)
    output_type = MLIRType(eltype(input_type), output_size)

    padding = Int64[
        pw pw
        ph ph
    ]
    dilation = Int64[dw, dh]
    strides = Int64[sw, sh]

    if !NNlib.flipkernel(cdims)
        reverse_op = push!(block,
            stablehlo.reverse(
                wop,
                [1, 2];
                loc=to_loc(op.line)
            )
        )
        wop = get_result(reverse_op)
    end

    conv0 = push!(block,
        stablehlo.convolution(
            output_type,
            [xop, wop],
            padding,
            dilation,
            strides;
            loc=to_loc(op.line)
        ),
    )
    ctx.operands[V(op)] = get_result(conv0, 1)
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.materialize)}, op)
    (; tape) = cg
    ctx = tape.c
    if !haskey(ctx.operands, first(op.args))
        @debug "skipping materialization of broadcast"
        return
    end
    ctx.operands[V(op)] = get_arg_operand!(cg, first(op.args))
    nothing
end
function lower_call!(cg, ::OpConfig{typeof(Base.Broadcast.broadcasted)}, op)
    (; mlir_ctx, block, tape) = cg
    ctx = tape.c

    val_arg2 = value(op.args[2])
    if !(val_arg2 isa AbstractArray) && !haskey(ctx.operands, op.args[2])
        @debug "broadcast of non-array" T = typeof(value(op.args[2]))
        return
    end

    if length(op.args) == 2
        func_arg = first(op.args)
        f = func_arg isa Variable ? tape[func_arg].val : func_arg
        operand = get_arg_operand!(cg, last(op.args))

        if f == identity
            ctx.operands[V(op)] = operand
        elseif f == NNlib.relu
            type = get_type(operand)
            T = julia_type(eltype(type))
            cst0 = push!(block, arith.constant(zeros(T), type; loc=to_loc(op.line)))
            mop = if T <: AbstractFloat
                arith.maximumf
            elseif T <: Unsigned
                arith.maxui
            elseif T <: Signed
                arith.maxsi
            else
                throw("cannot compute max on element of type $T")
            end
            max0 = push!(block, mop([operand, get_result(cst0, 1)]; loc=to_loc(op.line)))
            ctx.operands[V(op)] = MLIR.get_result(max0, 1)
        elseif f ∈ math_unary_functions
            fname = nameof(f)
            fop = getproperty(math, fname)
            mop = fop(operand; loc=to_loc(op.line))
            push!(block, mop)
            ctx.operands[V(op)] = MLIR.get_result(mop, 1)
        else
            type = get_type(operand)
            T = julia_type(eltype(type))
            val = zero(T) # use a placeholder arg

            inner_block = Block(MLIRType[], Location[])

            unary_ctx = Context(f)
            _, unary_tape = _trace(f, val; ctx=unary_ctx)

            unary_cg = CodegenContext(mlir_ctx, inner_block, unary_tape, cg.verbose, true)
            lower_tape!(unary_cg)

            ret_operand = unary_ctx.operands[unary_tape.result]
            push!(inner_block, linalg.yield(ret_operand; loc=@loc()))

            empty0 = push!(block, tensor.empty(type; loc=@loc()))
            mapping = push!(block, linalg.map_as_generic(
                inner_block,
                operand, MLIR.get_result(empty0, 1);
                loc=to_loc(op.line),
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

        if !MLIR.is_tensor(type)
            tt = MLIRType(type, shape)
            t = push!(block, tensor.empty(tt; loc=@loc()))
            t_op = push!(block, tensor.insert(operand, MLIR.get_result(t, 1); loc=@loc()))
            operand = MLIR.get_result(t_op, 1)
        end

        if julia_type(eltype(type)) != out_element_type && out_element_type <: AbstractFloat
            sitofp0 = push!(block,
                arith.sitofp(
                    operand,
                    MLIRType(MLIRType(out_element_type), shape);
                    loc=@loc()
                )
            )
            operand = MLIR.get_result(sitofp0, 1)
        end

        if shape != out_shape
            reshape_op = push!(
                block,
                stablehlo.broadcast_in_dim(
                    operand, out_shape;
                    loc=to_loc(op.line)
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
        :multiply
    else
        throw("unsupported broadcast function $(f)")
    end

    out_type = MLIRType(MLIRType(out_element_type), out_shape,)
    fop = getproperty(stablehlo, op_type)

    push!(block, fop(operands; loc=to_loc(op.line)))
end

function lower_call!(cg, ::OpConfig{typeof(fma)}, op)
    (; block, tape) = cg
    ctx = tape.c

    if !cg.allow_scalar_args && all(arg -> !is_tensor(tape, arg), op.args)
        return
    end

    (a, b, c) = get_arg_operands!(cg, op)
    mul0 = push!(block, gen_arith!(cg, *, [a, b]; loc=to_loc(op.line)))
    add0 = push!(block, gen_arith!(cg, +, [MLIR.get_result(mul0, 1), c]; loc=to_loc(op.line)))

    ctx.operands[V(op)] = MLIR.get_result(add0, 1)

    # ctx.operands[V(op)] = MLIR.get_result(
    #     push!(block, math.fma(operands; loc=to_loc(op.line))),
    #     1,
    # )
end

function lower_unary!(cg, op, f)
    (; block, tape) = cg
    ctx = tape.c

    fname = nameof(f)
    fop = getproperty(math, fname)
    operand = get_arg_operand!(cg, first(op.args))
    mop = fop(operand; loc=to_loc(op.line))
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
    (; block, tape) = cg

    if !cg.allow_scalar_args && all(arg -> !is_tensor(tape, arg), op.args)
        return
    end

    f = value(op.fn)

    operands = get_arg_operands!(cg, op)
    push!(block, gen_arith!(cg, f, operands; loc= isnothing(op.line) ? Location() : to_loc(op.line)))
end

function _convertto!(block, a, Ta, T)
    if Ta <: Integer && T <: AbstractFloat
        a = get_result(push!(block, arith.sitofp(a, MLIRType(T); loc=@loc())), 1)
    elseif Ta <: AbstractFloat && T <: AbstractFloat && sizeof(Ta) < sizeof(T)
        a = get_result(push!(block, arith.extf(a, MLIRType(T); loc=@loc())), 1)
    else
        a
    end
end

function gen_arith!(cg, f, operands; loc=Location())
    (; block) = cg

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
            sitofp0 = push!(block, arith.sitofp(operand; loc=@loc()))
            MLIR.get_result(sitofp0, 1)
        end
        is_integer = false
    end

    a, b = operands
    Ta = julia_type(eltype(get_type(a)))
    Tb = julia_type(eltype(get_type(b)))
    T = Base.promote_type(Ta, Tb)

    a = _convertto!(block, a, Ta, T)
    b = _convertto!(block, b, Tb, T)

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
    fop([a, b]; loc)
end

for f in (+, -, /)
    @eval function lower_call!(cg, ::OpConfig{typeof($f)}, op)
        lower_arith!(cg, op)
    end
end
function lower_call!(cg, ::OpConfig{typeof(Base.add_sum)}, op)
    lower_arith!(cg, Umlaut.Call(op.id, op.val, +, op.args; line=op.line)) # TODO: check if this is correct?
end

function lower_call!(cg, ::OpConfig{typeof(*)}, op)
    (; block, tape) = cg

    if all(arg -> !is_tensor(tape, arg), op.args)
        return lower_arith!(cg, op)
    end

    operands = get_arg_operands!(cg, op)

    result_type = MLIRType(op.val)
    op = MLIR.create_operation("stablehlo.dot", to_loc(op.line); operands,
        results=[
            result_type,
        ])

    push!(block, op)
end

function lower_call!(cg, ::OpConfig{typeof(reshape)}, op)
    (; block) = cg

    operand = get_arg_operand!(cg, first(op.args))

    current_size = size(MLIR.get_type(operand))
    out_size = value.(@view op.args[begin+1:end]) # const prod reshape sizes

    all_elements = prod(current_size)
    num_elements = prod(d -> d isa Colon ? 1 : d, out_size)
    out_size = [
        d == Colon() ? all_elements ÷ num_elements : d
        for d in out_size
    ]
    rev_perm = length(out_size):-1:1

    # https://www.tensorflow.org/mlir/hlo_ops#mhloreshape_mlirmhloreshapeop
    # the semantics of the stable.reshape op means that we need to convert to
    # row major before resizing so that the right element order is used.
    # see https://www.tensorflow.org/xla/operation_semantics#reshape
    # for more information

    operand = if length(current_size) > 1
        perm = length(current_size):-1:1
        transpose_op = push!(
            block,
            stablehlo.transpose(
                operand, perm;
                loc=to_loc(op.line)
            ),
        )
        get_result(transpose_op, 1)
    else
        operand
    end
    reshape_op = push!(
        block,
        stablehlo.reshape(
            operand,
            reverse(out_size);
            loc=to_loc(op.line)
        ),
    )
    if length(out_size) > 1
        push!(
            block,
            stablehlo.transpose(
                get_result(reshape_op, 1), rev_perm;
                loc=to_loc(op.line)
            )
        )
    else
        reshape_op
    end
end

countops(block) = begin
    s = 0
    for _ in MLIR.OperationIterator(block)
        s += 1
    end
    s
end

function lower_call!(cg, op)
    f = op.fn isa Variable ?
        cg.tape[op.fn].val : op.fn
    @debug "lowering for" f prim = Umlaut.isprimitive(cg.tape.c, f, op.args...) ops = countops(cg.block)
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

function compile_to_module(tape, args...; verbose, allow_scalar_args)
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))
    typeof_args = args = map(Core.Typeof, args)
    m = which(ctx.f, typeof_args)

    location = Location(string(m.file), m.line, 0)
    mlir_module = MModule(location)
    mlir_module_body = MLIR.get_body(mlir_module)

    block = Block(MLIRType[], Location[])

    # allow_scalar_args = any(arg -> arg.op.typ <: Real, inputs(tape))

    cg_context = CodegenContext(MLIR.context(), block, tape, verbose, allow_scalar_args)
    lower_tape!(cg_context)

    if !haskey(ctx.operands, bound(tape, tape.result))
        isempty(ctx.operands) && throw("failed to compile any operation $allow_scalar_args")
        last_computed = maximum(v -> v.id, keys(ctx.operands))
        last_variable = Umlaut.Variable(last_computed)
        op = tape[last_variable]
        throw("failed to compute a graph for the function @ $(op.line) $(allow_scalar_args)")
    end

    result_operand = ctx.operands[bound(tape, tape.result)]

    result_types = [MLIR.get_type(result_operand)]
    input_types = [
        MLIR.get_type(ctx.operands[bound(tape, arg)])
        for arg in ctx.block_args
    ]
    ftype = MLIRType(input_types => result_types)

    push!(block, func.return_([result_operand]))

    region = Region()
    push!(region, block)
    mlir_func_op = MLIR.create_operation("func.func", location; results=[],
                                        owned_regions=[region],
                                        attributes=[
        MLIR.NamedAttribute("sym_name", MLIR.Attribute(string(func_name))),
        MLIR.NamedAttribute("function_type", MLIR.Attribute(ftype)),
    ])

    push!(mlir_module_body, mlir_func_op)

    verbose && show(IOContext(stdout, :debug => true), mlir_func_op)

    MLIR.verifyall(mlir_module)

    return mlir_module
end

function with_compiler_session(f, sess)
    mlir_ctx = IREE.borrow_context(sess)
    IREE.register_all_dialects!(mlir_ctx)
    MLIR.context!(f, mlir_ctx)
end

function compile_tape(
    tape, args...;
    verbose=haskey(ENV, "COIL_VERBOSE"),
    device,
    hal_target,
    allow_scalar_args
)
    ctx = tape.c
    func_name = ctx.f isa Function ? nameof(ctx.f) : nameof(typeof(ctx.f))

    sess = IREE.CompilerSession()
    graph_call = with_compiler_session(sess) do
        mlir_module = compile_to_module(tape, args...; verbose, allow_scalar_args)

        # Apply row-major -> col-major passes

        @debug "running passes"
        MLIR.run!(Passes.TransposeArgsToRowMajorPass(), mlir_module)
        MLIR.run!(Passes.TransposeReturnTypePass(), mlir_module)
        @debug "done running passes"

        get_call(sess, mlir_module,
                 string("module.", func_name),
                 device, hal_target)
    end

    ###


    result_op = tape[tape.result]
    graph_call_op = Umlaut.mkcall(graph_call, ctx.block_args...; val=result_op.val)
    tape.result = push!(tape, graph_call_op)

    verbose && display(tape)

    verbose && @info "simplifying"
    simplify_tape!(tape)

    verbose && display(tape)

    f_to_device = iree(device, ctx.f)

    func_expr = Umlaut.to_expr(tape)
    abstractize_array_args!(func_expr, typeof(f_to_device))

    compiled_func = Base.eval(@__MODULE__, func_expr)
    (args...) -> compiled_func(f_to_device, (arg isa AbstractArray ? IREE.ColMajorBufferView(device, arg) : arg for arg in args)...)
end

function abstractize_array_args!(ex, f_type)
    @assert Meta.isexpr(ex, :function)
    head = ex.args[1]
    @assert Meta.isexpr(head, :call)

    for (i, arg) in enumerate(Iterators.drop(head.args, 1))
        @assert Meta.isexpr(arg, :(::), 2)
        if i == 1
            arg.args[2] = f_type
            continue
        end
        typ = arg.args[2]

        if typ <: AbstractArray
            arg.args[2] = AbstractArray{typ.parameters...}
        end
    end
end

function lower_tape!(cg)
    (; tape) = cg
    ctx = tape.c

    for op in tape
        if (op isa Umlaut.Input ||
            op isa Umlaut.Call ||
            op isa Umlaut.Constant) && is_trackable_tensor(op.val)

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
        push!(tape, Umlaut.Input(f))
        vargs = map(arg -> push!(tape, Umlaut.Input(arg)), args)
        tape.result = Umlaut.record_primitive!(tape, f, vargs...)
        return out, tape
    end
    Umlaut.trace(f, args...; ctx)
end

"""
  compile(
    f;
    verbose::Bool=false,
    discard_first_outputs::Bool=false,
    allow_scalar_args::Bool=false,
  )::Function

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
function compile(f; verbose=false, discard_first_outputs=false, device=IREE.Device("local-task"), hal_target="llvm-cpu", allow_scalar_args=false)
    compiled_f = nothing
    return (args...) -> begin
        if isnothing(compiled_f)
            # session = get_session()
            # f = iree(session, f)

            out, tape = _trace(f, args...)
            verbose && display(tape)

            compiled_f = compile_tape(tape, args...; verbose, device, hal_target, allow_scalar_args)

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
            sess = IREE.CompilerSession()
            with_compiler_session(sess) do
                compile_to_module(tape, args...;
                                  verbose=false,
                                  allow_scalar_args=true)
            end, sess
        end
    end
end

"""
    @code_linalg f(args...)

Similar to `@code_mlir` but with with the dialect translation option
enabled "iree-stablehlo-to-linalg-on-tensors".
"""
macro code_linalg(call)
    Meta.isexpr(call, :call) || throw("expected call Expr")
    quote
        let
            f = $(esc(first(call.args)))
            args = $(esc(Expr(:vect, call.args[begin+1:end]...)))
            ctx = Context(f)
            _, tape = _trace(f, args...; ctx)
            sess = IREE.CompilerSession()
            module_ = with_compiler_session(
                () -> compile_to_module(tape, args...; verbose=false, allow_scalar_args=true),
                sess,
            )
            lower_to_linalg!(sess, module_)
        end
    end
end

using .IREE: Compiler

function run_pipeline!(sess, module_, pipeline)
    invocation = Compiler.ireeCompilerInvocationCreate(sess)

    try
        success = Compiler.ireeCompilerInvocationImportBorrowModule(
            invocation,
            MLIR.get_operation(module_),
        )
        success || error("failed to import module") 

        success = Compiler.ireeCompilerInvocationRunPassPipeline(invocation, pipeline)
        success || error("failed to run pass pipeline \"$pipeline\"")

    finally
        Compiler.ireeCompilerInvocationDestroy(invocation)
    end

    module_
end

function lower_to_linalg!(sess, module_)
    run_pipeline!(sess, module_, "iree-stablehlo-to-linalg")
end

with_debug(val) = with_debug(stdout, val)
with_debug(io::IO, val) = show(IOContext(io, :debug => true), val)

### Compile utils

function compile_to_bytecode(sess, module_; input_type=MLIR.get_input_type(module_), hal_target="llvm-cpu")
    input_type ∈ (
        :none,
        :stablehlo,
        :xla,
        :tm_tensor,
        :tosa,
    ) || throw("invalid iree input type ($input_type)")

    # pm = MLIR.PassManager(module_.context)
    # op_pass = MLIR.OpPassManager(pm)
    options = [
        "--iree-hal-target-backends=$hal_target",
        "--iree-input-type=$input_type",
    ]

    # IREE.build_vm_pass_pipeline!(op_pass, options)
    # MLIR.run!(pm, module_)

    return IREE.translate_module_to_vm_bytecode(sess, module_, options)
end

function get_session(driver="local-task")
    instance = IREE.Instance()
    device = IREE.Device(instance, string(driver))

    session_options = IREE.SessionOptions()
    IREE.Session(instance, session_options, device)
end

function get_call(compiler_session, module_, name, device, hal_target)
    mod_op = MLIR.get_operation(module_)
    block = MLIR.get_first_block(mod_op)

    realname = chopprefix(name, "module.")

    args_transposed = false
    return_transposed = false
    for op in MLIR.OperationIterator(block)
        if MLIR.name(op) == "func.func"
            funcname = MLIR.get_attribute_by_name(op, "sym_name")
            if isnothing(funcname) || MLIR.string_value(funcname) != realname
                continue
            end

            args_transposed_attr = MLIR.get_attribute_by_name(op, Passes.COIL_ARGS_TRANSPOSED_ATTR_NAME)
            args_transposed = !isnothing(args_transposed_attr) && MLIR.bool_value(args_transposed_attr)

            return_transposed_attr = MLIR.get_attribute_by_name(op, Passes.COIL_RETURN_TRANSPOSED_ATTR_NAME)
            return_transposed = !isnothing(return_transposed_attr) && MLIR.bool_value(return_transposed_attr)
            break
        end
    end

    bytecode = compile_to_bytecode(compiler_session, module_; hal_target)
    session = IREE.Session(IREE.Instance(), IREE.SessionOptions(), device)

    IREE.append_bytecode!(session, bytecode)

    return IREE.Call(session, name, args_transposed, return_transposed)
end

"""
    simplify_tape!(tape::Tape)

Removes any op that is not needed to compute the tape result.
"""
function simplify_tape!(tape)
    res = tape.result
    resop = tape[bound(tape, res)]

    if resop isa Umlaut.Input || resop isa Umlaut.Constant
        filter!(!=(resop), tape.ops)
        return tape
    end

    universe = Set{Umlaut.Variable}([V(resop)])
    found_res = false

    for op in Iterators.reverse(tape.ops)
        if op isa Umlaut.Input || op isa Umlaut.Constant
            continue
        end

        if op == resop
            found_res = true
            union!(universe, filter(a -> a isa V, op.args))
            continue
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

struct BufferViewDeviceAdaptor
    device::IREE.Device
    col_major::Bool
end

function Adapt.adapt_storage(adaptor::BufferViewDeviceAdaptor, a::AbstractArray)
    adaptor.col_major ?
    IREE.ColMajorBufferView(adaptor.device, a) :
    BufferView(adaptor.device, a)
end

iree(device, x) = Functors.fmap(
    x -> Adapt.adapt(BufferViewDeviceAdaptor(device, true), x), x
)

vulkan(x) = iree(IREE.Device("vulkan"), x)

end # module Tracing
