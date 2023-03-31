"""
Transducer implements a Julia typed SSAIR to MLIR pipeline. It is
inspired by [Umlaut.jl](https://github.com/dfdx/Umlaut.jl) and
[Mjolnir.jl](https://github.com/FluxML/Mjolnir.jl).
"""
module Transducer

using Core.Compiler: IRCode
using Core: Const, Argument, SSAValue, PiNode, GotoIfNot, GotoNode, ReturnNode
import CompilerPluginTools

import ..Tracing
using ..MLIR, ..IREE
using ..MLIR: add_operands!
using ..func, ..cf

function context()
    mlir_ctx = MLIR.Context()
    IREE.register_all_dialects!(mlir_ctx)
    for dialect in ("builtin", "func", "mhlo", "arith", "math", "tosa", "linalg", "tensor", "cf")
        MLIR.get_or_load_dialect(mlir_ctx, dialect)
    end
    mlir_ctx
end

struct Tensor{T,Dims,N} <: AbstractArray{T,N}
    view::AbstractArray{T,N}
end

Base.eltype(::Type{Tensor{T}}) where {T} = T
Base.size(::Type{Tensor{T,Dims}}) where {T,Dims} = Dims

struct Scalar{T<:Number} <: Number
    val::T
end

Base.convert(::Type{T}, s::Scalar{T}) where {T} = copy(s.val)

mutable struct Operation{Dialect,Name,Results}
    attributes::Dict{Symbol,Any}

    function Operation{Dialect,Name,Results}(; kwargs...) where {Dialect,Name,Results}
        new{Dialect,Name,Results}(Dict{Symbol,Any}(kwargs))
    end
end

function gen_mlir(context, op::Operation{Dialect,Name,Results}, operands) where {Dialect,Name,Results}
    state = OperationState(string(Dialect, ".", Name), Location(context))

    add_attributes!(state, [])
    add_operands!(state, collect(operands))
    @assert !(Results <: Tuple)
    add_results!(state, [
        mlir_type(context, Results)
    ])

    MLIR.Operation(state)
end

mlir_type(context, ::Type{Tensor{T,Dims,N}}) where {T,Dims,N} = MType(context, mlir_type(context, T), Dims)
mlir_type(context, T::Type{<:Real}) = MType(context, T)
mlir_type(context, ::Type{Scalar{T}}) where {T} = MType(context, T)

"""
  emit_and_run(op::Operation, args...)

Generates a just-in-time MLIR function which calls the corresponding MLIR operation.
This provides a "fast" path for eager operation evaluation.
"""
function emit_and_run(op, args...)
    mlir_ctx = context()

    input_types = [mlir_type(mlir_ctx, typeof(arg)) for arg in args]
    block = Block(
        input_types,
        [Location(mlir_ctx) for _ in args],
    )
    operands = MLIR.Value[
        MLIR.get_argument(block, i)
        for i in 1:length(args)
    ]
    mlir_op = push!(block, gen_mlir(mlir_ctx, op, operands))
    ret_op = push!(block, func.return_(mlir_ctx, [MLIR.get_result(mlir_op)]))

    fn_state = OperationState("func.func", Location(mlir_ctx))
    func_name = MLIR.get_name(mlir_op)

    result_types = [MLIR.get_type(MLIR.get_operand(ret_op, 1))]
    ftype = MType(mlir_ctx, input_types => result_types)

    add_attributes!(fn_state, [
        NamedAttribute(mlir_ctx, "sym_name", MLIR.Attribute(mlir_ctx, func_name)),
        NamedAttribute(mlir_ctx, "function_type", MLIR.Attribute(ftype)),
    ])
    region = Region()
    push!(region, block)
    MLIR.add_owned_regions!(fn_state, [region])

    fn = MLIR.Operation(fn_state)

    mlir_module = MLIR.MModule(mlir_ctx)
    mlir_module_body = MLIR.get_body(mlir_module)
    push!(mlir_module_body, fn)

    MLIR.verifyall(mlir_module)

    # TODO: figure out how to get the hal_target from device
    hal_target = "llvm-cpu"
    device = IREE.Device()

    fn_call = Tracing.get_call(
        mlir_module, "module.$func_name",
        device, hal_target,
    )

    fn_call(args...)
end

function (op::Operation{D,N,R})(args...) where {D,N,R}
    emit_and_run(op, args...)::R # this type assertion helps type-inference in the IR
end

macro op(call)
    @assert Meta.isexpr(call, :(::), 2)
    call, ret = call.args
    name, args... = call.args

    @assert Meta.isexpr(name, :(.))
    dialect, name = name.args
    name = name.value

    :(Operation{$(QuoteNode(dialect)),$(QuoteNode(name)),$(esc(ret))}()($([esc(arg) for arg in args]...)))
end

function Base.reshape(a::Tensor{T,Dims,N}, new_dims::Union{Colon,Int64}...) where {T,Dims,N}
    Nn = length(new_dims)
    new_dims = Tuple(new_dims)
    @op mhlo.reshape(a)::Tensor{T,new_dims,Nn}
end

function Base.:*(a::Tensor{F,DimsA,2}, b::Tensor{F,DimsB,2}) where {F,DimsA,DimsB}
    ia, ka = DimsA
    kb, jb = DimsB
    @assert ka == kb
    Dims = (ia, jb)

    @op mhlo.dot(a, b)::Tensor{F,Dims,2}
end

function Base.:+(a::Tensor{Float32,Dims,N}, b::Tensor{Float32,Dims,N}) where {Dims,N}
    @op arith.addf(a, b)::Tensor{Float32,Dims,N}
end
function Base.:+(a::Tensor{I,Dims,N}, b::Tensor{I,Dims,N}) where {I<:Integer,Dims,N}
    @op arith.addi(a, b)::Tensor{I,Dims,N}
end

function Base.:-(a::Tensor{Float32,Dims,N}, b::Tensor{Float32,Dims,N}) where {Dims,N}
    @op arith.subf(a, b)::Tensor{Float32,Dims,N}
end
function Base.:-(a::Tensor{I,Dims,N}, b::Tensor{I,Dims,N}) where {I<:Integer,Dims,N}
    @op arith.subi(a, b)::Tensor{I,Dims,N}
end

## --- Scalars

Base.:+(a::Scalar{T}, b::Scalar{T}) where {T<:AbstractFloat} = @op arith.addf(a, b)::Scalar{T}
Base.:-(a::Scalar{T}, b::Scalar{T}) where {T<:AbstractFloat} = @op arith.subf(a, b)::Scalar{T}
Base.:*(a::Scalar{T}, b::Scalar{T}) where {T<:AbstractFloat} = @op arith.mulf(a, b)::Scalar{T}

Base.:+(a::Scalar{T}, b::Scalar{T}) where {T<:Integer} = @op arith.addi(a, b)::Scalar{T}
Base.:-(a::Scalar{T}, b::Scalar{T}) where {T<:Integer} = @op arith.subi(a, b)::Scalar{T}
Base.:*(a::Scalar{T}, b::Scalar{T}) where {T<:Integer} = @op arith.muli(a, b)::Scalar{T}

for f in (:cos, :sin, :tan, :tanh, :exp, :sqrt)
    @eval function Base.$(f)(x::Scalar{T}) where {T<:AbstractFloat}
        Operation{:math,$(QuoteNode(f)),Scalar{T}}()(x)
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui, :f),
)
    T = t == :si ? Signed : t == :ui ? Unsigned : AbstractFloat
    @eval function Base.$(f)(x::Scalar{T}, y::Scalar{T}) where {T<:$T}
        Operation{:arith,$(QuoteNode(Symbol(f, t))),Scalar{T}}()(x, y)
    end
end

## ---

mutable struct Tracer
    mlir_ctx::MLIR.Context
    region::MLIR.Region 
    current_block::MLIR.Block
    ret_block::Union{Nothing,MLIR.Block} # nothing for toplevel
    code::IRCode
    preamble::Vector{Any}
    argvals::Vector{Any}
    ssavals::Vector{Any}
    fn
    targs
end

function Tracer(f, targs)
    specs = CompilerPluginTools.code_ircode_by_signature(Tuple{typeof(f), targs.parameters...})
    ircode, ret = only(specs)
    Tracer(ircode, f, targs)
end

function build_mlir_block(mlir_ctx, targs)
    mlir_targs = [
        (i + 1, targ)
        for (i, targ) in enumerate(targs.parameters)
        if targ <: Union{Tensor,Scalar}
    ]
    block = MLIR.Block(
        [mlir_type(mlir_ctx, Targ) for (_, Targ) in mlir_targs],
        [Location(mlir_ctx) for _ in mlir_targs],
    )
    argvals = Vector{Any}(undef, 1 + length(targs.parameters))

    for (i, (argn, _)) in enumerate(mlir_targs)
        argvals[argn] = MLIR.get_argument(block, i)
    end

    block, argvals 
end

function Tracer(ircode::IRCode, f, targs)
    mlir_ctx = context()

    ssavals = Vector{Any}(undef, length(ircode.stmts))
    entry, argvals = build_mlir_block(mlir_ctx, targs)
    region = Region()
    push!(region, entry)

    Tracer(mlir_ctx, region, entry, nothing, ircode, [], argvals, ssavals, f, targs)
end

function mlir_val(tracer, val)::MLIR.Value
    if val isa Argument && isassigned(tracer.argvals, val.n)
        return tracer.argvals[val.n]
    elseif val isa SSAValue && isassigned(tracer.ssavals, val.id)
        return tracer.ssavals[val.id]
    else
        val isa SSAValue && @info tracer.code.stmts.inst[val.id]
        error("value $val is not a part of the MLIR lattice")
    end
end

function const_val(tracer, val)
    if val isa QuoteNode
        return val.value
    elseif val isa GlobalRef
        return getproperty(val.mod, val.name)
    elseif val isa SSAValue && isassigned(tracer.ssavals, val.id)
        return tracer.ssavals[val.id]
    elseif val isa Const
        return val.val
    else
        return val
    end
end

isconst(tracer, val::SSAValue) = isassigned(tracer.ssavals, val.id) && !(tracer.ssavals[val.id] isa MLIR.Value) # FIXME
isconst(tracer, ::Argument) = false
isconst(tracer, _) = true

function const_typ(tracer, val)
    if val isa SSAValue
        typ = tracer.code.stmts.type[val.id]
        return typ
    elseif val isa Argument
        return tracer.code.argtypes[val.n] 
    else
        return typeof(const_val(tracer, val))
    end
end

function invoke!(tracer, inst, rettype)
    @assert Meta.isexpr(inst, :invoke)

    f = const_val(tracer, inst.args[2])

    # is primitive v

    args = @view inst.args[begin+2:end]
    if all(arg -> isconst(tracer, arg), args) 
        argvals = map(arg -> const_val(tracer, arg), args)
        return f(argvals...)
    end
    
    if f == emit_and_run
        op = gen_mlir(tracer.mlir_ctx, const_val(tracer, args[1]), [
            mlir_val(tracer, arg)
            for arg in args[begin+1:end]
        ])
        push!(tracer.current_block, op)
        return MLIR.get_result(op, 1)
    end

    # is primitive ^

    targs = Tuple{(const_typ(tracer, arg) for arg in args)...}
    sig = Tuple{const_typ(tracer, f), targs.parameters...}

    codes = CompilerPluginTools.code_ircode_by_signature(sig)
    ircode, _ = only(codes)
    (; mlir_ctx,) = tracer

    entry, argvals = build_mlir_block(mlir_ctx, targs)
    ssavals = Vector{Any}(undef, length(ircode.stmts))

    push!(tracer.region, entry)
    push!(tracer.current_block, cf.br(mlir_ctx, entry, [
        argvals[i]
        for i in eachindex(argvals)
        if isassigned(argvals, i)
    ]))

    ret_block = Block([mlir_type(mlir_ctx, rettype)], [Location(mlir_ctx)])
    MLIR.insert_after!(tracer.region, entry, ret_block)

    new_tracer = Tracer(
        mlir_ctx, tracer.region,
        entry, ret_block,
        ircode, [],
        argvals, ssavals,
        f, targs,
    )
    trace!(new_tracer)

    tracer.current_block = ret_block
    MLIR.get_argument(ret_block, 1)
end

function trace!(tracer::Tracer)
    (; code, mlir_ctx) = tracer

    for i in 1:length(code.stmts) 
        stmt = code.stmts.inst[i]

        if Meta.isexpr(stmt, :invoke)
            rettype = const_typ(tracer, SSAValue(i))
            tracer.ssavals[i] = invoke!(tracer, stmt, rettype)
        elseif Meta.isexpr(stmt, :call)
            args = stmt.args[begin+1:end]
            if all(arg -> isconst(tracer, arg), args)
                f = const_val(stmt.args[1])
                tracer.ssavals[i] = f(args...)
            else
                @warn "skipping call" f = stmt.args[1]
            end
        elseif Meta.isexpr(stmt, :new)
            tracer.ssavals[i] = __new__((const_val(tracer, arg) for arg in stmt.args)...)
        elseif stmt isa ReturnNode
            if isnothing(tracer.ret_block)
                op = func.return_(mlir_ctx, [mlir_val(tracer, stmt.val)])
                push!(tracer.current_block, op)
                return
            end

            op = cf.br(mlir_ctx, tracer.ret_block, [mlir_val(tracer, stmt.val)])
            push!(tracer.current_block, op)
            return
        elseif stmt isa PiNode
            tracer.ssavals[i] = const_val(tracer, stmt.val)
        else
            @warn "skipping stmt" stmt
        end

        # Do dispatch
    end
end

function gen_fn(tracer::Tracer)
    (; mlir_ctx,) = tracer

    fn_state = OperationState("func.func", Location(mlir_ctx))
    func_name = tracer.fn isa Function ?
        nameof(tracer.fn) :
        nameof(typeof(tracer.fn))

    local ret_op
    for op in MLIR.OperationIterator(tracer.current_block)
        if MLIR.get_name(op) == "func.return"
            ret_op = op
            break
        end
    end

    entry = MLIR.get_first_block(tracer.region)

    result_types = [MLIR.get_type(MLIR.get_operand(ret_op, 1))]
    input_types = [
        MLIR.get_type(MLIR.get_argument(entry, i))
        for i in 1:MLIR.num_arguments(entry)
    ]
    ftype = MType(mlir_ctx, input_types => result_types)

    add_attributes!(fn_state, [
        NamedAttribute(mlir_ctx, "sym_name", MLIR.Attribute(mlir_ctx, string(func_name))),
        NamedAttribute(mlir_ctx, "function_type", MLIR.Attribute(ftype)),
    ])

    MLIR.add_owned_regions!(fn_state, [tracer.region])

    fn = MLIR.Operation(fn_state)

    MLIR.verifyall(fn)

    fn
end

function compile2(f, Targs)
    tracer = Tracer(f, Targs)
    trace!(tracer)
    gen_fn(tracer)
end

macro __splatnew__(T, args)
    esc(Expr(:splatnew, T, args))
end

"""
    __new__(T, args...)

User-level version of the `new()` pseudofunction.
Can be used to construct most Julia types, including structs
without default constructors, closures, etc.

> Taken from Umlaut.jl
"""
@inline function __new__(T, args...)
    @__splatnew__(T, args)
end

end # module Transducer
