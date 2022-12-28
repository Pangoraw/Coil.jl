module Coil

include("MLIR/MLIR.jl")
include("IREE/IREE.jl")

using .MLIR

module func

using ..MLIR

function return_(context, operands; loc=Location(context))
    state = OperationState("func.return", loc)
    MLIR.add_operands!(state, operands)
    Operation(state)
end

end

module arith

using ..MLIR

function addf(context, operands, type; loc=Location(context))
    state = OperationState("arith.addf", loc)
    MLIR.add_operands!(state, operands)
    MLIR.add_results!(state, [type])
    Operation(state)
end

function addi(context, operands, type; loc=Location(context))
    state = OperationState("arith.addi", loc)
    MLIR.add_operands!(state, operands)
    MLIR.add_results!(state, [type])
    Operation(state)
end

function constant(context, value::T; loc=Location(context)) where {T}
    state = OperationState("arith.constant", loc)
    MLIR.add_results!(state, [MType(context, T)])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value",
            Attribute(context, value)),
    ])
    Operation(state)
end

end # module arith

module mhlo

using ..MLIR

function return_(context, operand; loc=Location(context))
    state = OperationState("mhlo.return", loc)
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

function map(context, block, operand, dims=collect(1:ndims(operand)); loc=Location(context))
    state = OperationState("mhlo.map", loc)

    region = Region()
    push!(region, block)

    MLIR.add_owned_regions!(state, [region])
    MLIR.add_operands!(state, [operand])
    MLIR.add_results!(state, [MLIR.get_type(operand)])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(
            context, "dimensions",
            MLIR.Attribute(context, collect(dims .- 1)),
        ),
    ])

    Operation(state)
end

for f in (:minimum, :maximum)
    @eval function $f(context, operands; loc=Location(context))
        state = OperationState($(string("mhlo.", f)), loc)
        MLIR.add_results!(state, [MLIR.get_type(first(operands))])
        MLIR.add_operands!(state, operands)
        Operation(state)
    end
end

function convolution(context, output_type, operands; loc=Location(context))
    state = OperationState("mhlo.convolution", loc)
    MLIR.add_attributes!(state, [
        NamedAttribute(context, "batch_group_count", Attribute(context, 1)),
        NamedAttribute(context, "dimension_numbers", parse(Attribute, context, """
            #mhlo.conv<raw
              input_batch_dimension = 3,
              input_feature_dimension = 2,
              input_spatial_dimensions = [0, 1],
              kernel_input_feature_dimension = 2,
              kernel_output_feature_dimension = 2,
              kernel_spatial_dimensions = [0, 1],
              output_batch_dimension = 3,
              output_feature_dimension = 2,
              output_spatial_dimensions = [0, 1]
            >
        """)),
        NamedAttribute(context, "feature_group_count", Attribute(context, 1)),
        NamedAttribute(context, "padding", Attribute(context, zeros(Int64, 2, 2))),
        NamedAttribute(context, "rhs_dilation", Attribute(context, [1])),
        NamedAttribute(context, "window_strides", Attribute(context, [1])),
    ])
    #=
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
    =#
    MLIR.add_results!(state, [results])
    MLIR.add_operands!(state, operands)
    Operation(state)
end

"""
    _compute_broadcast_dims(::NTuple{N1,Int64}, ::NTuple{N2,Int64})::Vector{Int64}

Returns the dimensions which are broadcasted upon for a given input and broadcast size size.

```julia
julia> Coil.mhlo._compute_broadcast_dims((10,), (10,1))
Int64[1]

julia> Coil.mhlo._compute_broadcast_dims((1,1,3,1), (12,12,3,1))
Int64[1,2,3,4]
```
"""
function _compute_broadcast_dims(input_size, broadcast_size)
    i = 1
    j = 1
    broadcast_dimensions = Int64[]
    while i <= length(input_size) && j <= length(broadcast_size)
        if input_size[i] == broadcast_size[j]
            push!(broadcast_dimensions, i)
            i += 1
            j += 1
        elseif broadcast_size[j] == 1
            j += 1
        else
            push!(broadcast_dimensions, i)
            i += 1
        end
    end
    broadcast_dimensions
end

function broadcast_in_dim(context, operand, new_size; loc=Location(context))
    state = OperationState("mhlo.broadcast_in_dim", loc)
    MLIR.add_operands!(state, [operand])
    type = MLIR.get_type(operand)

    type_size = size(type)
    broadcast_dimensions = _compute_broadcast_dims(type_size, new_size)

    MLIR.add_results!(state, [
        MType(context, eltype(type), new_size),
    ])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(
            context, "broadcast_dimensions",
            Attribute(context, broadcast_dimensions .- 1),
        )
    ])
    Operation(state)
end

function add(context, operands; loc=Location(context))
    state = OperationState("mhlo.add", loc)
    MLIR.add_results!(state, [MLIR.get_type(first(operands))])
    MLIR.add_operands!(state, operands)
    Operation(state)
end

function compare(context, operands, direction; loc=Location(context))
    state = OperationState("mhlo.compare", loc)
    MLIR.add_operands!(state, operands)
    MLIR.add_results!(state, [MType(context, MType(context, Bool), ())])
    MLIR.add_attributes!(state, [
        NamedAttribute(
            context, "comparison_direction",
            parse(Attribute, context, "#mhlo<comparison_direction $direction>"),
        ),
    ])
    Operation(state)
end

function if_(context, cond, result, block1, block2; loc=Location(context))
    state = OperationState("mhlo.if", loc)
    MLIR.add_operands!(state, [cond])
    MLIR.add_results!(state, [result])

    region1 = Region()
    push!(region1, block1)
    region2 = Region()
    push!(region2, block2)
    MLIR.add_owned_regions!(state, [region1, region2])

    Operation(state)
end

function constant(context, value; loc=Location(context))
    etype = MType(context, eltype(value))
    ranked_type = MType(context, etype, size(value))
    state = OperationState("mhlo.constant", loc)
    MLIR.add_results!(state, [ranked_type,])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value", Attribute(context, fill(value)))
    ])
    Operation(state)
end

function reduce(context, operands, input_size, dim; loc=Location(context))
    input_type = MLIR.get_type(first(operands))

    etype = eltype(input_type)
    output_dim = [
        s
        for (i, s) in enumerate(input_size)
        if i != dim
    ]
    output_type = MType(context, etype, output_dim)
    f32_unary = MType(context, MType(context, Float32), ())

    region = Region()
    inner_block = Block([f32_unary, f32_unary], [loc, loc])
    push!(region, inner_block)

    add = push!(
        inner_block,
        mhlo.add(context, [
            MLIR.get_argument(inner_block, 1),
            MLIR.get_argument(inner_block, 2),
        ])
    )
    push!(inner_block,
        mhlo.return_(context, MLIR.get_result(add, 1)))

    state = OperationState("mhlo.reduce", loc)
    MLIR.add_owned_regions!(state, [region])

    MLIR.add_results!(state, [output_type])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "dimensions",
            MLIR.Attribute(context, [dim - 1])),
    ])
    MLIR.add_operands!(state, operands)

    Operation(state)
end

# relu is not a valid mhlo op but we keep codegen for it here
# relu(x::T) where {T} = max(zero(T), x)
function relu!(block, context, operand; loc=Location(context))
    type = MLIR.get_type(operand)
    T = MLIR.julia_type(eltype(type))

    cst0 = push!(block, mhlo.constant(context, zero(T)))
    push!(block,
        mhlo.maximum(context, [
            operand,
            MLIR.get_result(cst0, 1),
        ])
    )
end

end # module mhlo

function gen_func(context, name, argtypes, rettypes; loc=Location(context))
    MLIR.get_or_load_dialect(context, "func")
    MLIR.get_or_load_dialect(context, "arith")

    region = Region()

    block = Block(collect(argtypes), [Location(context) for _ in argtypes])
    push!(region, block)

    state = OperationState("func.func", loc)
    MLIR.add_owned_regions!(state, [region])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "sym_name", Attribute(context, name)),
        MLIR.NamedAttribute(context, "function_type", Attribute(MType(context, argtypes => rettypes))),
    ])

    ### Implementation

    add_op = all(MLIR.is_integer, argtypes) ? arith.addi : arith.addf
    add = push!(block,
        add_op(context, [MLIR.get_argument(block, 1), MLIR.get_argument(block, 2)], rettypes[1]))
    add0 = MLIR.get_result(add, 1)
    push!(block,
        func.return_(context, [add0]))

    ###

    Operation(state)
end

function gen_const_func(context, name; loc=Location(context))
    MLIR.get_or_load_dialect(context, "func")

    region = Region()
    block = Block(MType[], Location[])
    push!(region, block)

    i32 = MLIR.MType(context, Int32)

    state = OperationState("func.func", loc)
    MLIR.add_owned_regions!(state, [region])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "sym_name", Attribute(context, name)),
        MLIR.NamedAttribute(context, "function_type", Attribute(MType(context, () => (i32,)))),
    ])

    ## Implementation

    constant = arith.constant(context, Int32(42))
    cst0 = MLIR.get_result(constant, 1)

    push!(block, constant)
    push!(block,
        func.return_(context, [cst0]))

    ##

    Operation(state)
end

function gen_mhlo_func(context, name; loc=Location(context))
    MLIR.get_or_load_dialect(context, "func")
    MLIR.get_or_load_dialect(context, "mhlo")

    input_type = MType(context, Array{Float32,1}, (10,))
    output_type = MType(context, MType(context, Float32), ())

    region = Region()
    block = push!(region, Block(MType[input_type], Location[Location(context)]))

    functype = MType(context, (input_type,) => (output_type,))

    state = OperationState("func.func", loc)
    MLIR.add_owned_regions!(state, [region])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "sym_name", Attribute(context, name)),
        MLIR.NamedAttribute(context, "function_type", Attribute(functype)),
    ])

    ###

    cst0 = mhlo.constant(context, zero(Float32))
    push!(block, cst0)

    arg0 = MLIR.get_argument(block, 1)
    vcst0 = MLIR.get_result(cst0, 1)
    reduce0 = mhlo.reduce(context, [arg0, vcst0], (10,), 1)
    push!(block, reduce0)

    ret = func.return_(context, [MLIR.get_result(reduce0, 1)])
    push!(block, ret)

    ###

    Operation(state)
end

include("tracing.jl")

import .Tracing: compile

end # module Coil
