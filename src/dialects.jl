module func
# https://mlir.llvm.org/docs/Dialects/Func/

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

function return_(context, operands; loc=Location(context))
    state = OperationState("func.return", loc)
    MLIR.add_operands!(state, operands)
    Operation(state)
end

end # module func

module math
# https://mlir.llvm.org/docs/Dialects/MathOps/

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

for op in (:cos, :sin, :tan, :tanh, :exp, :sqrt)
    @eval function $op(context, operand; loc=Location(context))
        state = OperationState($(string("math.", op)), loc)
        MLIR.add_operands!(state, [operand])
        MLIR.add_results!(state, [MLIR.get_type(operand)])
        Operation(state)
    end
end

end # module math

module arith
# https://mlir.llvm.org/docs/Dialects/ArithOps/

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

for (f, t) in Iterators.product(
    (:add, :sub, :mul),
    (:i, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=MLIR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("arith.", fname)), loc)
        MLIR.add_operands!(state, operands)
        MLIR.add_results!(state, [type])
        Operation(state)
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=MLIR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("arith.", fname)), loc)
        MLIR.add_operands!(state, operands)
        MLIR.add_results!(state, [type])
        Operation(state)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-mlirarithindexcastop
for f in (:index_cast, :index_castui)
    @eval function $f(context, operand; loc=Location(context))
        state = OperationState($(string("arith.", f)), loc)
        add_operands!(state, [operand])
        add_results!(state, [MLIR.IndexType(context)])
        Operation(state)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-mlirarithextfop
function extf(context, operand, type; loc=Location(context))
    state = OperationState("arith.exf", loc)
    MLIR.add_results!(state, [type])
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-mlirarithsitofpop
function sitofp(context, operand, ftype=float(julia_type(eltype(get_type(operand)))); loc=Location(context))
    state = OperationState("arith.sitofp", loc)
    type = get_type(operand)
    MLIR.add_results!(state, [
        MLIR.is_tensor(type) ?
        MType(context, ftype isa MType ? eltype(ftype) : MType(context, ftype), size(type)) :
        MType(context, ftype)
    ])
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-mlirarithfptosiop
function fptosi(context, operand, itype; loc=Location(context))
    state = OperationState("arith.fptosi", loc)
    type = get_type(operand)
    MLIR.add_results!(state, [
        MLIR.is_tensor(type) ?
        MType(context, itype isa MType ? itype : MType(context, itype), size(type)) :
        MType(context, itype)
    ])
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(
    context, values::AbstractArray,
    type=MType(context, MType(context, eltype(values)), size(values));
    loc=Location(context))

    state = OperationState("arith.constant", loc)
    MLIR.add_results!(state, [type])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value",
            Attribute(context, values, type)),
    ])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(context, value, type=MType(context, typeof(value)); loc=Location(context))
    state = OperationState("arith.constant", loc)
    MLIR.add_results!(state, [type])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value",
            Attribute(context, value)),
    ])
    Operation(state)
end

end # module arith

module tosa
# https://mlir.llvm.org/docs/Dialects/TOSA/

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosamaximum-mlirtosamaximumop
# https://mlir.llvm.org/docs/Dialects/TOSA/#tosaminimum-mlirtosaminimumop
for f in (:maximum, :minimum)
    @eval function $f(context, operands; loc=Location(context))
        state = OperationState($(string("tosa.", f)), loc)
        MLIR.add_operands!(state, operands)
        MLIR.add_results!(state, [MLIR.get_type(first(operands))])
        Operation(state)
    end
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosaconst-mlirtosaconstop
function const_(context, value, type=MType(context, typeof(value), size(value)); loc=Location(context))
    state = OperationState("tosa.const", loc)
    MLIR.add_results!(state, [type])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value",
            Attribute(context, value, type)),
    ])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosareshape-mlirtosareshapeop
function reshape(context, operand, shape; loc=Location(context))
    state = OperationState("tosa.reshape", loc)
    out_type = MType(context, eltype(MLIR.get_type(operand)), shape)
    add_results!(state, [out_type])
    add_operands!(state, [operand])
    add_attributes!(state, [
        NamedAttribute(context, "new_shape", MLIR.ArrayAttribute(context, collect(shape)))
    ])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosareduce_sum-mlirtosareducesumop
function reduce_sum(context, input, axis; loc=Location(context))
    state = OperationState("tosa.reduce_sum", loc)

    type = get_type(input)
    input_shape = size(type)
    output_shape = [i == axis ? 1 : d for (i, d) in enumerate(input_shape)]

    add_results!(state, [MType(context, eltype(type), output_shape)])
    add_operands!(state, [input])
    add_attributes!(state, [
        NamedAttribute(context, "axis", Attribute(context, axis - 1))
    ])

    Operation(state)
end

end # module tosa

module linalg

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmap-mlirlinalgmapop
function map(context, inner_block, in, out; loc=Location(context))
    state = OperationState("linalg.map", loc)
    region = Region()
    push!(region, inner_block)
    MLIR.add_owned_regions!(state, [region])
    MLIR.add_results!(state, [MLIR.get_type(out)])
    MLIR.add_operands!(state, [in, out])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgreduce-mlirlinalgreduceop
function reduce(
    context,
    inner_block,
    in, out,
    dims=1:ndims(get_type(in));
    loc=Location(context)
)
    state = OperationState("linalg.reduce", loc)
    region = Region()
    push!(region, inner_block)
    MLIR.add_owned_regions!(state, [region])
    add_results!(state, [get_type(out)])
    add_operands!(state, [in, out])
    add_attributes!(state, [
        NamedAttribute(context, "dimensions",
           MLIR.ArrayAttribute(context, [d - 1 for d in dims]))
    ])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgyield-mlirlinalgyieldop
function yield(context, operand; loc=Location(context))
    state = OperationState("linalg.yield", loc)
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgfill-mlirlinalgfillop
function fill(context, operands...; loc=Location(context))
    state = OperationState("linalg.fill", loc)
    MLIR.add_owned_regions!(state, [Region()])
    add_operands!(state, collect(operands))
    add_results!(state, [get_type(last(operands))])
    add_attributes!(state, [
        NamedAttribute(context, "operand_segment_sizes",
            Attribute(context, collect(Int32.(length.(size.(get_type.(operands))))))),
    ])
    Operation(state)
end

end # module linalg

module tensor

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-mlirtensoremptyop
function empty(context, type; loc=Location(context))
    state = OperationState("tensor.empty", loc)
    MLIR.add_results!(state, [type])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract-mlirtensorextractop
function extract(context, operand, I...; loc=Location(context))
    state = OperationState("tensor.extract", loc)
    MLIR.add_results!(state, [eltype(get_type(operand))])
    MLIR.add_operands!(state, [operand, I...])
    Operation(state)
end

end # module tensor

module mhlo

using ..MLIR
using ..MLIR: get_type, julia_type, add_operands!, add_results!, add_attributes!

function dot(context, operands; loc=Location(context))
    state = OperationState("mhlo.dot", loc)
    MLIR.add_operands!(state, operands)
    Ta = MLIR.get_type(first(operands))
    sa = size(Ta)
    Tb = MLIR.get_type(last(operands))
    sb = size(Tb)
    output_size = (sa[begin], sb[end])
    result_type = MType(context, eltype(Ta), output_size)
    MLIR.add_results!(state, [result_type])
    Operation(state)
end

function reshape(context, operand, shape; loc=Location(context))
    state = OperationState("mhlo.reshape", loc)
    out_type = MType(context, eltype(MLIR.get_type(operand)), shape)
    MLIR.add_results!(state, [out_type])
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

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

for f in (:minimum, :maximum, :add, :substract, :divide, :mul)
    @eval function $f(context, operands, out_type=MLIR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("mhlo.", f)), loc)
        MLIR.add_results!(state, [out_type])
        MLIR.add_operands!(state, operands)
        Operation(state)
    end
end

function convolution(context, output_type, operands; loc=Location(context))
    state = OperationState("mhlo.convolution", loc)
    MLIR.add_attributes!(state, [
        NamedAttribute(context, "batch_group_count", Attribute(context, 1)),
        NamedAttribute(context, "dimension_numbers", parse(
            Attribute,
            context,
            """
            #mhlo.conv<raw
              input_batch_dimension = 3,
              input_feature_dimension = 2,
              input_spatial_dimensions = [0, 1],
              kernel_input_feature_dimension = 2,
              kernel_output_feature_dimension = 3,
              kernel_spatial_dimensions = [0, 1],
              output_batch_dimension = 3,
              output_feature_dimension = 2,
              output_spatial_dimensions = [0, 1],
            >
            """
        )),
        NamedAttribute(context, "feature_group_count", Attribute(context, 1)),
        NamedAttribute(context, "padding", Attribute(context, zeros(Int64, 2, 2))),
        NamedAttribute(context, "rhs_dilation", Attribute(context, [1, 1])),
        NamedAttribute(context, "window_strides", Attribute(context, [1, 1])),
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
    MLIR.add_results!(state, [output_type])
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

function reduce(context, inner_block, operands, dims; loc=Location(context))
    input_type = MLIR.get_type(first(operands))
    input_size = size(input_type)

    etype = eltype(input_type)
    output_dim = [
        s
        for (i, s) in enumerate(input_size)
        if i ∉ dims
    ]
    output_type = MType(context, etype, output_dim)

    region = Region()
    push!(region, inner_block)

    state = OperationState("mhlo.reduce", loc)
    MLIR.add_owned_regions!(state, [region])

    MLIR.add_results!(state, [output_type])
    MLIR.add_attributes!(state, [
        NamedAttribute(context, "dimensions",
            Attribute(context, collect(dims .- 1))),
    ])
    MLIR.add_operands!(state, operands)

    Operation(state)
end

end # module mhlo

