module func
# https://mlir.llvm.org/docs/Dialects/Func/

using ..MLIR
using ..MLIR: get_type, julia_type

function return_(operands; loc=Location())
    MLIR.create_operation("func.return", loc; operands, results=[])
end

end # module func

module math
# https://mlir.llvm.org/docs/Dialects/MathOps/

using ..MLIR
using ..MLIR: get_type, julia_type

for op in (:cos, :sin, :tan, :tanh, :exp, :sqrt)
    @eval function $op(operand; loc=Location())
        MLIR.create_operation($(string("math.", op)), loc;
            operands=[operand],
            results=[MLIR.get_type(operand)])
    end
end

end # module math

module arith
# https://mlir.llvm.org/docs/Dialects/ArithOps/

using ..MLIR
using ..MLIR: get_type, julia_type

for (f, t) in Iterators.product(
    (:add, :sub, :mul),
    (:i, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(operands, type=MLIR.get_type(first(operands)); loc=Location())
        MLIR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui),
)
    fname = Symbol(f, t)
    @eval function $fname(operands, type=MLIR.get_type(first(operands)); loc=Location())
        MLIR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

for f in (:div, :maximum, :minimum)
    fname = Symbol(f, "f")
    @eval function $fname(operands, type=MLIR.get_type(first(operands)); loc=Location())
        MLIR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-mlirarithindexcastop
for f in (:index_cast, :index_castui)
    @eval function $f(operand; loc=Location())
        MLIR.create_operation($(string("arith.", f)), loc;
            operands=[operand],
            results=[MLIR.IndexType()])
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-mlirarithextfop
function extf(operand, type; loc=Location())
    MLIR.create_operation("arith.exf", loc; operands=[operand], results=[type])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-mlirarithsitofpop
function sitofp(operand, ftype=float(julia_type(eltype(get_type(operand)))); loc=Location())
    type = get_type(operand)
    MLIR.create_operation("arith.sitofp", loc; operands=[operand], results=[
        MLIR.is_tensor(type) ?
        MLIRType(ftype isa MLIRType ? eltype(ftype) : MLIRType(ftype), size(type)) :
        MLIRType(ftype)
    ])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-mlirarithfptosiop
function fptosi(operand, itype; loc=Location())
    type = get_type(operand)
    MLIR.create_operation("arith.fptosi", loc; operands=[operand], results=[
        MLIR.is_tensor(type) ?
        MLIRType(itype isa MLIRType ? itype : MLIRType(itype), size(type)) :
        MLIRType(itype)
    ])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(
    values::AbstractArray,
    type=MLIRType(MLIRType(eltype(values)), size(values));
    loc=Location())

    MLIR.create_operation("arith.constant", loc; results=[type], attributes=[
        MLIR.NamedAttribute("value",
            Attribute(values, type)),
    ])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(value, type=MLIRType(typeof(value)); loc=Location())
    MLIR.create_operation("arith.constant", loc; results=[type], attributes=[
        MLIR.NamedAttribute("value", Attribute(value, type)),
    ])
end

end # module arith

module tosa
# https://mlir.llvm.org/docs/Dialects/TOSA/

using ..MLIR
using ..MLIR: get_type, julia_type

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosamaximum-mlirtosamaximumop
# https://mlir.llvm.org/docs/Dialects/TOSA/#tosaminimum-mlirtosaminimumop
for f in (:maximum, :minimum)
    @eval function $f(operands; loc=Location())
        MLIR.create_operation($(string("tosa.", f)), loc; operands,
            results=[MLIR.get_type(first(operands))])
    end
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosaconst-mlirtosaconstop
function const_(value, type=MLIRType(typeof(value), size(value)); loc=Location())
    MLIR.create_operation("tosa.const", loc; results=[type], attributes=[
        MLIR.NamedAttribute("value",
            Attribute(value, type)),
    ])
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosareshape-mlirtosareshapeop
function reshape(operand, shape; loc=Location())
    out_type = MLIRType(eltype(MLIR.get_type(operand)), shape)
    MLIR.create_operation("tosa.reshape", loc;
        results=[out_type],
        operands=[operand],
        attributes=[
            NamedAttribute("new_shape", MLIR.ArrayAttribute(collect(shape)))
        ])
end

# https://mlir.llvm.org/docs/Dialects/TOSA/#tosareduce_sum-mlirtosareducesumop
function reduce_sum(input, axis; loc=Location())
    type = get_type(input)
    input_shape = size(type)
    output_shape = [i == axis ? 1 : d for (i, d) in enumerate(input_shape)]

    MLIR.create_operation("tosa.reduce_sum", loc;
        results=[MLIRType(eltype(type), output_shape)],
        operands=[input],
        attributes=[
            NamedAttribute("axis", Attribute(axis - 1))
        ])
end

#=
#https://mlir.llvm.org/docs/Dialects/TOSA/#tosamax_pool2d-mlirtosamaxpool2dop
function maxpool_2d(operand, kernel, stride, pad; loc=Location())
    MLIR.create_operation("tosa.maxpool_2d", loc;
    operands=[operand],
    attributes=[
        NamedAttribute("kernel", Attribute(kernel)),
        NamedAttribute("stride", Attribute(stride))
        NamedAttribute("pad", Attribute(pad))
    ])
    input_type = get_type(operand)
    input_size = input_size
    add_results!(state, [output_type])
    Operation(state)
end
=#

end # module tosa

module linalg

using ..MLIR
using ..MLIR: get_type, julia_type


function map_as_generic(inner_block, in, out; loc=Location())
    region = Region()
    push!(region, inner_block)

    typ = MLIR.get_type(in)
    MLIR.push_argument!(inner_block, eltype(typ), loc)

    affine_map = MLIR.API.mlirAffineMapMultiDimIdentityGet(MLIR.context(), ndims(typ))
    affine_map_attr = MLIR.Attribute(MLIR.API.mlirAffineMapAttrGet(affine_map))
    iterator_attr = parse(Attribute, "#linalg.iterator_type<parallel>")

    MLIR.create_operation("linalg.generic", loc;
        owned_regions=[region],
        results=[typ],
        operands=[in, out],
        attributes=[
            NamedAttribute("indexing_maps", MLIR.ArrayAttribute([affine_map_attr, affine_map_attr])),
            NamedAttribute("iterator_types", MLIR.ArrayAttribute([iterator_attr for _ in 1:ndims(typ)])),
            NamedAttribute("operand_segment_sizes", MLIR.DenseArrayAttribute(Int32[1,1])),
        ],
    )
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmap-mlirlinalgmapop
function map(inner_block, in, out; loc=Location())
    region = Region()
    push!(region, inner_block)

    MLIR.create_operation("linalg.map", loc;
        owned_regions=[region],
        results=[MLIR.get_type(out)],
        operands=[in, out])
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgreduce-mlirlinalgreduceop
function reduce(
    inner_block,
    operands,
    dims=1:ndims(get_type(in));
    loc=Location()
)
    region = Region()
    push!(region, inner_block)
    MLIR.create_operation("linalg.reduce", loc;
        owned_regions=[region],
        results=[get_type(last(operands))],
        operands,
        attributes=[
            NamedAttribute("dimensions",
                MLIR.DenseArrayAttribute(collect(dims .- 1))),
        ])
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgyield-mlirlinalgyieldop
function yield(operand; loc=Location())
    MLIR.create_operation("linalg.yield", loc; operands=[operand], results=[])
end

# https://mlir.llvm.org/docs/Dialects/Linalg/#linalgfill-mlirlinalgfillop
function fill(operands...; loc=Location())
    MLIR.create_operation("linalg.fill", loc;
        owned_regions=[Region()],
        operands=collect(operands),
        results=[get_type(last(operands))],
        attributes=[
            NamedAttribute("operand_segment_sizes",
                Attribute(collect(Int32.(length.(size.(get_type.(operands))))))),
        ])
end

end # module linalg

module tensor

using ..MLIR
using ..MLIR: get_type, julia_type

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert-tensorinsertop
function insert(scalar, dest, indices...; loc=Location())
    MLIR.create_operation("tensor.insert", loc;
                          results=[MLIR.get_type(dest)],
                          operands=Value[scalar, dest, indices...])
end

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-mlirtensoremptyop
function empty(type; loc=Location())
    MLIR.create_operation("tensor.empty", loc;
        results=[type])
end

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract-mlirtensorextractop
function extract(operand, I...; loc=Location())
    MLIR.create_operation("tensor.extract", loc;
        results=[eltype(get_type(operand))],
        operands=[operand, I...])
end

# Using https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcollapse_shape-mlirtensorcollapseshapeop
function transpose(operand, perm; loc=Location())
    reassociation = [[d] for d in perm]
    tensor.collapse_shape(operand, reassociation; loc)
end

# https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcollapse_shape-mlirtensorcollapseshapeop
function collapse_shape(operand, reassociation; loc=Location())
    input_type = MLIR.get_type(operand)
    input_size = size(input_type)
    output_size = [
        prod([input_size[i] for i in dim_assoc])
        for dim_assoc in reassociation
    ]
    output_type = MLIRType(eltype(input_type), output_size)

    MLIR.create_operation("tensor.collapse_shape", loc;
        operands=[operand],
        attributes=[
            MLIR.NamedAttribute(
                "reassociation",
                MLIR.ArrayAttribute(
                    [MLIR.ArrayAttribute(Int64.(dim_assoc .- 1))
                     for dim_assoc in reassociation],
                ),
            )
        ],
        results=[output_type])
end

end # module tensor

module mhlo

using ..MLIR
using ..MLIR: get_type, julia_type

function dot(operands; loc=Location())
    Ta = MLIR.get_type(first(operands))
    sa = size(Ta)
    Tb = MLIR.get_type(last(operands))
    sb = size(Tb)
    output_size = (sa[begin], sb[end])
    result_type = MLIRType(eltype(Ta), output_size)

    MLIR.create_operation("mhlo.dot", loc; operands,
        results=[result_type])
end

function reshape(operand, shape; loc=Location())
    out_type = MLIRType(eltype(MLIR.get_type(operand)), shape)
    MLIR.create_operation("mhlo.reshape", loc;
        results=[out_type],
        operands=[operand])
end

function transpose(operand, perm; loc=Location())

    input_type = MLIR.get_type(operand)
    input_shape = size(input_type)
    out_shape = [
        input_shape[i]
        for i in perm
    ]
    out_type = MLIRType(eltype(input_type), out_shape)

    MLIR.create_operation("mhlo.transpose", loc;
        operands=[operand],
        results=[out_type],
        attributes=[
            MLIR.NamedAttribute(
                "permutation",
                MLIR.Attribute(collect(Int64.(perm) .- 1)),
            ),
        ])
end

function return_(operand; loc=Location())
    MLIR.create_operation("mhlo.return", loc;
        operands=[operand], results=[])
end

function map(block, operand, dims=collect(1:ndims(operand)); loc=Location())

    region = Region()
    push!(region, block)

    MLIR.create_operation("mhlo.map", loc;
        owned_regions=[region],
        operands=[operand],
        results=[MLIR.get_type(operand)],
        attributes=[
            MLIR.NamedAttribute(
                "dimensions",
                MLIR.Attribute(collect(dims .- 1)),
            ),
        ])
end

for f in (:minimum, :maximum, :add, :substract, :divide, :multiply)
    @eval function $f(operands, out_type=MLIR.get_type(first(operands)); loc=Location())
        MLIR.create_operation($(string("mhlo.", f)), loc;
            results=[out_type], operands)
    end
end

function reduce_window(
    operands,
    region,
    window_dimensions,
    window_strides,
    base_dilations,
    window_dilations,
    padding,
    output_type;
    loc=Location()
)
    MLIR.create_operation("mhlo.reduce_window", loc;
        owned_regions=[region],
        results=[output_type],
        attributes=[
            NamedAttribute("window_dimensions", Attribute(collect(Int64.(window_dimensions)))),
            NamedAttribute("window_strides", Attribute(collect(Int64.(window_strides)))),
            NamedAttribute("base_dilations", Attribute(collect(Int64.(base_dilations)))),
            NamedAttribute("window_dilations", Attribute(collect(Int64.(window_dilations)))),
            NamedAttribute("padding", Attribute(collect(Int64.(padding)))),
        ],
        operands=collect(operands))
end

# https://www.tensorflow.org/mlir/hlo_ops#mhloreverse_mlirmhloreverseop
function reverse(operand, dims; loc=Location())
    MLIR.create_operation("mhlo.reverse", loc;
        operands=MLIR.Value[operand],
        attributes=[
            NamedAttribute("dimensions", Attribute(collect(dims .- 1))),
        ],
        results=[MLIR.get_type(operand)])
end

function convolution(
    output_type,
    operands,
    padding,
    rhs_dilation,
    window_strides; loc=Location()
)
    MLIR.create_operation("mhlo.convolution", loc;
        attributes=[
            NamedAttribute("batch_group_count", Attribute(1)),
            NamedAttribute("dimension_numbers", parse(
                Attribute,
                """
                #mhlo.conv<raw
                  input_batch_dimension = 3,
                  input_feature_dimension = 2,
                  input_spatial_dimensions = [0, 1],
                  kernel_output_feature_dimension = 3,
                  kernel_input_feature_dimension = 2,
                  kernel_spatial_dimensions = [0, 1],
                  output_batch_dimension = 3,
                  output_feature_dimension = 2,
                  output_spatial_dimensions = [0, 1],
                >
                """
            )),
            NamedAttribute("feature_group_count", Attribute(1)),
            NamedAttribute("padding", Attribute(padding)),
            NamedAttribute("rhs_dilation", Attribute(rhs_dilation)),
            NamedAttribute("lhs_dilation", Attribute(ones(Int64, length(rhs_dilation)))),
            NamedAttribute("window_strides", Attribute(window_strides)),
            # NamedAttribute("feature_group_count", Attribute(1)),
            # NamedAttribute("padding", Attribute(fill(2, 2, 2))),
            # NamedAttribute("rhs_dilation", Attribute()),
            # NamedAttribute("window_strides", Attribute([4, 4])),
        ],
        results=[output_type],
        operands)
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

function broadcast_in_dim(operand, new_size; loc=Location())
    type = MLIR.get_type(operand)

    type_size = size(type)
    broadcast_dimensions = _compute_broadcast_dims(type_size, new_size)

    MLIR.create_operation("mhlo.broadcast_in_dim", loc;
        operands=[operand],
        results=[
            MLIRType(eltype(type), new_size),
        ],
        attributes=[
            MLIR.NamedAttribute(
                "broadcast_dimensions",
                Attribute(broadcast_dimensions .- 1),
            )
        ])
end

function compare(operands, direction; loc=Location())
    MLIR.create_operation("mhlo.compare", loc;
        operands,
        results=[MLIRType(MLIRType(Bool), ())],
        attributes=[
            NamedAttribute(
                "comparison_direction",
                parse(Attribute, "#mhlo<comparison_direction $direction>"),
            ),
        ])
end

function if_(cond, result, block1, block2; loc=Location())

    region1 = Region()
    push!(region1, block1)
    region2 = Region()
    push!(region2, block2)

    MLIR.create_operation("mhlo.if", loc;
        operands=[cond],
        results=[result],
        owned_regions=[region1, region2])
end

function constant(value; loc=Location())
    etype = MLIRType(eltype(value))
    ranked_type = MLIRType(etype, size(value))

    MLIR.create_operation("mhlo.constant", loc;
        results=[ranked_type,],
        attributes=[
            MLIR.NamedAttribute("value", Attribute(fill(value)))
        ])
end

function reduce(inner_block, operands, dims; loc=Location())
    input_type = MLIR.get_type(first(operands))
    input_size = size(input_type)

    etype = eltype(input_type)
    output_dim = [
        s
        for (i, s) in enumerate(input_size)
        if i ∉ dims
    ]
    output_type = MLIRType(etype, output_dim)

    region = Region()
    push!(region, inner_block)

    MLIR.create_operation("mhlo.reduce", loc;
        owned_regions=[region],
        results=[output_type],
        attributes=[
            NamedAttribute("dimensions",
                Attribute(collect(dims .- 1))),
        ], operands)
end

end # module mhlo

module cf

using ..MLIR

# https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfassert-mlircfassertop
function assert(cond, message; loc=Location())
    MLIR.create_operation("cf.assert", loc;
        operands=[cond],
        attributes=[
            MLIR.NamedAttribute("msg",
                Attribute(message)),
        ])
end

# https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfbr-mlircfbranchop
function br(dest, operands; loc=Location())
    MLIR.create_operation("cf.br", loc;
        successors=MLIR.Block[dest],
        operands=collect(operands))
end

# https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfcond_br-mlircfcondbranchop
function cond_br(
    cond,
    truedest, truedest_operands,
    falsedest, falsedest_operands;
    loc=Location()
)
    MLIR.create_operation("cf.cond_br", loc;
        successors=[truedest, falsedest],
        operands=[
            cond,
            truedest_operands...,
            falsedest_operands...,
        ])
end

end # module cf

module stablehlo
using ..MLIR

function func(ftype; loc=Location())
    MLIR.create_operation("stablehlo.func", loc)
end

function return_(operands...; loc=Location())
    MLIR.create_operation("stablehlo.return", loc;
                          operands=collect(operands))
end

# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape
function reshape(operand, size; loc=Location())
    input_type = MLIR.get_type(operand)
    output_type = MLIRType(eltype(input_type), size)
    MLIR.create_operation("stablehlo.reshape", loc;
        operands=[operand],
        results=[output_type])
end

# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select
function select(pred, on_true, on_false; loc=Location())
    MLIR.create_operation("stablehlo.select", loc;
        operands=[pred, on_true, on_false],
        results=[MLIR.get_type(on_true)])
end

for op in (
    :add, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add
    :and, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and
    :divide, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide
    :maximum, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum
    :minimum, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum
    :multiply, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply
    :or, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or
    :power, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power
    :subtract, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract
)
    @eval function $op(operands; loc=Location())
        MLIR.create_operation($(string("stablehlo.", op)), loc;
                             results=[MLIR.get_type(first(operands))],
                             operands)
    end
end

for op in (
    :cosine, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine
    :exponential, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential
    :exponential_minus_one, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one
    :floor, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor
    :log, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log
    :log_plus_one, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one
    :logistic, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic
    :negate, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate
    :not, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not
    :reverse, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse
    :rsqrt, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt
    :sine, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine
    :sqrt, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt
    :tanh, # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh
)
    @eval function $op(operand; loc=Location())
        MLIR.create_operation($(string("stablehlo.", op)), loc;
                            results=[MLIR.get_type(operand)],
                            operands=[operand])
    end
end

function constant(
    values::AbstractArray,
    type=MLIRType(MLIRType(eltype(values)), size(values));
    loc=Location())

    MLIR.create_operation("stablehlo.constant", loc; results=[type], attributes=[
        MLIR.NamedAttribute("value",
            Attribute(values, type)),
    ])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(value, type=MLIRType(typeof(value)); loc=Location())
    MLIR.create_operation("stablehlo.constant", loc; results=[type], attributes=[
        MLIR.NamedAttribute("value", Attribute(value, type)),
    ])
end

# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size
function get_dimension_size(operand, dimension; loc=Location())
    MLIR.create_operation("stablehlo.get_dimension_size", loc;
        results=[MLIRType(Int32, ())],
        operands=[operand])
end

function transpose(operand, perm; loc=Location())

    input_type = MLIR.get_type(operand)
    input_shape = size(input_type)
    out_shape = [
        input_shape[i]
        for i in perm
    ]
    out_type = MLIRType(eltype(input_type), out_shape)

    MLIR.create_operation("stablehlo.transpose", loc;
        operands=[operand],
        results=[out_type],
        attributes=[
            MLIR.NamedAttribute(
                "permutation",
                MLIR.Attribute(collect(Int64.(perm) .- 1)),
            ),
        ])
end

"""
    _compute_broadcast_dims(::NTuple{N1,Int64}, ::NTuple{N2,Int64})::Vector{Int64}

Returns the dimensions which are broadcasted upon for a given input and broadcast size size.

```julia
julia> Coil.stablehlo._compute_broadcast_dims((10,), (10,1))
Int64[1]

julia> Coil.stablehlo._compute_broadcast_dims((1,1,3,1), (12,12,3,1))
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

function broadcast_in_dim(operand, new_size; loc=Location())
    type = MLIR.get_type(operand)

    type_size = size(type)
    broadcast_dimensions = _compute_broadcast_dims(type_size, new_size)

    MLIR.create_operation("stablehlo.broadcast_in_dim", loc;
        operands=[operand],
        results=[
            MLIRType(eltype(type), new_size),
        ],
        attributes=[
            MLIR.NamedAttribute(
                "broadcast_dimensions",
                Attribute(broadcast_dimensions .- 1),
            )
        ])
end

function convolution(
    output_type,
    operands,
    padding,
    rhs_dilation,
    window_strides; loc=Location()
)
    MLIR.create_operation("stablehlo.convolution", loc;
        attributes=[
            NamedAttribute("batch_group_count", Attribute(1)),
            NamedAttribute("dimension_numbers", parse(
                Attribute,
                """
                #stablehlo.conv<raw
                  input_batch_dimension = 3,
                  input_feature_dimension = 2,
                  input_spatial_dimensions = [0, 1],
                  kernel_output_feature_dimension = 3,
                  kernel_input_feature_dimension = 2,
                  kernel_spatial_dimensions = [0, 1],
                  output_batch_dimension = 3,
                  output_feature_dimension = 2,
                  output_spatial_dimensions = [0, 1],
                >
                """
            )),
            NamedAttribute("feature_group_count", Attribute(1)),
            NamedAttribute("padding", Attribute(padding)),
            NamedAttribute("rhs_dilation", Attribute(rhs_dilation)),
            NamedAttribute("lhs_dilation", Attribute(ones(Int64, length(rhs_dilation)))),
            NamedAttribute("window_strides", Attribute(window_strides)),
            # NamedAttribute("feature_group_count", Attribute(1)),
            # NamedAttribute("padding", Attribute(fill(2, 2, 2))),
            # NamedAttribute("rhs_dilation", Attribute()),
            # NamedAttribute("window_strides", Attribute([4, 4])),
        ],
        results=[output_type],
        operands)
end

function reverse(operand, dims; loc=Location())
    MLIR.create_operation("stablehlo.reverse", loc;
        operands=MLIR.Value[operand],
        attributes=[
            NamedAttribute("dimensions", Attribute(collect(dims .- 1))),
        ],
        results=[MLIR.get_type(operand)])
end

end # module stablehlo
