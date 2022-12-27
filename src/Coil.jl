module Coil

include("MLIR/MLIR.jl")
include("IREE/IREE.jl")

import .MLIR:
    Operation,
    OperationState,
    Block, Region,
    MType, Location,
    Attribute

function gen_return(context, operands; loc=Location(context))
    state = OperationState("func.return", loc)
    MLIR.add_operands!(state, operands)
    Operation(state)
end

function gen_add(context, operands, type; loc=Location(context))
    state = OperationState("arith.addi", loc)
    MLIR.add_operands!(state, operands)
    MLIR.add_results!(state, [type])
    Operation(state)
end

function gen_constant(context, value::T; loc=Location(context)) where {T}
    state = OperationState("arith.constant", loc)
    MLIR.add_results!(state, [MType(context, T)])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value",
            Attribute(context, value)),
            # parse(MLIR.Attribute, context, "42 : i32"))
    ])
    Operation(state)
end

function gen_mhlo_return(context, operand; loc=Location(context))
    state = OperationState("mhlo.return", loc)
    # MLIR.add_results!(state, [MLIR.get_type(operand)])
    MLIR.add_operands!(state, [operand])
    Operation(state)
end

function gen_mhlo_add(context, operands; loc=Location(context))
    state = OperationState("mhlo.add", loc)
    MLIR.add_results!(state, [MLIR.get_type(first(operands))])
    MLIR.add_operands!(state, operands)
    Operation(state)
end

function gen_mhlo_constant(context, value; loc=Location(context))
    etype = MType(context, eltype(value))
    ranked_type = MType(context, etype, size(value))
    state = OperationState("mhlo.constant", loc)
    MLIR.add_results!(state, [ranked_type,])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "value", Attribute(context,  fill(value)))
    ])
    Operation(state)
end

function gen_reduce(context, operands, input_size, dim; loc=Location(context))
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
        gen_mhlo_add(context, [
            MLIR.get_argument(inner_block, 1),
            MLIR.get_argument(inner_block, 2),
        ])
    )
    push!(inner_block,
        gen_mhlo_return(context, MLIR.get_result(add, 1)))

    state = OperationState("mhlo.reduce", loc)
    MLIR.add_owned_regions!(state, [region])

    MLIR.add_results!(state, [output_type])
    MLIR.add_attributes!(state, [
        MLIR.NamedAttribute(context, "dimensions",
            MLIR.Attribute(context, [dim-1])),
    ])
    MLIR.add_operands!(state, operands)

    Operation(state)
end

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

    add = push!(block,
        gen_add(context, [MLIR.get_argument(block, 1), MLIR.get_argument(block, 2)], rettypes[1]))
    add0 = MLIR.get_result(add, 1)
    push!(block,
        gen_return(context, [add0]))

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

    constant = gen_constant(context, Int32(42))
    cst0 = MLIR.get_result(constant, 1)

    push!(block, constant)
    push!(block,
        gen_return(context, [cst0]))

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
    MLIR.add_attributes!(state,[
        MLIR.NamedAttribute(context, "sym_name", Attribute(context, name)),
        MLIR.NamedAttribute(context, "function_type", Attribute(functype)),
    ])

    ###

    cst0 = gen_mhlo_constant(context, zero(Float32))
    push!(block, cst0)

    arg0 = MLIR.get_argument(block, 1)
    vcst0 = MLIR.get_result(cst0, 1)
    reduce0 = gen_reduce(context, [arg0, vcst0], (10,), 1)
    push!(block, reduce0)

    ret = gen_return(context, [MLIR.get_result(reduce0, 1)])
    push!(block, ret)

    ###

    Operation(state)
end

end # module Coil
