module Passes

import ..MLIR
import ..mhlo

# helper to run an AbstractPass pass on a module
function MLIR.run(pass::MLIR.AbstractPass, mod::MLIR.MModule)
    ctx = mod.context
    opname = MLIR.get_opname(pass)

    nameof_pass = string(nameof(typeof(pass)))

    pm = MLIR.PassManager(ctx)
    mlir_pass = MLIR.create_external_pass!(pm, pass, nameof_pass, nameof_pass, "", opname)

    if isempty(opname)
        MLIR.add_owned_pass!(pm, mlir_pass)
    else
        @assert MLIR.is_registered_operation(ctx, opname) "$opname is not registered"
        opm = MLIR.OpPassManager(pm, opname)
        MLIR.add_owned_pass!(opm, mlir_pass)
    end

    MLIR.run(pm, mod)
end

"""
    TransposeArgsToRowMajorPass() <: MLIR.AbstractPass

This pass replaces the argument for `func.func` operations with a transposed version and
putting reshape operations at the beginning of the block.

The type of the function therefore goes from:
MType(#= (tensor<2x3xf32>) -> tensor<3x2xf32> =#)

to:
MType(#= (tensor<3x2xf32>) -> tensor<3x2xf32> =#)
"""
struct TransposeArgsToRowMajorPass <: MLIR.AbstractPass end

MLIR.get_opname(::TransposeArgsToRowMajorPass) = "func.func"

function MLIR.pass_run(ctx::MLIR.Context, ::TransposeArgsToRowMajorPass, func_op::MLIR.Operation)
    block = MLIR.get_first_block(func_op)
    isnothing(block) && return

    ftype_attr = MLIR.get_attribute_by_name(func_op, "function_type")
    ftype = MLIR.get_type_value(ftype_attr)

    arg_types = [
        MLIR.get_input(ftype, i)
        for i in 1:MLIR.get_num_inputs(ftype)
    ]

    last_reshape_op = nothing

    Nargs = MLIR.num_arguments(block)
    for iarg in 1:Nargs
        arg = MLIR.get_argument(block, iarg)
        typeof_arg = MLIR.get_type(arg)

        if MLIR.is_tensor(typeof_arg) && ndims(typeof_arg) > 1
            old_shape = size(typeof_arg)
            new_shape = reverse(old_shape)

            new_type = MLIR.MType(ctx, eltype(typeof_arg), new_shape)
            arg_types[iarg] = new_type
            MLIR.set_type!(arg, new_type)

            perm = reverse(1:ndims(typeof_arg))
            reshape_op = mhlo.transpose(ctx, arg, perm)

            if isnothing(last_reshape_op)
                pushfirst!(block, reshape_op)
                last_reshape_op = reshape_op
            else
                MLIR.insert_after!(block, last_reshape_op, reshape_op)
                last_reshape_op = reshape_op
            end

            reshaped_arg = MLIR.get_result(reshape_op, 1)

            for (use_op, iuse) in Iterators.drop(MLIR.uses(arg), 1)
                MLIR.set_operand!(use_op, iuse, reshaped_arg)
            end
        end
    end

    if MLIR.get_num_results(ftype) != 1
        throw("unsupported num_results $(MLIR.get_num_results(ftype))")
    end

    result_type = MLIR.get_result(ftype, 1)
    new_ftype = MLIR.MType(ctx, arg_types => (result_type,))
    new_ftype_attr = MLIR.Attribute(new_ftype)
    MLIR.set_attribute_by_name!(func_op, "function_type", new_ftype_attr)
end

struct TransposeReturnTypePass <: MLIR.AbstractPass end

MLIR.get_opname(::TransposeReturnTypePass) = "func.func"

function MLIR.pass_run(ctx::MLIR.Context, ::TransposeReturnTypePass, func_op::MLIR.Operation)
    block = MLIR.get_first_block(func_op)

    ret_op = nothing
    for op in MLIR.OperationIterator(block)
        if MLIR.get_name(op) == "func.return"
            ret_op = op
            break
        end
    end

    if isnothing(ret_op) || MLIR.num_operands(ret_op) != 1
        return
    end

    operand = MLIR.get_operand(ret_op, 1)
    ret_type = MLIR.get_type(operand)

    if !MLIR.is_tensor(ret_type) || ndims(ret_type) <= 1
        return
    end

    block = MLIR.get_block(ret_op)

    op_shape = size(ret_type)
    perm = reverse(1:ndims(ret_type))

    transpose_op = mhlo.transpose(ctx, operand, perm)
    MLIR.insert_before!(block, ret_op, transpose_op)

    new_ret_operand = MLIR.get_result(transpose_op, 1)
    MLIR.set_operand!(ret_op, 1, new_ret_operand)

    ftype_attr = MLIR.get_attribute_by_name(func_op, "function_type")
    ftype = MLIR.get_type_value(ftype_attr)

    arg_types = [
        MLIR.get_input(ftype, i)
        for i in 1:MLIR.get_num_inputs(ftype)
    ]

    new_result_type = MLIR.get_type(new_ret_operand)
    new_ftype = MLIR.MType(ctx, arg_types => (new_result_type,))
    new_ftype_attr = MLIR.Attribute(new_ftype)

    MLIR.set_attribute_by_name!(func_op, "function_type", new_ftype_attr)
end

end # module Passes
