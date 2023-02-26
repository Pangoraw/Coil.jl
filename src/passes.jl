module Passes

import ..MLIR
import ..mhlo

# helper to run an AbstractPass pass on a module
function MLIR.run(mod::MLIR.MModule, pass::MLIR.AbstractPass)
    ctx = mod.context
    opname = MLIR.get_opname(pass)

    nameof_pass = string(nameof(typeof(pass)))

    pm = MLIR.PassManager(ctx)
    mlir_pass = MLIR.create_external_pass!(pm, pass, nameof_pass, nameof_pass, "", opname)

    if isempty(opname)
        MLIR.add_owned_pass!(pm, pass)
    else
        @assert MLIR.is_registered_operation(ctx, opname) "$opname is not registered"
        MLIR.add_owned_pass!(MLIR.OpPassManager(pm, opname), mlir_pass)
    end

    MLIR.run(pm, mod)
end

"""
    ReshapeArgsToRowMajorPass() <: MLIR.AbstractPass

This pass replaces the argument for `func.func` operations with a transposed version and
putting reshape operations at the beginning of the block.

The type of the function therefore goes from:
MType(#= (tensor<2x3xf32>) -> tensor<3x2xf32> =#)

to:
MType(#= (tensor<3x2xf32>) -> tensor<3x2xf32> =#)
"""
struct ReshapeArgsToRowMajorPass <: MLIR.AbstractPass end

MLIR.get_opname(::ReshapeArgsToRowMajorPass) = "func.func"

function MLIR.pass_run(ctx::MLIR.Context, ::ReshapeArgsToRowMajorPass, func_op::MLIR.Operation)
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

            reshape_op = mhlo.reshape(ctx, arg, old_shape)

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

end # module Passes
