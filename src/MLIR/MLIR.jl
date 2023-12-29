module MLIR

include("./LibMLIR.jl")
const API = LibMLIR
const IR = @__MODULE__

include("./IR.jl")
include("./state.jl")

function get_dialects!(dialects::Set{Symbol}, op::Operation)
    push!(dialects, dialect(op))

    visit(op) do op
        get_dialects!(dialects, op)
    end

    dialects
end

function get_input_type(module_)
    dialects = Set{Symbol}()

    op = get_operation(module_)
    get_dialects!(dialects, op)

    if :tosa ∈ dialects
        :tosa
    elseif :stablehlo ∈ dialects
        :stablehlo
    else
        :none
    end
end

end # module MLIR
