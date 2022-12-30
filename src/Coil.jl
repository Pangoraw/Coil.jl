module Coil

include("MLIR/MLIR.jl")
include("IREE/IREE.jl")

include("dialects.jl")
include("tracing.jl")

import .Tracing: compile, @code_mlir
export compile, @code_mlir

using .MLIR

end # module Coil
