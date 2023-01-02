module Coil

include("MLIR/MLIR.jl")
include("IREE/IREE.jl")

include("dialects.jl")
include("tracing.jl")

import .Tracing: compile, @code_mlir, @code_linalg, @tape
export compile, @code_mlir, @code_linalg

end # module Coil
