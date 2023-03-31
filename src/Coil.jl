module Coil

include("utils.jl")

include("MLIR/MLIR.jl")
include("IREE/IREE.jl")

include("dialects.jl")
include("passes.jl")

include("tracing.jl")
include("transducer.jl")

import .Tracing: compile, @code_mlir, @code_linalg, @tape
export compile, @code_mlir, @code_linalg

end # module Coil
