using Revise
using Coil

using Flux
using Metalhead

model = Chain([
    Dense(10, 10, relu)
    for _ in 1:2
])

iree_model = Coil.Tracing.iree(Coil.Tracing.get_session(), model)
@show iree_model

cmodel = Coil.compile(model)

x = randn(Float32, 10, 10)

out1 = cmodel(x)
out2 = cmodel(x)

@show out1 ≈ out2
