using Revise 
using Coil

using Metalhead
using Flux

img = randn(Float32, 224, 224, 3, 1)

#=
c = Conv((3,3), 3 => 12) 
cc = Coil.compile(c; verbose=true)

@time cc(img)
@time cc(img)
=#

anet = Metalhead.AlexNet()
canet = Coil.compile(anet)

# @time anet(img)
@time canet(img)
@time canet(img)
