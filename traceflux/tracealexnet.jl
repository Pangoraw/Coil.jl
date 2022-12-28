using Revise 
using Coil

using Metalhead
using Flux


conv = Conv((3,3), 3 => 12) 
cconv = Coil.compile(conv)

img = randn(Float32, 224, 224, 3, 1)
@time cconv(img)
@time cconv(img)


#=
anet = Metalhead.AlexNet()
canet = Coil.compile(anet)

# @time anet(img)
@time canet(img)
@time canet(img)
=#
