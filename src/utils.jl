import Libdl

dlvsym(h, s, v="VER_0") = @ccall dlvsym(h::Ptr{Cvoid}, s::Cstring, v::Cstring)::Ptr{Cvoid}

const handles = Dict{String,Ptr{Cvoid}}()

"""
The `libIREECompiler.so` shared library provided in the GitHub releases is stripped and
versionned using glibc, we use `dlvsym` to recover the symbols.
"""
macro cvcall(call, ver="VER_0")
    @assert Meta.isexpr(call, :(::))
    def = call.args[1].args[1]
    @assert Meta.isexpr(def, :(.), 2)
    lib, name = def.args

    f = gensym(:func)
    call.args[1].args[1] = Expr(Symbol("\$"), f)

    cexpr = var"@ccall"(__source__, __module__, call)
    ref = Ref(C_NULL)

    quote
        $(esc(f)) = let
            handles = $(handles)
            handle = get!(handles, $(esc(lib))) do
                Libdl.dlopen($(esc(lib)))
            end

            ref = $(ref)
            if ref[] == C_NULL
                ref[] = dlvsym(handle, $(name), $(esc(ver)))
            end
            @assert ref[] != C_NULL

            ref[]
        end

        $cexpr
    end
end
