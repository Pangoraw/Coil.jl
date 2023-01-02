module MLIR

include("./LibMLIR.jl")

export
    Operation,
    OperationState,
    Location,
    Context,
    MModule,
    MType,
    Region,
    Block,
    Attribute,
    NamedAttribute

export
    add_results!,
    add_attributes!,
    add_owned_regions!

import Base: ==, String
using .LibMLIR:
    MlirAttribute,
    MlirNamedAttribute,
    MlirDialect,
    MlirStringRef,
    MlirOperation,
    MlirOperationState,
    MlirLocation,
    MlirBlock,
    MlirRegion,
    MlirModule,
    MlirContext,
    MlirType,
    MlirValue,
    MlirIdentifier,
    MlirPassManager,
    MlirOpPassManager

function print_callback(str::MlirStringRef, userdata)
    data = unsafe_wrap(Array, str.char, str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

### String Ref

String(strref::MlirStringRef) = Base.unsafe_string(strref.char, strref.length)
Base.convert(::Type{MlirStringRef}, s::String) =
    MlirStringRef(Base.unsafe_convert(Ptr{Cchar}, s), sizeof(s))

### Identifier

String(ident::MlirIdentifier) = String(LibMLIR.mlirIdentifierStr(ident))

### Dialect

struct Dialect
    dialect::MlirDialect

    Dialect(dialect) = begin
        @assert !LibMLIR.mlirDialectIsNull(dialect) "cannot create Dialect from null MlirDialect"
        new(dialect)
    end
end

Base.convert(::Type{MlirDialect}, dialect::Dialect) = dialect.dialect
function Base.show(io::IO, dialect::Dialect)
    print(io, "Dialect(\"", String(LibMLIR.mlirDialectGetNamespace(dialect)), "\")")
end

### Context

mutable struct Context
    context::MlirContext

    Context(context) = begin
        @assert !LibMLIR.mlirContextIsNull(context) "cannot create Context with null MlirContext"
        finalizer(new(context)) do context
            LibMLIR.mlirContextDestroy(context.context)
        end
    end
end
Context() = Context(LibMLIR.mlirContextCreate())

Base.convert(::Type{MlirContext}, c::Context) = c.context

num_loaded_dialects(context) = LibMLIR.mlirContextGetNumLoadedDialects(context)
function get_or_load_dialect(context, dialect)
    mlir_dialect = LibMLIR.mlirContextGetOrLoadDialect(context, dialect)
    if LibMLIR.mlirDialectIsNull(mlir_dialect)
        error("could not load dialect \"$dialect\"")
    else
        Dialect(mlir_dialect)
    end
end
is_registered_operation(context, opname) = LibMLIR.mlirContextIsRegisteredOperation(context, opname)

### Location

struct Location
    location::MlirLocation

    Location(location) = begin
        @assert !LibMLIR.mlirLocationIsNull(location) "cannot create Location with null MlirLocation"
        new(location)
    end
end

Location(context::Context) = Location(LibMLIR.mlirLocationUnknownGet(context))
Location(context::Context, filename, line, column=0) =
    Location(LibMLIR.mlirLocationFileLineColGet(context, filename, line, column))
Location(context::Context, lin::Core.LineInfoNode) =
    Location(context, string(lin.file), lin.line)
Location(context::Context, lin::LineNumberNode) =
    isnothing(lin.file) ?
    Location(context) :
    Location(context, string(lin.file), lin.line)
Location(context::Context, ::Nothing) = Location(context)

Base.convert(::Type{MlirLocation}, location::Location) = location.location

function Base.show(io::IO, location::Location)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    print(io, "Location(#= ")
    GC.@preserve ref LibMLIR.mlirLocationPrint(location, c_print_callback, ref)
    print(io, " =#)")
end

### Type

struct MType
    type::MlirType

    MType(type) = begin
        @assert !LibMLIR.mlirTypeIsNull(type)
        new(type)
    end
end

MType(t::MType) = t
MType(context::Context, T::Type{<:Signed}) =
    MType(LibMLIR.mlirIntegerTypeGet(context, sizeof(T) * 8))
MType(context::Context, T::Type{<:Unsigned}) =
    MType(LibMLIR.mlirIntegerTypeGet(context, sizeof(T) * 8))
MType(context::Context, ::Type{Bool}) =
    MType(LibMLIR.mlirIntegerTypeSignedGet(context, 1))
MType(context::Context, ::Type{Float32}) =
    MType(LibMLIR.mlirF32TypeGet(context))
MType(context::Context, ::Type{Float64}) =
    MType(LibMLIR.mlirF64TypeGet(context))
MType(context::Context, ft::Pair) =
    MType(LibMLIR.mlirFunctionTypeGet(context,
        length(ft.first), [MType(t) for t in ft.first],
        length(ft.second), [MType(t) for t in ft.second]))
MType(context, a::AbstractArray{T}) where {T} = MType(context, MType(context, T), size(a))
MType(context, ::Type{<:AbstractArray{T,N}}, dims) where {T,N} =
    MType(LibMLIR.mlirRankedTensorTypeGetChecked(
        Location(context),
        N, collect(dims),
        MType(context, T),
        Attribute(),
    ))
MType(context, element_type::MType, dims) =
    MType(LibMLIR.mlirRankedTensorTypeGetChecked(
        Location(context),
        length(dims), collect(dims),
        element_type,
        Attribute(),
    ))
MType(context, ::T) where {T<:Real} = MType(context, T)
MType(_, type::MType) = type

IndexType(context) = MType(LibMLIR.mlirIndexTypeGet(context))

Base.convert(::Type{MlirType}, mtype::MType) = mtype.type

function Base.eltype(type::MType)
    if LibMLIR.mlirTypeIsAShaped(type)
        MType(LibMLIR.mlirShapedTypeGetElementType(type))
    else
        type
    end
end

function show_inner(io::IO, type::MType)
    if LibMLIR.mlirTypeIsAInteger(type)
        is_signless = LibMLIR.mlirIntegerTypeIsSignless(type)
        is_signed = LibMLIR.mlirIntegerTypeIsSigned(type)

        width = LibMLIR.mlirIntegerTypeGetWidth(type)
        t = if is_signed
            "si" 
       elseif is_signless
            "i"
        else
            "u"
        end
        print(io, t, width)
    elseif LibMLIR.mlirTypeIsAF64(type)
        print(io, "f64")
    elseif LibMLIR.mlirTypeIsAF32(type)
        print(io, "f32")
    elseif LibMLIR.mlirTypeIsARankedTensor(type)
        print(io, "tensor<")
        s = size(type)
        print(io, join(s, "x"), "x")
        show_inner(io, eltype(type))
        print(io, ">")
    elseif LibMLIR.mlirTypeIsAIndex(type)
        print(io, "index")
    else
        print(io, "unknown")
    end
end

function Base.show(io::IO, type::MType)
    print(io, "MType(\"")
    show_inner(io, type)
    print(io, "\")")
end

function julia_type(type::MType)
    if LibMLIR.mlirTypeIsAInteger(type)
        is_signed = LibMLIR.mlirIntegerTypeIsSigned(type) ||
                    LibMLIR.mlirIntegerTypeIsSignless(type)
        width = LibMLIR.mlirIntegerTypeGetWidth(type)

        if (is_signed, width) == (true, 32)
            Int32
        elseif (is_signed, width) == (false, 32)
            UInt32
        elseif (is_signed, width) == (true, 16)
            Int16
        elseif (is_signed, width) == (false, 16)
            UInt16
        elseif (is_signed, width) == (true, 8)
            Int8
        elseif (is_signed, width) == (false, 8)
            UInt8
        elseif (is_signed, width) == (true, 64)
            Int64
        elseif (is_signed, width) == (false, 64)
            UInt64
        elseif (is_signed, width) == (true, 1)
            Bool
        else
            t = is_signed ? "i" : "u"
            throw("could not convert type $(t)$(width) to julia")
        end
    elseif LibMLIR.mlirTypeIsAF32(type)
        Float32
    elseif LibMLIR.mlirTypeIsAF64(type)
        Float64
    else
        throw("could not convert type $type to julia")
    end
end

Base.ndims(type::MType) =
    if LibMLIR.mlirTypeIsAShaped(type) && LibMLIR.mlirShapedTypeHasRank(type)
        LibMLIR.mlirShapedTypeGetRank(type)
    else
        0
    end

Base.size(type::MType, i::Int) = LibMLIR.mlirShapedTypeGetDimSize(type, i - 1)
Base.size(type::MType) = Tuple(size(type, i) for i in 1:ndims(type))

function is_tensor(type::MType)
    LibMLIR.mlirTypeIsAShaped(type)
end

function is_integer(type::MType)
    LibMLIR.mlirTypeIsAInteger(type)
end

### Attribute

struct Attribute
    attribute::MlirAttribute
end

Attribute() = Attribute(LibMLIR.mlirAttributeGetNull())
Attribute(context, s::AbstractString) = Attribute(LibMLIR.mlirStringAttrGet(context, s))
Attribute(type::MType) = Attribute(LibMLIR.mlirTypeAttrGet(type))
Attribute(context, f::Float32) = Attribute(
    LibMLIR.mlirFloatAttrDoubleGet(context, MType(context, Float32), Float64(f))
)
Attribute(context, f::Float64) = Attribute(
    LibMLIR.mlirFloatAttrDoubleGet(context, MType(context, Float64), f)
)
Attribute(context, i::T) where {T<:Integer} = Attribute(
    LibMLIR.mlirIntegerAttrGet(MType(context, T), Int64(i))
)
function Attribute(context, values::T) where {T<:AbstractArray{Int32}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Int64}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float64}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrDoubleGet(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float32}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int32}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Float32}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function ArrayAttribute(context, values::AbstractVector{Int})
    elements = Attribute.((context,), values)
    Attribute(
        LibMLIR.mlirArrayAttrGet(context, length(elements), elements)
    )
end

Base.convert(::Type{MlirAttribute}, attribute::Attribute) = attribute.attribute
Base.parse(::Type{Attribute}, context, s) =
    Attribute(LibMLIR.mlirAttributeParseGet(context, s))

### Named Attribute

struct NamedAttribute
    named_attribute::MlirNamedAttribute
end

NamedAttribute(context, name, attribute) =
    NamedAttribute(LibMLIR.mlirNamedAttributeGet(
        LibMLIR.mlirIdentifierGet(context, name),
        attribute
    ))

Base.convert(::Type{MlirAttribute}, named_attribute::NamedAttribute) =
    named_attribute.named_attribute

### Value

struct Value
    value::MlirValue

    Value(value) = begin
        @assert !LibMLIR.mlirValueIsNull(value) "cannot create Value with null MlirValue"
        new(value)
    end
end

get_type(value) = MType(LibMLIR.mlirValueGetType(value))

Base.convert(::Type{MlirValue}, value::Value) = value.value
Base.size(value::Value) = Base.size(get_type(value))
Base.ndims(value::Value) = Base.ndims(get_type(value))

function Base.show(io::IO, value::Value)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirValuePrint(value, c_print_callback, ref)
end

### OperationState

struct OperationState
    opstate::MlirOperationState
end

OperationState(name, location) = OperationState(LibMLIR.mlirOperationStateGet(name, location))

add_results!(state, results) =
    LibMLIR.mlirOperationStateAddResults(state, length(results), results)
add_operands!(state, operands) =
    LibMLIR.mlirOperationStateAddOperands(state, length(operands), operands)
function add_owned_regions!(state, regions)
    mlir_regions = Base.convert.(MlirRegion, regions)
    lose_ownership!.(regions)
    LibMLIR.mlirOperationStateAddOwnedRegions(state, length(mlir_regions), mlir_regions)
end
add_attributes!(state, attributes) =
    LibMLIR.mlirOperationStateAddAttributes(state, length(attributes), attributes)

enable_type_inference!(state) =
    LibMLIR.mlirOperationStateEnableResultTypeInference(state)

Base.unsafe_convert(::Type{Ptr{MlirOperationState}}, state::OperationState) =
    Base.unsafe_convert(Ptr{MlirOperationState}, Base.pointer_from_objref(state.opstate))

### Operation

mutable struct Operation
    operation::MlirOperation
    @atomic owned::Bool

    Operation(operation, owned=true) = begin
        @assert !LibMLIR.mlirOperationIsNull(operation) "cannot create Operation with null MlirOperation"
        finalizer(new(operation, owned)) do op
            if op.owned
                LibMLIR.mlirOperationDestroy(op.operation)
            end
        end
    end
end

Operation(state::OperationState) = Operation(LibMLIR.mlirOperationCreate(state), true)

copy(operation) = Operation(LibMLIR.mlirOperationClone(operation))

num_regions(operation) = LibMLIR.mlirOperationGetNumRegions(operation)
function get_region(operation, i)
    i ∈ 1:num_regions(operation) && throw(BoundsError(operation, i))
    Region(LibMLIR.mlirOperationGetRegion(operation, i - 1), false)
end
num_results(operation) = LibMLIR.mlirOperationGetNumResults(operation)
get_results(operation) = [
    get_result(operation, i)
    for i in 1:num_results(operation)
]
function get_result(operation, i)
    i ∉ 1:num_results(operation) && throw(BoundsError(operation, i))
    Value(LibMLIR.mlirOperationGetResult(operation, i - 1))
end
get_location(operation) = Location(LibMLIR.mlirOperationGetLocation(operation))
get_name(operation) = String(LibMLIR.mlirOperationGetName(operation))
get_block(operation) = Block(LibMLIR.mlirOperationGetBlock(operation), false)
get_parent_operation(operation) = Operation(LibMLIR.mlirOperationGetParentOperation(operation), false)
get_dialect(operation) = first(split(get_name(operation), '.')) |> Symbol

op::Operation == other::Operation = LibMLIR.mlirOperationEqual(op, other)

Base.convert(::Type{MlirOperation}, op::Operation) = op.operation

function lose_ownership!(operation::Operation)
    @assert operation.owned
    @atomic operation.owned = false
    operation
end

function Base.show(io::IO, operation::Operation)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    flags = LibMLIR.mlirOpPrintingFlagsCreate()
    get(io, :debug, false) && LibMLIR.mlirOpPrintingFlagsEnableDebugInfo(flags, true, true)
    GC.@preserve ref LibMLIR.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    println(io)
end

### Block

mutable struct Block
    block::MlirBlock
    @atomic owned::Bool

    Block(block, owned=true) = begin
        @assert !LibMLIR.mlirBlockIsNull(block) "cannot create Block with null MlirBlock"
        finalizer(new(block, owned)) do block
            if block.owned
                LibMLIR.mlirBlockDestroy(block.block)
            end
        end
    end
end

function Block(args::Vector{MType}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    Block(LibMLIR.mlirBlockCreate(length(args), args, locs))
end

function Base.push!(block::Block, op::Operation)
    LibMLIR.mlirBlockAppendOwnedOperation(block, lose_ownership!(op))
    op
end
function Base.insert!(block::Block, pos, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperation(block, pos, lose_ownership!(op))
    op
end
function insert_after!(block::Block, reference::Operation, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperationAfter(block, reference, lose_ownership!(op))
    op
end
function insert_before!(block::Block, reference::Operation, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperationBefore(block, reference, lose_ownership!(op))
    op
end

num_arguments(block::Block) =
    LibMLIR.mlirBlockGetNumArguments(block)
function get_argument(block::Block, i)
    i ∉ 1:num_arguments(block) && throw(BoundsError(block, i))
    Value(LibMLIR.mlirBlockGetArgument(block, i - 1))
end
add_argument!(block::Block, type, loc) =
    Value(LibMLIR.mlirBlockAddArgument(block, type, loc))

Base.convert(::Type{MlirBlock}, block::Block) = block.block

function lose_ownership!(block::Block)
    @assert block.owned
    @atomic block.owned = false
    block
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirBlockPrint(block, c_print_callback, ref)
end

### Region

mutable struct Region
    region::MlirRegion
    @atomic owned::Bool # TODO: make atomic?

    Region(region, owned=true) = begin
        @assert !LibMLIR.mlirRegionIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                LibMLIR.mlirRegionDestroy(region.region)
            end
        end
    end
end

Region() = Region(LibMLIR.mlirRegionCreate())

function Base.push!(region::Region, block::Block)
    LibMLIR.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    block
end
Base.insert!(region::Region, pos, block::Block) =
    LibMLIR.mlirRegionInsertOwnedBlock(region, pos, lose_ownership!(block))
insert_after!(region::Region, reference::Block, block::Block) =
    LibMLIR.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))
insert_before!(region::Region, reference::Block, block::Block) =
    LibMLIR.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))

get_first_block(region::Region) = Block(LibMLIR.mlirRegionGetFirstBlock(region), false)

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    region
end

Base.convert(::Type{MlirRegion}, region::Region) = region.region

### Module

mutable struct MModule
    module_::MlirModule
    context::Context

    MModule(module_, context) = begin
        @assert !LibMLIR.mlirModuleIsNull(module_) "cannot create MModule with null MlirModule"
        finalizer(LibMLIR.mlirModuleDestroy, new(module_, context))
    end
end

MModule(context::Context, loc=Location(context)) =
    MModule(LibMLIR.mlirModuleCreateEmpty(loc), context)
get_operation(module_) = Operation(LibMLIR.mlirModuleGetOperation(module_), false)
get_body(module_) = Block(LibMLIR.mlirModuleGetBody(module_), false)

Base.convert(::Type{MlirModule}, module_::MModule) = module_.module_
Base.parse(::Type{MModule}, context, module_) = MModule(LibMLIR.mlirModuleCreateParse(context, module_), context)

function Base.show(io::IO, module_::MModule)
    println(io, "MModule:")
    show(io, get_operation(module_))
end

### Pass Manager

mutable struct PassManager
    pass::MlirPassManager
    context::Context

    PassManager(pass::MlirPassManager, context) = begin
        @assert !LibMLIR.mlirPassManagerIsNull(pass) "cannot create PassManager with null MlirPassManager"
        finalizer(new(pass, context)) do pass
            LibMLIR.mlirPassManagerDestroy(pass.pass)
        end
    end
end

function enable_verifier!(pass)
    LibMLIR.mlirPassManagerEnableVerifier(pass)
    pass
end

PassManager(context) =
    PassManager(LibMLIR.mlirPassManagerCreate(context), context)

function run(pass, module_)
    status = LibMLIR.mlirPassManagerRun(pass, module_)
    if LibMLIR.mlirLogicalResultIsFailure(status)
        throw("failed to run pass manager on module")
    end
    return nothing
end

Base.convert(::Type{MlirPassManager}, pass::PassManager) = pass.pass

### Op Pass Manager

struct OpPassManager
    op_pass::MlirOpPassManager
    pass::PassManager

    OpPassManager(op_pass, pass) = begin
        @assert !LibMLIR.mlirOpPassManagerIsNull(op_pass) "cannot create OpPassManager with null MlirOpPassManager"
        new(op_pass, pass)
    end
end

OpPassManager(pass::PassManager) = OpPassManager(LibMLIR.mlirPassManagerGetAsOpPassManager(pass), pass)

Base.convert(::Type{MlirOpPassManager}, op_pass::OpPassManager) = op_pass.op_pass

function Base.show(io::IO, op_pass::OpPassManager)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    println(io, "OpPassManager(\"\"\"")
    GC.@preserve ref LibMLIR.mlirPrintPassPipeline(op_pass, c_print_callback, ref)
    println(io)
    print(io, "\"\"\")")
end

function add_pipeline!(op_pass::OpPassManager, pipeline)
    io = IOBuffer()
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    result = GC.@preserve io LibMLIR.mlirOpPassManagerAddPipeline(op_pass, pipeline, c_print_callback, io)
    if LibMLIR.mlirLogicalResultIsFailure(result)
        msg = String(take!(io))
        throw("failed to add pipeline: $msg")
    end
    op_pass
end

### Iterators

"""
    BlockIterator(region::Region)

Iterates over all blocks in the given region.
"""
struct BlockIterator
    region::Region
end

function Base.iterate(it::BlockIterator)
    reg = it.region
    raw_block = LibMLIR.mlirRegionGetFirstBlock(reg)
    if LibMLIR.mlirBlockIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(it::BlockIterator, block)
    raw_block = LibMLIR.mlirBlockGetNextInRegion(block)
    if LibMLIR.mlirBlockIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

"""
    OperationIterator(block::Block)

Iterates over all operations for the given block.
"""
struct OperationIterator
    block::Block
end

function Base.iterate(it::OperationIterator)
    raw_op = LibMLIR.mlirBlockGetFirstOperation(it.block)
    if LibMLIR.mlirOperationIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(it::OperationIterator, op)
    raw_op = LibMLIR.mlirOperationGetNextInBlock(op)
    if LibMLIR.mlirOperationIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

"""
    RegionIterator(::Operation)

Iterates over all sub-regions for the given operation.
"""
struct RegionIterator
    op::Operation
end

function Base.iterate(it::RegionIterator)
    raw_region = LibMLIR.mlirOperationGetFirstRegion(it.op)
    if LibMLIR.mlirRegionIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = LibMLIR.mlirRegionGetNextInOperation(region)
    if LibMLIR.mlirRegionIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

### Utils

function get_dialects!(dialects::Set{Symbol}, op::Operation)
    push!(dialects, get_dialect(op))

    for region in RegionIterator(op)
        for block in BlockIterator(region)
            for op in OperationIterator(block)
                get_dialects!(dialects, op)
            end
        end
    end

    dialects
end

function get_input_type(module_)
    dialects = Set{Symbol}()

    op = get_operation(module_)
    get_dialects!(dialects, op)

    if :mhlo ∈ dialects
        # :tosa ∉ dialects || throw("cannot have both tosa and mhlo operations")
        :mhlo
    elseif :tosa ∈ dialects
        :tosa
    else
        :none
    end
end

end # module MLIR
