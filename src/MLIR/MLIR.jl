module MLIR

include("./LibMLIR.jl")

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
Location(context::Context, filename, line, column) =
    Location(LibMLIR.mlirLocationmlirLocationFileLineColGet(context, filename, line, column))

Base.convert(::Type{MlirLocation}, location::Location) = location.location

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
MType(context::Context, ::Type{Float32}) =
    MType(LibMLIR.mlirF32TypeGet(context))
MType(context::Context, ::Type{Float64}) =
    MType(LibMLIR.mlirF64TypeGet(context))
MType(context::Context, ft::Pair) =
    MType(LibMLIR.mlirFunctionTypeGet(context,
        length(ft.first), [MType(t) for t in ft.first],
        length(ft.second), [MType(t) for t in ft.second]))
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

Base.convert(::Type{MlirType}, mtype::MType) = mtype.type

function Base.eltype(type::MType)
    if LibMLIR.mlirTypeIsAShaped(type)
        MType(LibMLIR.mlirShapedTypeGetElementType(type))
    else
        type
    end
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
function Attribute(context, values::T) where {T<:AbstractArray{Int64}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float32}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrFloatGet(type, length(values), values)
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

num_results(operation) = LibMLIR.mlirOperationGetNumResults(operation)
function get_result(operation, i)
    i ∉ 1:num_results(operation) && throw(BoundsError(operation, i))
    Value(LibMLIR.mlirOperationGetResult(operation, i - 1))
end
get_location(operation) = Location(LibMLIR.mlirOperationGetLocation(operation))
get_name(operation) = String(LibMLIR.mlirOperationGetName(operation))
get_block(operation) = Block(LibMLIR.mlirOperationGetBlock(operation), false)
get_parent_operation(operation) = Operation(LibMLIR.mlirOperationGetParentOperation(operation), false)

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
    GC.@preserve ref LibMLIR.mlirOperationPrint(operation, c_print_callback, ref)
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

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    region
end

Base.convert(::Type{MlirRegion}, region::Region) = region.region

### Module

mutable struct MModule
    module_::MlirModule

    MModule(module_) = begin
        @assert !LibMLIR.mlirModuleIsNull(module_) "cannot create MModule with null MlirModule"
        finalizer(LibMLIR.mlirModuleDestroy, new(module_))
    end
end

MModule(location::Location) = MModule(LibMLIR.mlirModuleCreateEmpty(location))
get_operation(module_) = Operation(LibMLIR.mlirModuleGetOperation(module_), false)
get_body(module_) = Block(LibMLIR.mlirModuleGetBody(module_), false)

Base.convert(::Type{MlirModule}, module_::MModule) = module_.module_

function Base.show(io::IO, module_::MModule)
    println(io, "MModule:")
    show(io, get_operation(module_))
end

### Pass Manager

mutable struct PassManager
    pass::MlirPassManager

    PassManager(pass::MlirPassManager) = begin
        @assert !LibMLIR.mlirPassManagerIsNull(pass) "cannot create PassManager with null MlirPassManager"
        finalizer(new(pass)) do pass
            LibMLIR.mlirPassManagerDestroy(pass.pass)
        end
    end
end

PassManager(context) = PassManager(LibMLIR.mlirPassManagerCreate(context))
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

end # module MLIR
