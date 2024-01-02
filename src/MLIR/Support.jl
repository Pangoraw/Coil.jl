function mlirIsNull(val)
   val.ptr == C_NULL 
end

###

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Type{API.MlirStringRef}, s::Union{Symbol, String}) = s
Base.cconvert(::Type{API.MlirStringRef}, s::AbstractString) =
    Base.cconvert(API.MlirStringRef, String(s)::String)

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(::Type{API.MlirStringRef}, s::Union{Symbol, String, AbstractVector{UInt8}})
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return API.MlirStringRef(p, sizeof(s))
end


### Identifier

String(s::MlirStringRef) = unsafe_string(s.data, s.length)
String(ident::MlirIdentifier) = String(API.mlirIdentifierStr(ident))

### Logical Result

mlirLogicalResultSuccess() = API.MlirLogicalResult(1)
mlirLogicalResultFailure() = API.MlirLogicalResult(0)

mlirLogicalResultIsSuccess(result) = result.value != 0
mlirLogicalResultIsFailure(result) = result.value == 0

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
    raw_block = API.mlirRegionGetFirstBlock(reg)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(it::BlockIterator, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
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
    raw_op = API.mlirBlockGetFirstOperation(it.block)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(it::OperationIterator, op)
    raw_op = API.mlirOperationGetNextInBlock(op)
    if mlirIsNull(raw_op)
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
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

### Utils

function visit(f, op)
    for region in RegionIterator(op)
        for block in BlockIterator(region)
            for op in OperationIterator(block)
                f(op)
            end
        end
    end
end

"""
    verifyall(operation; debug=false)

Prints the operations which could not be verified.
"""
function verifyall(operation::Operation; debug=false)
    io = IOContext(stdout, :debug => debug)
    visit(operation) do op
        if !verify(op)
            show(io, op)
        end
    end
end
verifyall(module_::MModule) = get_operation(module_) |> verifyall

### uses

struct UsesIterator
    value::Value
end
uses(value) = UsesIterator(value)

function n_uses(value)
    n = 0
    for _ in uses(value) 
        n += 1
    end
    n
end

function Base.iterate(it::UsesIterator)
    op_operand = first_use(it.value)
    if isnothing(op_operand)
        return nothing
    end

    return (
        (owner(op_operand), operand_number(op_operand)),
        op_operand,
    )
end

function Base.iterate(::UsesIterator, op_operand)
    next_op_operand = next_use(op_operand)
    if isnothing(next_op_operand)
        return nothing
    end

    return (
        (owner(next_op_operand), operand_number(next_op_operand)),
        next_op_operand,
    )
end
