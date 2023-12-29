module LibMLIR

import Artifacts
import ...Coil: @cvcall

const libmlir = joinpath(Artifacts.artifact"libIREECompiler", "lib/libIREECompiler.so")
if !isfile(libmlir)
    error("🔴🔴🔴 '$libmlir' not found, try changing its definition at $(@__FILE__):$(@__LINE__() - 2)")
end

# using MLIR_jll: libMLIR
# const libmlir = libMLIR

### Types

# storage = void*
for T in (:MlirContext,
    :MlirDialect,
    :MlirDialectRegistry,
    :MlirOperation,
    :MlirOpOperand,
    :MlirOpPrintingFlags,
    :MlirBlock,
    :MlirRegion,
    :MlirSymbolTable,
    :MlirPass,
    :MlirExternalPass,
    :MlirPassManager,
    :MlirOpPassManager,
    :MlirTypeIDAllocator,
    :MlirAffineMap,
    :MlirAffineExpr,)
    @eval struct $T
        ptr::Ptr{Cvoid}
    end
end

# storage = const void*
for T in (:MlirAttribute,
    :MlirIdentifier,
    :MlirLocation,
    :MlirModule,
    :MlirType,
    :MlirValue,
    :MlirTypeID,
    :MlirDialectHandle,)
    @eval struct $T
        ptr::Ptr{Cvoid}
    end
end

struct MlirLogicalResult
    value::Int8
end

struct MlirStringRef
    data::Ptr{Cchar}
    length::Csize_t
end

struct MlirNamedAttribute
    name::MlirIdentifier
    attribute::MlirAttribute
end

const intptr_t = Clong

mlirIsNull(val) = val.ptr == C_NULL

### Logical Result

mlirLogicalResultIsSuccess(result) = result.value != 0
mlirLogicalResultIsFailure(result) = result.value == 0
mlirLogicalResultSuccess() = MlirLogicalResult(one(Int8))
mlirLogicalResultFailure() = MlirLogicalResult(zero(Int8))

### Dialect Registry

mlirDialectRegistryCreate() = @cvcall libmlir.mlirDialectRegistryCreate()::MlirDialectRegistry
mlirDialectRegistryDestroy(registry) = @cvcall libmlir.mlirDialectRegistryDestroy(registry::MlirDialectRegistry)::Cvoid
mlirRegisterAllDialects(registry) = @cvcall libmlir.mlirRegisterAllDialects(registry::MlirDialectRegistry)::Cvoid

### Context

mlirContextCreate() = @cvcall libmlir.mlirContextCreate()::MlirContext
mlirContextDestroy(context) = @cvcall libmlir.mlirContextDestroy(context::MlirContext)::Cvoid
mlirContextGetNumLoadedDialects(context) = @cvcall libmlir.mlirContextGetNumLoadedDialects(context::MlirContext)::Cint
mlirContextGetOrLoadDialect(context, dialect) = @cvcall libmlir.mlirContextGetOrLoadDialect(context::MlirContext, dialect::MlirStringRef)::MlirDialect
mlirContextLoadAllAvailableDialects(context) = @cvcall libmlir.mlirContextLoadAllAvailableDialects(context::MlirContext)::Cvoid
mlirContextIsRegisteredOperation(context, op) = @cvcall libmlir.mlirContextIsRegisteredOperation(context::MlirContext, op::MlirStringRef)::Bool
mlirContextIsNull(context) = mlirIsNull(context)
mlirContextAppendDialectRegistry(context, registry) = @cvcall libmlir.mlirContextAppendDialectRegistry(context::MlirContext, registry::MlirDialectRegistry)::Cvoid

### Dialect

mlirDialectIsNull(dialect) = mlirIsNull(dialect)
mlirDialectGetNamespace(dialect) = @cvcall libmlir.mlirDialectGetNamespace(dialect::MlirDialect)::MlirStringRef
mlirDialectGetContext(dialect) = @cvcall libmlir.mlirDialectGetContext(dialect::MlirDialect)::MlirContext

### Location

mlirLocationIsNull(location) = mlirIsNull(location)
mlirLocationPrint(location, callback, userdata) =
    @cvcall libmlir.mlirLocationPrint(location::MlirLocation, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirLocationUnknownGet(context) = @cvcall libmlir.mlirLocationUnknownGet(context::MlirContext)::MlirLocation
mlirLocationFileLineColGet(context, filename, line, column) =
    @cvcall libmlir.mlirLocationFileLineColGet(
        context::MlirContext,
        filename::MlirStringRef,
        line::Cuint,
        column::Cuint,
    )::MlirLocation

### Type

mlirTypeDump(type) = @cvcall libmlir.mlirTypeDump(type::MlirType)::Cvoid
mlirTypePrint(type, callback, userdata) = @cvcall libmlir.mlirTypePrint(type::MlirType, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirTypeIsNull(type) = mlirIsNull(type)
mlirIndexTypeGet(context) = @cvcall libmlir.mlirIndexTypeGet(context::MlirContext)::MlirType
mlirIntegerTypeGet(context, size) = @cvcall libmlir.mlirIntegerTypeGet(context::MlirContext, size::Cuint)::MlirType
mlirIntegerTypeSignedGet(context, size) =
    @cvcall libmlir.mlirIntegerTypeSignedGet(context::MlirContext, size::Cuint)::MlirType
mlirF32TypeGet(context) = @cvcall libmlir.mlirF32TypeGet(context::MlirContext)::MlirType
mlirF64TypeGet(context) = @cvcall libmlir.mlirF64TypeGet(context::MlirContext)::MlirType
mlirFunctionTypeGet(context, nargs, args, nresults, results) =
    @cvcall libmlir.mlirFunctionTypeGet(context::MlirContext, nargs::intptr_t, args::Ptr{MlirType}, nresults::intptr_t, results::Ptr{MlirType})::MlirType
mlirFunctionTypeGetNumResults(ftype) = @cvcall libmlir.mlirFunctionTypeGetNumResults(ftype::MlirType)::intptr_t
mlirFunctionTypeGetNumInputs(ftype) = @cvcall libmlir.mlirFunctionTypeGetNumInputs(ftype::MlirType)::intptr_t
mlirFunctionTypeGetResult(ftype, pos) = @cvcall libmlir.mlirFunctionTypeGetResult(ftype::MlirType, pos::intptr_t)::MlirType
mlirFunctionTypeGetInput(ftype, pos) = @cvcall libmlir.mlirFunctionTypeGetInput(ftype::MlirType, pos::intptr_t)::MlirType
mlirRankedTensorTypeGetChecked(location, rank, shape, eltype, encoding) =
    @cvcall libmlir.mlirRankedTensorTypeGetChecked(
        location::MlirLocation,
        rank::intptr_t,
        shape::Ptr{Clong},
        eltype::MlirType,
        encoding::MlirAttribute,
    )::MlirType
mlirShapedTypeGetDimSize(type, i) =
    @cvcall libmlir.mlirShapedTypeGetDimSize(type::MlirType, i::intptr_t)::Int64
mlirShapedTypeHasRank(type) =
    @cvcall libmlir.mlirShapedTypeHasRank(type::MlirType)::Bool
mlirShapedTypeGetRank(type) =
    @cvcall libmlir.mlirShapedTypeGetRank(type::MlirType)::Int64
mlirShapedTypeGetElementType(type) =
    @cvcall libmlir.mlirShapedTypeGetElementType(type::MlirType)::MlirType
mlirTypeIsARankedTensor(type) =
    @cvcall libmlir.mlirTypeIsARankedTensor(type::MlirType)::Bool
mlirTypeIsAShaped(type) =
    @cvcall libmlir.mlirTypeIsAShaped(type::MlirType)::Bool
mlirIntegerTypeGetWidth(type) =
    @cvcall libmlir.mlirIntegerTypeGetWidth(type::MlirType)::UInt32
mlirIntegerTypeIsSignless(type) =
    @cvcall libmlir.mlirIntegerTypeIsSignless(type::MlirType)::Bool
mlirIntegerTypeIsSigned(type) =
    @cvcall libmlir.mlirIntegerTypeIsSigned(type::MlirType)::Bool
mlirTypeIsAInteger(type) =
    @cvcall libmlir.mlirTypeIsAInteger(type::MlirType)::Bool
mlirTypeIsAF32(type) =
    @cvcall libmlir.mlirTypeIsAF32(type::MlirType)::Bool
mlirTypeIsAF64(type) =
    @cvcall libmlir.mlirTypeIsAF64(type::MlirType)::Bool
mlirTypeIsAIndex(type) =
    @cvcall libmlir.mlirTypeIsAIndex(type::MlirType)::Bool
mlirTypeIsAFunction(type) =
    @cvcall libmlir.mlirTypeIsAFunction(type::MlirType)::Bool

### Attributes

mlirAttributeIsNull(attr) = mlirIsNull(attr)
mlirAttributeGetNull() = @cvcall libmlir.mlirAttributeGetNull()::MlirAttribute
mlirStringAttrGet(context, str) = @cvcall libmlir.mlirStringAttrGet(context::MlirContext, str::MlirStringRef)::MlirAttribute
mlirAttributeIsAString(attr) = @cvcall libmlir.mlirAttributeIsAString(attr::MlirAttribute)::Bool
mlirStringAttrGetValue(attr) = @cvcall libmlir.mlirStringAttrGetValue(attr::MlirAttribute)::MlirStringRef
mlirTypeAttrGet(type) = @cvcall libmlir.mlirTypeAttrGet(type::MlirType)::MlirAttribute
mlirTypeAttrGetValue(attr) = @cvcall libmlir.mlirTypeAttrGetValue(attr::MlirAttribute)::MlirType
mlirAttributeGetContext(attribute) = @cvcall libmlir.mlirAttributeGetContext(attribute::MlirAttribute)::MlirContext
mlirAttributeParseGet(context, str) = @cvcall libmlir.mlirAttributeParseGet(context::MlirContext, str::MlirStringRef)::MlirAttribute
mlirAttributePrint(attribute, callback, userdata) = @cvcall libmlir.mlirAttributePrint(attribute::MlirAttribute, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirAttributeGetType(attribute) = @cvcall libmlir.mlirAttributeGetType(attribute::MlirAttribute)::MlirType
mlirDenseElementsAttrFloatGet(type, nfloats, floats) =
    @cvcall libmlir.mlirDenseElementsAttrFloatGet(type::MlirType, nfloats::intptr_t, floats::Ptr{Cfloat})::MlirAttribute
mlirDenseElementsAttrDoubleGet(type, ndoubles, doubles) =
    @cvcall libmlir.mlirDenseElementsAttrDoubleGet(type::MlirType, ndoubles::intptr_t, doubles::Ptr{Cdouble})::MlirAttribute
mlirDenseElementsAttrInt32Get(type, nints, ints) =
    @cvcall libmlir.mlirDenseElementsAttrInt32Get(type::MlirType, nints::intptr_t, ints::Ptr{Int32})::MlirAttribute
mlirDenseElementsAttrInt64Get(type, nints, ints) =
    @cvcall libmlir.mlirDenseElementsAttrInt64Get(type::MlirType, nints::intptr_t, ints::Ptr{Int64})::MlirAttribute
mlirDenseI32ArrayGet(context, size, values) =
    @cvcall libmlir.mlirDenseI32ArrayGet(context::MlirContext, size::intptr_t, values::Ptr{Int32})::MlirAttribute
mlirDenseI64ArrayGet(context, size, values) =
    @cvcall libmlir.mlirDenseI64ArrayGet(context::MlirContext, size::intptr_t, values::Ptr{Int64})::MlirAttribute
mlirFloatAttrDoubleGet(context, type, f) =
    @cvcall libmlir.mlirFloatAttrDoubleGet(context::MlirContext, type::MlirType, f::Cdouble)::MlirAttribute
mlirIntegerAttrGet(type, i) =
    @cvcall libmlir.mlirIntegerAttrGet(type::MlirType, i::Int64)::MlirAttribute
mlirIntegerAttrGetValueInt(attribute) =
    @cvcall libmlir.mlirIntegerAttrGetValueInt(attribute::MlirAttribute)::Clong
mlirAttributeIsAInteger(attribute) =
    @cvcall libmlir.mlirAttributeIsAInteger(attribute::MlirAttribute)::Bool
mlirAttributeIsAType(attribute) =
    @cvcall libmlir.mlirAttributeIsAType(attribute::MlirAttribute)::Bool
mlirAttributeIsAFloat(attribute) =
    @cvcall libmlir.mlirAttributeIsAFloat(attribute::MlirAttribute)::Bool
mlirArrayAttrGet(context, nelements, elements) =
    @cvcall libmlir.mlirArrayAttrGet(context::MlirContext, nelements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute
mlirBoolAttrGet(context, value) = @cvcall libmlir.mlirBoolAttrGet(context::MlirContext, value::Cint)::MlirAttribute
mlirAttributeIsABool(attribute) = @cvcall libmlir.mlirAttributeIsABool(attribute::MlirAttribute)::Bool
mlirBoolAttrGetValue(attribute) = @cvcall libmlir.mlirBoolAttrGetValue(attribute::MlirAttribute)::Bool
mlirAffineMapAttrGet(map) = @cvcall libmlir.mlirAffineMapAttrGet(map::MlirAffineMap)::MlirAttribute

### Affine

mlirAffineMulExprGet(a, b) = @cvcall libmlir.mlirAffineMulExprGet(a::MlirAffineExpr, b::MlirAffineExpr)::MlirAffineExpr
mlirAffineAddExprGet(a, b) = @cvcall libmlir.mlirAffineAddExprGet(a::MlirAffineExpr, b::MlirAffineExpr)::MlirAffineExpr
mlirAffineModExprGet(a, b) = @cvcall libmlir.mlirAffineModExprGet(a::MlirAffineExpr, b::MlirAffineExpr)::MlirAffineExpr
mlirAffineSymbolExprGet(context, position) = @cvcall libmlir.mlirAffineSymbolExprGet(context::MlirContext, position::intptr_t)::MlirAffineExpr
mlirAffineDimExprGet(context, position) = @cvcall libmlir.mlirAffineDimExprGet(context::MlirContext, position::intptr_t)::MlirAffineExpr
mlirAffineConstantExprGet(context, val) = @cvcall libmlir.mlirAffineConstantExprGet(context::MlirContext, val::Int64)::MlirAffineExpr

mlirAffineMapDump(map) = @cvcall libmlir.mlirAffineMapDump(map::MlirAffineMap)::Cvoid
mlirAffineMapGet(context, dimCount, symbolCount, nAffineExprs, exprs) =
    @cvcall libmlir.mlirAffineMapGet(context::MlirContext, dimCount::intptr_t, symbolCount::intptr_t, nAffineExprs::intptr_t, exprs::Ptr{MlirAffineExpr})::MlirAffineMap
mlirAffineMapMultiDimIdentityGet(context, ndims) =
    @cvcall libmlir.mlirAffineMapMultiDimIdentityGet(context::MlirContext, ndims::intptr_t)::MlirAffineMap

### Identifier

mlirIdentifierGet(context, str) = @cvcall libmlir.mlirIdentifierGet(context::MlirContext, str::MlirStringRef)::MlirIdentifier
mlirIdentifierStr(ident) = @cvcall libmlir.mlirIdentifierStr(ident::MlirIdentifier)::MlirStringRef

### Named Attributes

mlirNamedAttributeGet(name, attr) =
    @cvcall libmlir.mlirNamedAttributeGet(name::MlirIdentifier, attr::MlirAttribute)::MlirNamedAttribute

### Region

mlirRegionCreate() = @cvcall libmlir.mlirRegionCreate()::MlirRegion
mlirRegionDestroy(region) = @cvcall libmlir.mlirRegionDestroy(region::MlirRegion)::Cvoid
mlirRegionIsNull(region) = mlirIsNull(region)
mlirRegionEqual(region, other) = @cvcall libmlir.mlirRegionEqual(region::MlirRegion, other::MlirRegion)::Bool
mlirRegionAppendOwnedBlock(region, block) = @cvcall libmlir.mlirRegionAppendOwnedBlock(region::MlirRegion, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlock(region, pos, block) =
    @cvcall libmlir.mlirRegionInsertOwnedBlock(region::MlirRegion, pos::intptr_t, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlockAfter(region, reference, block) =
    @cvcall libmlir.mlirRegionInsertOwnedBlockAfter(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlockBefore(region, reference, block) =
    @cvcall libmlir.mlirRegionInsertOwnedBlockBefore(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
mlirRegionGetFirstBlock(region) = @cvcall libmlir.mlirRegionGetFirstBlock(region::MlirRegion)::MlirBlock
mlirRegionGetNextInOperation(region) = @cvcall libmlir.mlirRegionGetNextInOperation(region::MlirRegion)::MlirRegion

### Block

mlirBlockCreate(nargs, args, locs) = @cvcall libmlir.mlirBlockCreate(nargs::intptr_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation})::MlirBlock
mlirBlockDestroy(block) = @cvcall libmlir.mlirBlockDestroy(block::MlirBlock)::Cvoid
mlirBlockIsNull(block) = mlirIsNull(block)
mlirBlockDetach(block) = @cvcall libmlir.mlirBlockDetach(block::MlirBlock)::Cvoid
mlirBlockAppendOwnedOperation(block, operation) =
    @cvcall libmlir.mlirBlockAppendOwnedOperation(block::MlirBlock, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperation(block, pos, operation) =
    @cvcall libmlir.mlirBlockInsertOwnedOperation(block::MlirBlock, pos::intptr_t, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperationAfter(block, reference, operation) =
    @cvcall libmlir.mlirBlockInsertOwnedOperationAfter(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperationBefore(block, reference, operation) =
    @cvcall libmlir.mlirBlockInsertOwnedOperationBefore(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
mlirBlockGetArgument(block, i) = @cvcall libmlir.mlirBlockGetArgument(block::MlirBlock, i::intptr_t)::MlirValue
mlirBlockGetNumArguments(block) = @cvcall libmlir.mlirBlockGetNumArguments(block::MlirBlock)::intptr_t
mlirBlockAddArgument(block, type, loc) =
    @cvcall libmlir.mlirBlockAddArgument(block::MlirBlock, type::MlirType, loc::MlirLocation)::MlirValue
mlirBlockPrint(block, callback, userdata) =
    @cvcall libmlir.mlirBlockAddArgument(block::MlirBlock, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirBlockGetNextInRegion(block) = @cvcall libmlir.mlirBlockGetNextInRegion(block::MlirBlock)::MlirBlock
mlirBlockGetFirstOperation(block) = @cvcall libmlir.mlirBlockGetFirstOperation(block::MlirBlock)::MlirOperation
mlirBlockGetParentRegion(block) = @cvcall libmlir.mlirBlockGetParentRegion(block::MlirBlock)::MlirRegion

### Value

mlirValueGetType(value) = @cvcall libmlir.mlirValueGetType(value::MlirValue)::MlirType
mlirValueIsNull(value) = mlirIsNull(value)
mlirValueDump(value) = @cvcall libmlir.mlirValueDump(value::MlirValue)::Cvoid
mlirValuePrint(value, callback, userdata) = @cvcall libmlir.mlirValuePrint(value::MlirValue, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirValueGetFirstUse(value) = @cvcall libmlir.mlirValueGetFirstUse(value::MlirValue)::MlirOpOperand
mlirValueIsAOpResult(value) = @cvcall libmlir.mlirValueIsAOpResult(value::MlirValue)::Bool
mlirOpResultGetOwner(value) = @cvcall libmlir.mlirOpResultGetOwner(value::MlirValue)::MlirOperation
mlirValueIsABlockArgument(value) = @cvcall libmlir.mlirValueIsABlockArgument(value::MlirValue)::Bool
mlirBlockArgumentGetOwner(value) = @cvcall libmlir.mlirBlockArgumentGetOwner(value::MlirValue)::MlirBlock
mlirBlockArgumentSetType(value, type) = @cvcall libmlir.mlirBlockArgumentSetType(value::MlirValue, type::MlirType)::Cvoid

### Op Operand

mlirOpOperandIsNull(op_operand) = mlirIsNull(op_operand)
mlirOpOperandGetOwner(op_operand) = @cvcall libmlir.mlirOpOperandGetOwner(op_operand::MlirOpOperand)::MlirOperation
mlirOpOperandGetOperandNumber(op_operand) = @cvcall libmlir.mlirOpOperandGetOperandNumber(op_operand::MlirOpOperand)::Cuint
mlirOpOperandGetNextUse(op_operand) = @cvcall libmlir.mlirOpOperandGetNextUse(op_operand::MlirOpOperand)::MlirOpOperand

### OperationState

"""
    OperationState(name::AbstractString, loc::Location)

An auxiliary class for constructing operations.

This class contains all the information necessary to construct the
operation. It owns the MlirRegions it has pointers to and does not own
anything else. By default, the state can be constructed from a name and
location, the latter being also used to access the context, and has no other
components. These components can be added progressively until the operation
is constructed. Users are not expected to rely on the internals of this
class and should use mlirOperationState* functions instead.
"""
mutable struct MlirOperationState
    name::MlirStringRef
    location::MlirLocation
    nResults::intptr_t
    results::Ptr{MlirType}
    nOperands::intptr_t
    operands::Ptr{MlirValue}
    nRegions::intptr_t
    regions::Ptr{MlirRegion}
    nSuccessors::intptr_t
    successors::Ptr{MlirBlock}
    nAttributes::intptr_t
    attributes::Ptr{MlirNamedAttribute}
    enableResultTypeInference::Bool
end

mlirOperationStateGet(name, location) =
    @cvcall libmlir.mlirOperationStateGet(name::MlirStringRef, location::MlirLocation)::MlirOperationState
mlirOperationStateAddResults(state, n, results) =
    @cvcall libmlir.mlirOperationStateAddResults(
        state::Ptr{MlirOperationState},
        n::intptr_t, results::Ptr{MlirType},
    )::Cvoid
mlirOperationStateAddOperands(state, n, operands) =
    @cvcall libmlir.mlirOperationStateAddOperands(
        state::Ptr{MlirOperationState},
        n::intptr_t, operands::Ptr{MlirValue},
    )::Cvoid
mlirOperationStateAddOwnedRegions(state, n, regions) =
    @cvcall libmlir.mlirOperationStateAddOwnedRegions(
        state::Ptr{MlirOperationState},
        n::intptr_t, regions::Ptr{MlirRegion},
    )::Cvoid
mlirOperationStateAddSuccessors(state, n, successors) =
    @cvcall libmlir.mlirOperationStateAddSuccessors(
        state::Ptr{MlirOperationState},
        n::intptr_t, successors::Ptr{MlirBlock},
    )::Cvoid
mlirOperationStateAddAttributes(state, n, attributes) =
    @cvcall libmlir.mlirOperationStateAddAttributes(
        state::Ptr{MlirOperationState},
        n::intptr_t,
        attributes::Ptr{MlirNamedAttribute}
    )::Cvoid
mlirOperationStateEnableResultTypeInference(state) =
    @cvcall libmlir.mlirOperationStateEnableResultTypeInference(state::Ptr{MlirOperationState})::Cvoid

### Op Printing Flags

mlirOpPrintingFlagsCreate() =
    @cvcall libmlir.mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
mlirOpPrintingFlagsDestroy(flags) =
    @cvcall libmlir.mlirOpPrintingFlagsDestroy(flags::MlirOpPrintingFlags)::Cvoid
mlirOpPrintingFlagsEnableDebugInfo(flags, enable, pretty) =
    @cvcall libmlir.mlirOpPrintingFlagsEnableDebugInfo(flags::MlirOpPrintingFlags, enable::Bool, pretty::Bool)::Cvoid

### Operation

mlirOperationCreate(state) = @cvcall libmlir.mlirOperationCreate(state::Ptr{MlirOperationState})::MlirOperation
mlirOperationDestroy(operation) = @cvcall libmlir.mlirOperationDestroy(operation::MlirOperation)::Cvoid
mlirOperationIsNull(operation) = mlirIsNull(operation)
mlirOperationDump(operation) = @cvcall libmlir.mlirOperationDump(operation::MlirOperation)::Cvoid
mlirOperationPrintWithFlags(operation, flags, callback, userdata) =
    @cvcall libmlir.mlirOperationPrintWithFlags(
        operation::MlirOperation,
        flags::MlirOpPrintingFlags,
        callback::Ptr{Cvoid},
        userdata::Any,
    )::Cvoid
mlirOperationPrint(operation, callback, userdata) =
    @cvcall libmlir.mlirOperationPrint(operation::MlirOperation, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirOperationGetNumRegions(operation) = @cvcall libmlir.mlirOperationGetNumRegions(operation::MlirOperation)::intptr_t
mlirOperationGetRegion(operation, pos) = @cvcall libmlir.mlirOperationGetRegion(operation::MlirOperation, pos::intptr_t)::MlirRegion
mlirOperationGetNumResults(operation) = @cvcall libmlir.mlirOperationGetNumResults(operation::MlirOperation)::intptr_t
mlirOperationGetResult(operation, i) = @cvcall libmlir.mlirOperationGetResult(operation::MlirOperation, i::intptr_t)::MlirValue
mlirOperationClone(operation) = @cvcall libmlir.mlirOperationClone(operation::MlirOperation)::MlirOperation
mlirOperationEqual(op, other) = @cvcall libmlir.mlirOperationEqual(op::MlirOperation, other::MlirOperation)::Bool
mlirOperationGetLocation(operation) = @cvcall libmlir.mlirOperationGetLocation(operation::MlirOperation)::MlirLocation
mlirOperationGetName(operation) = @cvcall libmlir.mlirOperationGetName(operation::MlirOperation)::MlirIdentifier
mlirOperationGetBlock(operation) = @cvcall libmlir.mlirOperationGetBlock(operation::MlirOperation)::MlirBlock
mlirOperationGetParentOperation(operation) = @cvcall libmlir.mlirOperationGetParentOperation(operation::MlirOperation)::MlirOperation
mlirOperationGetNextInBlock(operation) = @cvcall libmlir.mlirOperationGetNextInBlock(operation::MlirOperation)::MlirOperation
mlirOperationGetFirstRegion(operation) = @cvcall libmlir.mlirOperationGetFirstRegion(operation::MlirOperation)::MlirRegion
mlirOperationVerify(operation) = @cvcall libmlir.mlirOperationVerify(operation::MlirOperation)::Bool
mlirOperationGetOperand(operation, i) = @cvcall libmlir.mlirOperationGetOperand(operation::MlirOperation, i::intptr_t)::MlirValue
mlirOperationSetOperand(operation, i, value) = @cvcall libmlir.mlirOperationSetOperand(operation::MlirOperation, i::intptr_t, value::MlirValue)::Cvoid
mlirOperationGetNumOperands(operation) = @cvcall libmlir.mlirOperationGetNumOperands(operation::MlirOperation)::intptr_t
mlirOperationGetAttributeByName(operation, name) = @cvcall libmlir.mlirOperationGetAttributeByName(operation::MlirOperation, name::MlirStringRef)::MlirAttribute
mlirOperationSetAttributeByName(operation, name, attribute) = @cvcall libmlir.mlirOperationSetAttributeByName(operation::MlirOperation, name::MlirStringRef, attribute::MlirAttribute)::Cvoid

### Module

mlirModuleCreateEmpty(location) = @cvcall libmlir.mlirModuleCreateEmpty(location::MlirLocation)::MlirModule
mlirModuleCreateParse(context, module_) = @cvcall libmlir.mlirModuleCreateParse(context::MlirContext, module_::MlirStringRef)::MlirModule
mlirModuleDestroy(module_) = @cvcall libmlir.mlirModuleDestroy(module_::MlirModule)::Cvoid
mlirModuleIsNull(module_) = mlirIsNull(module_)
mlirModuleGetBody(module_) = @cvcall libmlir.mlirModuleGetBody(module_::MlirModule)::MlirBlock
mlirModuleGetOperation(module_) = @cvcall libmlir.mlirModuleGetOperation(module_::MlirModule)::MlirOperation

### Pass Manager

mlirPassManagerCreate(context) = @cvcall libmlir.mlirPassManagerCreate(context::MlirContext)::MlirPassManager
mlirPassManagerDestroy(pm) = @cvcall libmlir.mlirPassManagerDestroy(pm::MlirPassManager)::Cvoid
mlirPassManagerIsNull(pm) = mlirIsNull(pm)
mlirPassManagerGetAsOpPassManager(pm) = @cvcall libmlir.mlirPassManagerGetAsOpPassManager(pm::MlirPassManager)::MlirOpPassManager
mlirPassManagerRunOnOp(pm, op) = @cvcall libmlir.mlirPassManagerRunOnOp(pm::MlirPassManager, op::MlirOperation)::MlirLogicalResult
mlirPassManagerEnableVerifier(pm) = @cvcall libmlir.mlirPassManagerEnableVerifier(pm::MlirPassManager)::Cvoid
mlirPassManagerAddOwnedPass(pm, pass) = @cvcall libmlir.mlirPassManagerAddOwnedPass(pm::MlirPassManager, pass::MlirPass)::Cvoid
mlirPassManagerGetNestedUnder(pm, opname) = @cvcall libmlir.mlirPassManagerGetNestedUnder(pm::MlirPassManager, opname::MlirStringRef)::MlirOpPassManager

### Op Pass Manager

mlirOpPassManagerIsNull(op_pass) = mlirIsNull(op_pass)
mlirOpPassManagerAddPipeline(op_pass, pipeline, callback, userdata) =
    @cvcall libmlir.mlirOpPassManagerAddPipeline(op_pass::MlirOpPassManager,pipeline::MlirStringRef, callback::Ptr{Cvoid}, userdata::Any)::MlirLogicalResult
mlirParsePassPipeline(op_pass, pipeline, callback, userdata) =
    @cvcall libmlir.mlirParsePassPipeline(op_pass::MlirOpPassManager, pipeline::MlirStringRef, callback::Ptr{Cvoid}, userdata::Any)::MlirLogicalResult
mlirPrintPassPipeline(op_pass, callback, userdata) =
    @cvcall libmlir.mlirPrintPassPipeline(op_pass::MlirOpPassManager, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirOpPassManagerAddOwnedPass(op_pass, pass) =
    @cvcall libmlir.mlirOpPassManagerAddOwnedPass(op_pass::MlirOpPassManager, pass::MlirPass)::Cvoid
mlirOpPassManagerGetNestedUnder(op_pass, opname) =
    @cvcall libmlir.mlirOpPassManagerGetNestedUnder(op_pass::MlirOpPassManager, opname::MlirStringRef)::MlirOpPassManager

### TypeID

# `ptr` must be 8 byte aligned and unique to a type valid for the duration of
#  the returned type id's usage
mlirTypeIDCreate(ptr) = @cvcall libmlir.mlirTypeIDCreate(ptr::MlirTypeIDAllocator)::MlirTypeID
mlirTypeIDIsNull(typeid) = mlirIsNull(typeid)
mlirTypeIDEqual(typeid1, typeid2) = @cvcall libmlir.mlirTypeIDEqual(typeid1::MlirTypeID, typeid2::MlirTypeID)::Bool
mlirTypeIDHashValue(typeid) = @cvcall libmlir.mlirTypeIDHashValue(typeid::MlirTypeID)::Csize_t

### TypeIDAllocator

mlirTypeIDAllocatorCreate() = @cvcall libmlir.mlirTypeIDAllocatorCreate()::MlirTypeIDAllocator
mlirTypeIDAllocatorDestroy(allocator) = @cvcall libmlir.mlirTypeIDAllocatorDestroy(allocator::MlirTypeIDAllocator)::Cvoid
mlirTypeIDAllocatorAllocateTypeID(allocator) =
    @cvcall libmlir.mlirTypeIDAllocatorAllocateTypeID(allocator::MlirTypeIDAllocator)::MlirTypeID

### Pass

"""
Structure of external `MlirPass` callbacks.
All callbacks are required to be set unless otherwise specified.
"""
struct MlirExternalPassCallbacks
    # This callback is called from the pass is created.
    # This is analogous to a C++ pass constructor.
    construct::Ptr{Cvoid}

    # This callback is called when the pass is destroyed
    # This is analogous to a C++ pass destructor.
    destruct::Ptr{Cvoid}

    # This callback is optional.
    # The callback is called before the pass is run, allowing a chance to
    # initialize any complex state necessary for running the pass.
    # See Pass::initialize(MLIRContext *).
    initialize::Ptr{Cvoid}

    # This callback is called when the pass is cloned.
    # See Pass::clonePass().
    clone::Ptr{Cvoid}

    # This callback is called when the pass is run.
    # See Pass::runOnOperation().
    run::Ptr{Cvoid}
end

# Creates an external `MlirPass` that calls the supplied `callbacks` using the
# supplied `userData`. If `opName` is empty, the pass is a generic operation
# pass. Otherwise it is an operation pass specific to the specified pass name.
function mlirCreateExternalPass(passID, name, argument,
                       description, opName,
                       nDependentDialects, dependentDialects,
                       callbacks, userData)
    @cvcall libmlir.mlirCreateExternalPass(
        passID::MlirTypeID, name::MlirStringRef, argument::MlirStringRef,
        description::MlirStringRef, opName::MlirStringRef,
        nDependentDialects::intptr_t, dependentDialects::Ptr{MlirDialectHandle},
        callbacks::MlirExternalPassCallbacks, userData::Ptr{Cvoid},
    )::MlirPass
end

# This signals that the pass has failed. This is only valid to call during
# the `run` callback of `MlirExternalPassCallbacks`.
# See Pass::signalPassFailure().
mlirExternalPassSignalFailure(pass) =
    @cvcall libmlir.mlirExternalPassSignalFailure(pass::MlirExternalPass)::Cvoid

end # module LibMLIR
