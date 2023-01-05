module LibMLIR

const libmlir = expanduser("~/Projects/iree-build/lib/libIREECompiler.so.0")
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
    :MlirOpPassManager,)
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
    :MlirValue,)
    @eval struct $T
        ptr::Ptr{Cvoid}
    end
end

struct MlirLogicalResult
    value::Int8
end

struct MlirStringRef
    char::Ptr{Cchar}
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

mlirDialectRegistryCreate() = @ccall libmlir.mlirDialectRegistryCreate()::MlirDialectRegistry
mlirDialectRegistryDestroy(registry) = @ccall libmlir.mlirDialectRegistryDestroy(registry::MlirDialectRegistry)::Cvoid
mlirRegisterAllDialects(registry) = @ccall libmlir.mlirRegisterAllDialects(registry::MlirDialectRegistry)::Cvoid

### Context

mlirContextCreate() = @ccall libmlir.mlirContextCreate()::MlirContext
mlirContextDestroy(context) = @ccall libmlir.mlirContextDestroy(context::MlirContext)::Cvoid
mlirContextGetNumLoadedDialects(context) = @ccall libmlir.mlirContextGetNumLoadedDialects(context::MlirContext)::Cint
mlirContextGetOrLoadDialect(context, dialect) = @ccall libmlir.mlirContextGetOrLoadDialect(context::MlirContext, dialect::MlirStringRef)::MlirDialect
mlirContextLoadAllAvailableDialects(context) = @ccall libmlir.mlirContextLoadAllAvailableDialects(context::MlirContext)::Cvoid
mlirContextIsRegisteredOperation(context, op) = @ccall libmlir.mlirContextIsRegisteredOperation(context::MlirContext, op::MlirStringRef)::Bool
mlirContextIsNull(context) = mlirIsNull(context)
mlirContextAppendDialectRegistry(context, registry) = @ccall libmlir.mlirContextAppendDialectRegistry(context::MlirContext, registry::MlirDialectRegistry)::Cvoid

### Dialect

mlirDialectIsNull(dialect) = mlirIsNull(dialect)
mlirDialectGetNamespace(dialect) = @ccall libmlir.mlirDialectGetNamespace(dialect::MlirDialect)::MlirStringRef
mlirDialectGetContext(dialect) = @ccall libmlir.mlirDialectGetContext(dialect::MlirDialect)::MlirContext

### Location

mlirLocationIsNull(location) = mlirIsNull(location)
mlirLocationPrint(location, callback, userdata) =
    @ccall libmlir.mlirLocationPrint(location::MlirLocation, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirLocationUnknownGet(context) = @ccall libmlir.mlirLocationUnknownGet(context::MlirContext)::MlirLocation
mlirLocationFileLineColGet(context, filename, line, column) =
    @ccall libmlir.mlirLocationFileLineColGet(
        context::MlirContext,
        filename::MlirStringRef,
        line::Cuint,
        column::Cuint,
    )::MlirLocation

### Type

mlirTypeDump(type) = @ccall libmlir.mlirTypeDump(type::MlirType)::Cvoid
mlirTypePrint(type, callback, userdata) = @ccall libmlir.mlirTypePrint(type::MlirType, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirTypeIsNull(type) = mlirIsNull(type)
mlirIndexTypeGet(context) = @ccall libmlir.mlirIndexTypeGet(context::MlirContext)::MlirType
mlirIntegerTypeGet(context, size) = @ccall libmlir.mlirIntegerTypeGet(context::MlirContext, size::Cuint)::MlirType
mlirIntegerTypeSignedGet(context, size) =
    @ccall libmlir.mlirIntegerTypeSignedGet(context::MlirContext, size::Cuint)::MlirType
mlirF32TypeGet(context) = @ccall libmlir.mlirF32TypeGet(context::MlirContext)::MlirType
mlirF64TypeGet(context) = @ccall libmlir.mlirF64TypeGet(context::MlirContext)::MlirType
mlirFunctionTypeGet(context, nargs, args, nresults, results) =
    @ccall libmlir.mlirFunctionTypeGet(context::MlirContext, nargs::intptr_t, args::Ptr{MlirType}, nresults::intptr_t, results::Ptr{MlirType})::MlirType
mlirRankedTensorTypeGetChecked(location, rank, shape, eltype, encoding) =
    @ccall libmlir.mlirRankedTensorTypeGetChecked(
        location::MlirLocation,
        rank::intptr_t,
        shape::Ptr{Clong},
        eltype::MlirType,
        encoding::MlirAttribute,
    )::MlirType
mlirShapedTypeGetDimSize(type, i) =
    @ccall libmlir.mlirShapedTypeGetDimSize(type::MlirType, i::intptr_t)::Int64
mlirShapedTypeHasRank(type) =
    @ccall libmlir.mlirShapedTypeHasRank(type::MlirType)::Bool
mlirShapedTypeGetRank(type) =
    @ccall libmlir.mlirShapedTypeGetRank(type::MlirType)::Int64
mlirShapedTypeGetElementType(type) =
    @ccall libmlir.mlirShapedTypeGetElementType(type::MlirType)::MlirType
mlirTypeIsARankedTensor(type) =
    @ccall libmlir.mlirTypeIsARankedTensor(type::MlirType)::Bool
mlirTypeIsAShaped(type) =
    @ccall libmlir.mlirTypeIsAShaped(type::MlirType)::Bool
mlirIntegerTypeGetWidth(type) =
    @ccall libmlir.mlirIntegerTypeGetWidth(type::MlirType)::UInt32
mlirIntegerTypeIsSignless(type) =
    @ccall libmlir.mlirIntegerTypeIsSignless(type::MlirType)::Bool
mlirIntegerTypeIsSigned(type) =
    @ccall libmlir.mlirIntegerTypeIsSigned(type::MlirType)::Bool
mlirTypeIsAInteger(type) =
    @ccall libmlir.mlirTypeIsAInteger(type::MlirType)::Bool
mlirTypeIsAF32(type) =
    @ccall libmlir.mlirTypeIsAF32(type::MlirType)::Bool
mlirTypeIsAF64(type) =
    @ccall libmlir.mlirTypeIsAF64(type::MlirType)::Bool
mlirTypeIsAIndex(type) =
    @ccall libmlir.mlirTypeIsAIndex(type::MlirType)::Bool

### Attributes

mlirAttributeGetNull() = @ccall libmlir.mlirAttributeGetNull()::MlirAttribute
mlirStringAttrGet(context, str) = @ccall libmlir.mlirStringAttrGet(context::MlirContext, str::MlirStringRef)::MlirAttribute
mlirTypeAttrGet(type) = @ccall libmlir.mlirTypeAttrGet(type::MlirType)::MlirAttribute
mlirAttributeGetContext(attribute) = @ccall libmlir.mlirAttributeGetContext(attribute::MlirAttribute)::MlirContext
mlirAttributeParseGet(context, str) = @ccall libmlir.mlirAttributeParseGet(context::MlirContext, str::MlirStringRef)::MlirAttribute
mlirAttributePrint(attribute, callback, userdata) = @ccall libmlir.mlirAttributePrint(attribute::MlirAttribute, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirAttributeGetType(attribute) = @ccall libmlir.mlirAttributeGetType(attribute::MlirAttribute)::MlirType
mlirDenseElementsAttrFloatGet(type, nfloats, floats) =
    @ccall libmlir.mlirDenseElementsAttrFloatGet(type::MlirType, nfloats::intptr_t, floats::Ptr{Cfloat})::MlirAttribute
mlirDenseElementsAttrDoubleGet(type, ndoubles, doubles) =
    @ccall libmlir.mlirDenseElementsAttrDoubleGet(type::MlirType, ndoubles::intptr_t, doubles::Ptr{Cdouble})::MlirAttribute
mlirDenseElementsAttrInt32Get(type, nints, ints) =
    @ccall libmlir.mlirDenseElementsAttrInt32Get(type::MlirType, nints::intptr_t, ints::Ptr{Int32})::MlirAttribute
mlirDenseElementsAttrInt64Get(type, nints, ints) =
    @ccall libmlir.mlirDenseElementsAttrInt64Get(type::MlirType, nints::intptr_t, ints::Ptr{Int64})::MlirAttribute
mlirDenseI64ArrayGet(context, size, values) =
    @ccall libmlir.mlirDenseI64ArrayGet(context::MlirContext, size::intptr_t, values::Ptr{Int64})::MlirAttribute
mlirFloatAttrDoubleGet(context, type, f) =
    @ccall libmlir.mlirFloatAttrDoubleGet(context::MlirContext, type::MlirType, f::Cdouble)::MlirAttribute
mlirIntegerAttrGet(type, i) =
    @ccall libmlir.mlirIntegerAttrGet(type::MlirType, i::Int64)::MlirAttribute
mlirIntegerAttrGetValueInt(attribute) =
    @ccall libmlir.mlirIntegerAttrGetValueInt(attribute::MlirAttribute)::Clong
mlirAttributeIsAInteger(attribute) =
    @ccall libmlir.mlirAttributeIsAInteger(attribute::MlirAttribute)::Bool
mlirAttributeIsAType(attribute) =
    @ccall libmlir.mlirAttributeIsAType(attribute::MlirAttribute)::Bool
mlirAttributeIsAFloat(attribute) =
    @ccall libmlir.mlirAttributeIsAFloat(attribute::MlirAttribute)::Bool
mlirArrayAttrGet(context, nelements, elements) =
    @ccall libmlir.mlirArrayAttrGet(context::MlirContext, nelements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute

### Identifier

mlirIdentifierGet(context, str) = @ccall libmlir.mlirIdentifierGet(context::MlirContext, str::MlirStringRef)::MlirIdentifier
mlirIdentifierStr(ident) = @ccall libmlir.mlirIdentifierStr(ident::MlirIdentifier)::MlirStringRef

### Named Attributes

mlirNamedAttributeGet(name, attr) =
    @ccall libmlir.mlirNamedAttributeGet(name::MlirIdentifier, attr::MlirAttribute)::MlirNamedAttribute

### Region

mlirRegionCreate() = @ccall libmlir.mlirRegionCreate()::MlirRegion
mlirRegionDestroy(region) = @ccall libmlir.mlirRegionDestroy(region::MlirRegion)::Cvoid
mlirRegionIsNull(region) = mlirIsNull(region)
mlirRegionAppendOwnedBlock(region, block) = @ccall libmlir.mlirRegionAppendOwnedBlock(region::MlirRegion, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlock(region, pos, block) =
    @ccall libmlir.mlirRegionInsertOwnedBlock(region::MlirRegion, pos::intptr_t, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlockAfter(region, reference, block) =
    @ccall libmlir.mlirRegionInsertOwnedBlockAfter(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
mlirRegionInsertOwnedBlockBefore(region, reference, block) =
    @ccall libmlir.mlirRegionInsertOwnedBlockBefore(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
mlirRegionGetFirstBlock(region) = @ccall libmlir.mlirRegionGetFirstBlock(region::MlirRegion)::MlirBlock
mlirRegionGetNextInOperation(region) = @ccall libmlir.mlirRegionGetNextInOperation(region::MlirRegion)::MlirRegion

### Block

mlirBlockCreate(nargs, args, locs) = @ccall libmlir.mlirBlockCreate(nargs::intptr_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation})::MlirBlock
mlirBlockDestroy(block) = @ccall libmlir.mlirBlockDestroy(block::MlirBlock)::Cvoid
mlirBlockIsNull(block) = mlirIsNull(block)
mlirBlockDetach(block) = @ccall libmlir.mlirBlockDetach(block::MlirBlock)::Cvoid
mlirBlockAppendOwnedOperation(block, operation) =
    @ccall libmlir.mlirBlockAppendOwnedOperation(block::MlirBlock, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperation(block, pos, operation) =
    @ccall libmlir.mlirBlockInsertOwnedOperation(block::MlirBlock, pos::intptr_t, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperationAfter(block, reference, operation) =
    @ccall libmlir.mlirBlockInsertOwnedOperationAfter(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
mlirBlockInsertOwnedOperationBefore(block, reference, operation) =
    @ccall libmlir.mlirBlockInsertOwnedOperationBefore(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
mlirBlockGetArgument(block, i) = @ccall libmlir.mlirBlockGetArgument(block::MlirBlock, i::intptr_t)::MlirValue
mlirBlockGetNumArguments(block) = @ccall libmlir.mlirBlockGetNumArguments(block::MlirBlock)::intptr_t
mlirBlockAddArgument(block, type, loc) =
    @ccall libmlir.mlirBlockAddArgument(block::MlirBlock, type::MlirType, loc::MlirLocation)::MlirValue
mlirBlockPrint(block, callback, userdata) =
    @ccall libmlir.mlirBlockAddArgument(block::MlirBlock, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirBlockGetNextInRegion(block) = @ccall libmlir.mlirBlockGetNextInRegion(block::MlirBlock)::MlirBlock
mlirBlockGetFirstOperation(block) = @ccall libmlir.mlirBlockGetFirstOperation(block::MlirBlock)::MlirOperation

### Value

mlirValueGetType(value) = @ccall libmlir.mlirValueGetType(value::MlirValue)::MlirType
mlirValueIsNull(value) = mlirIsNull(value)
mlirValueDump(value) = @ccall libmlir.mlirValueDump(value::MlirValue)::Cvoid
mlirValuePrint(value, callback, userdata) = @ccall libmlir.mlirValuePrint(value::MlirValue, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirValueGetFirstUse(value) = @ccall libmlir.mlirValueGetFirstUse(value::MlirValue)::MlirOpOperand
mlirValueIsAOpResult(value) = @ccall libmlir.mlirValueIsAOpResult(value::MlirValue)::Bool
mlirOpResultGetOwner(value) = @ccall libmlir.mlirOpResultGetOwner(value::MlirValue)::MlirOperation
mlirValueIsABlockArgument(value) = @ccall libmlir.mlirValueIsABlockArgument(value::MlirValue)::Bool
mlirBlockArgumentGetOwner(value) = @ccall libmlir.mlirBlockArgumentGetOwner(value::MlirValue)::MlirBlock

### Op Operand

mlirOpOperandIsNull(op_operand) = mlirIsNull(op_operand)
mlirOpOperandGetOwner(op_operand) = @ccall libmlir.mlirOpOperandGetOwner(op_operand::MlirOpOperand)::MlirValue
mlirOpOperandGetOperandNumber(op_operand) = @ccall libmlir.mlirOpOperandGetOperandNumber(op_operand::MlirOpOperand)::Cuint
mlirOpOperandGetNextUse(op_operand) = @ccall libmlir.mlirOpOperandGetNextUse(op_operand::MlirOpOperand)::MlirOpOperand

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
    @ccall libmlir.mlirOperationStateGet(name::MlirStringRef, location::MlirLocation)::MlirOperationState
mlirOperationStateAddResults(state, n, results) =
    @ccall libmlir.mlirOperationStateAddResults(
        state::Ptr{MlirOperationState},
        n::intptr_t, results::Ptr{MlirType}
    )::Cvoid
mlirOperationStateAddOperands(state, n, operands) =
    @ccall libmlir.mlirOperationStateAddOperands(
        state::Ptr{MlirOperationState},
        n::intptr_t, operands::Ptr{MlirValue}
    )::Cvoid
mlirOperationStateAddOwnedRegions(state, n, regions) =
    @ccall libmlir.mlirOperationStateAddOwnedRegions(
        state::Ptr{MlirOperationState},
        n::intptr_t, regions::Ptr{MlirRegion}
    )::Cvoid
mlirOperationStateAddSuccessors(state, n, successors) =
    @ccall libmlir.mlirOperationStateAddSuccessors(
        state::Ptr{MlirOperationState},
        n::intptr_t, successors::Ptr{MlirBlock}
    )::Cvoid
mlirOperationStateAddAttributes(state, n, attributes) =
    @ccall libmlir.mlirOperationStateAddAttributes(
        state::Ptr{MlirOperationState},
        n::intptr_t,
        attributes::Ptr{MlirNamedAttribute}
    )::Cvoid
mlirOperationStateEnableResultTypeInference(state) =
    @ccall libmlir.mlirOperationStateEnableResultTypeInference(state::Ptr{MlirOperationState})::Cvoid

### Op Printing Flags

mlirOpPrintingFlagsCreate() =
    @ccall libmlir.mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
mlirOpPrintingFlagsEnableDebugInfo(flags, enable, pretty) =
    @ccall libmlir.mlirOpPrintingFlagsEnableDebugInfo(flags::MlirOpPrintingFlags, enable::Bool, pretty::Bool)::Cvoid

### Operation

mlirOperationCreate(state) = @ccall libmlir.mlirOperationCreate(state::Ptr{MlirOperationState})::MlirOperation
mlirOperationDestroy(operation) = @ccall libmlir.mlirOperationDestroy(operation::MlirOperation)::Cvoid
mlirOperationIsNull(operation) = mlirIsNull(operation)
mlirOperationDump(operation) = @ccall libmlir.mlirOperationDump(operation::MlirOperation)::Cvoid
mlirOperationPrintWithFlags(operation, flags, callback, userdata) =
    @ccall libmlir.mlirOperationPrintWithFlags(
        operation::MlirOperation,
        flags::MlirOpPrintingFlags,
        callback::Ptr{Cvoid},
        userdata::Any,
    )::Cvoid
mlirOperationPrint(operation, callback, userdata) =
    @ccall libmlir.mlirOperationPrint(operation::MlirOperation, callback::Ptr{Cvoid}, userdata::Any)::Cvoid
mlirOperationGetNumRegions(operation) = @ccall libmlir.mlirOperationGetNumRegions(operation::MlirOperation)::intptr_t
mlirOperationGetRegion(operation, pos) = @ccall libmlir.mlirOperationGetRegion(operation::MlirOperation, pos::intptr_t)::MlirRegion
mlirOperationGetNumResults(operation) = @ccall libmlir.mlirOperationGetNumResults(operation::MlirOperation)::intptr_t
mlirOperationGetResult(operation, i) = @ccall libmlir.mlirOperationGetResult(operation::MlirOperation, i::intptr_t)::MlirValue
mlirOperationClone(operation) = @ccall libmlir.mlirOperationClone(operation::MlirOperation)::MlirOperation
mlirOperationEqual(op, other) = @ccall libmlir.mlirOperationEqual(op::MlirOperation, other::MlirOperation)::Bool
mlirOperationGetLocation(operation) = @ccall libmlir.mlirOperationGetLocation(operation::MlirOperation)::MlirLocation
mlirOperationGetName(operation) = @ccall libmlir.mlirOperationGetName(operation::MlirOperation)::MlirIdentifier
mlirOperationGetBlock(operation) = @ccall libmlir.mlirOperationGetBlock(operation::MlirOperation)::MlirBlock
mlirOperationGetParentOperation(operation) = @ccall libmlir.mlirOperationGetParentOperation(operation::MlirOperation)::MlirOperation
mlirOperationGetNextInBlock(operation) = @ccall libmlir.mlirOperationGetNextInBlock(operation::MlirOperation)::MlirOperation
mlirOperationGetFirstRegion(operation) = @ccall libmlir.mlirOperationGetFirstRegion(operation::MlirOperation)::MlirRegion
mlirOperationVerify(operation) = @ccall libmlir.mlirOperationVerify(operation::MlirOperation)::Bool
mlirOperationGetOperand(operation, i) = @ccall libmlir.mlirOperationGetOperand(operation::MlirOperation, i::intptr_t)::MlirValue
mlirOperationSetOperand(operation, i, value) = @ccall libmlir.mlirOperationSetOperand(operation::MlirOperation, i::intptr_t, value::MlirValue)::Cvoid
mlirOperationGetNumOperands(operation) = @ccall libmlir.mlirOperationGetNumOperands(operation::MlirOperation)::intptr_t

### Module

mlirModuleCreateEmpty(location) = @ccall libmlir.mlirModuleCreateEmpty(location::MlirLocation)::MlirModule
mlirModuleCreateParse(context, module_) = @ccall libmlir.mlirModuleCreateParse(context::MlirContext, module_::MlirStringRef)::MlirModule
mlirModuleDestroy(module_) = @ccall libmlir.mlirModuleDestroy(module_::MlirModule)::Cvoid
mlirModuleIsNull(module_) = mlirIsNull(module_)
mlirModuleGetBody(module_) = @ccall libmlir.mlirModuleGetBody(module_::MlirModule)::MlirBlock
mlirModuleGetOperation(module_) = @ccall libmlir.mlirModuleGetOperation(module_::MlirModule)::MlirOperation

### Pass Manager

mlirPassManagerCreate(context) = @ccall libmlir.mlirPassManagerCreate(context::MlirContext)::MlirPassManager
mlirPassManagerDestroy(pass) = @ccall libmlir.mlirPassManagerDestroy(pass::MlirPassManager)::Cvoid
mlirPassManagerIsNull(pass) = mlirIsNull(pass)
mlirPassManagerGetAsOpPassManager(pass) = @ccall libmlir.mlirPassManagerGetAsOpPassManager(pass::MlirPassManager)::MlirOpPassManager
mlirPassManagerRun(pass, module_) = @ccall libmlir.mlirPassManagerRun(pass::MlirPassManager, module_::MlirModule)::MlirLogicalResult
mlirPassManagerEnableVerifier(pass) = @ccall libmlir.mlirPassManagerEnableVerifier(pass::MlirPassManager)::Cvoid

### Op Pass Manager

mlirOpPassManagerIsNull(op_pass) = mlirIsNull(op_pass)
mlirOpPassManagerAddPipeline(op_pass, pipeline, callback, userdata) =
    @ccall libmlir.mlirOpPassManagerAddPipeline(op_pass::MlirOpPassManager,pipeline::MlirStringRef, callback::Ptr{Cvoid}, userdata::Any)::MlirLogicalResult
mlirParsePassPipeline(op_pass, pipeline, callback, userdata) =
    @ccall libmlir.mlirParsePassPipeline(op_pass::MlirOpPassManager, pipeline::MlirStringRef, callback::Ptr{Cvoid}, userdata::Any)::MlirLogicalResult
mlirPrintPassPipeline(op_pass, callback, userdata) =
    @ccall libmlir.mlirPrintPassPipeline(op_pass::MlirOpPassManager, callback::Ptr{Cvoid}, userdata::Any)::Cvoid

end # module LibMLIR
