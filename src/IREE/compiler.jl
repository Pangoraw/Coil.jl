module Compiler

import ...MLIR.LibMLIR:
    MlirContext,
    MlirLogicalResult,
    MlirOpPassManager,
    MlirOperation

import ...Coil: @cvcall
import ..IREE: libiree

### Types

# storage = void*
struct IreeCompilerOptions
    ptr::Ptr{Cvoid}
end

### Registration

ireeCompilerRegisterTargetBackends() = @cvcall libiree.ireeCompilerRegisterTargetBackends()::Cvoid
ireeCompilerRegisterAllPasses() = @cvcall libiree.ireeCompilerRegisterAllPasses()::Cvoid
ireeCompilerRegisterAllDialects(context) =
    @cvcall libiree.ireeCompilerRegisterAllDialects(context::MlirContext)::Cvoid

### Compiler Options

ireeCompilerOptionsCreate() = @cvcall libiree.ireeCompilerOptionsCreate()::IreeCompilerOptions
ireeCompilerOptionsDestroy(options) = @cvcall libiree.ireeCompilerOptionsDestroy(options::IreeCompilerOptions)::Cvoid
ireeCompilerOptionsSetFlags(options, nflags, flags, onerror, userdata) =
    @cvcall libiree.ireeCompilerOptionsSetFlags(
        options::IreeCompilerOptions,
        nflags::Cint, flags::Ptr{Cstring},
        onerror::Ptr{Cvoid}, userdata::Ptr{Cvoid},
    )::MlirLogicalResult

### Pass Manager

ireeCompilerBuildIREEVMPassPipeline(options, op_pass) =
    @cvcall libiree.ireeCompilerBuildIREEVMPassPipeline(options::IreeCompilerOptions, op_pass::MlirOpPassManager)::Cvoid

### Translation

ireeCompilerTranslateModuletoVMBytecode(options, operation, callback, userdata) =
    @cvcall libiree.ireeCompilerTranslateModuletoVMBytecode(
        options::IreeCompilerOptions,
        operation::MlirOperation,
        callback::Ptr{Cvoid}, userdata::Ptr{Cvoid}
    )::MlirLogicalResult

end # module Compiler
