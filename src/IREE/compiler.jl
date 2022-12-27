module Compiler

import ...MLIR.LibMLIR:
    MlirContext,
    MlirLogicalResult,
    MlirOpPassManager,
    MlirOperation

import ..IREE: libiree

### Types

# storage = void*
struct IreeCompilerOptions
    ptr::Ptr{Cvoid}
end

### Registration

ireeCompilerRegisterTargetBackends() = @ccall libiree.ireeCompilerRegisterTargetBackends()::Cvoid
ireeCompilerRegisterAllPasses() = @ccall libiree.ireeCompilerRegisterAllPasses()::Cvoid
ireeCompilerRegisterAllDialects(context) =
    @ccall libiree.ireeCompilerRegisterAllDialects(context::MlirContext)::Cvoid

### Compiler Options

ireeCompilerOptionsCreate() = @ccall libiree.ireeCompilerOptionsCreate()::IreeCompilerOptions
ireeCompilerOptionsDestroy(options) = @ccall libiree.ireeCompilerOptionsDestroy(options::IreeCompilerOptions)::Cvoid
ireeCompilerOptionsSetFlags(options, nflags, flags, onerror, userdata) =
    @ccall libiree.ireeCompilerOptionsSetFlags(
        options::IreeCompilerOptions,
        nflags::Cint, flags::Ptr{Cstring},
        onerror::Ptr{Cvoid}, userdata::Ptr{Cvoid},
    )::MlirLogicalResult

### Pass Manager

ireeCompilerBuildIREEVMPassPipeline(options, op_pass) =
    @ccall libiree.ireeCompilerBuildIREEVMPassPipeline(options::IreeCompilerOptions, op_pass::MlirOpPassManager)::Cvoid

### Translation

ireeCompilerTranslateModuletoVMBytecode(options, operation, callback, userdata) =
    @ccall libiree.ireeCompilerTranslateModuletoVMBytecode(
        options::IreeCompilerOptions,
        operation::MlirOperation,
        callback::Ptr{Cvoid}, userdata::Ptr{Cvoid}
    )::MlirLogicalResult

end # module Compiler
