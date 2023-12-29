module Compiler

using CEnum

import ...MLIR.LibMLIR:
    MlirContext,
    MlirLogicalResult,
    MlirOpPassManager,
    MlirOperation,
    MlirDialectRegistry

import ...Coil: @cvcall
import ..IREE: libiree

###

struct iree_compiler_error_t end
struct iree_compiler_session_t end
struct iree_compiler_invocation_t end
struct iree_compiler_source_t end
struct iree_compiler_output_t end

### Errors

"""
Destroys an error. Only non-nullptr errors must be destroyed, but it is
legal to destroy nullptr.
"""
ireeCompilerErrorDestroy(error) = @cvcall libiree.ireeCompilerErrorDestroy(error::Ptr{iree_compiler_error_t})::Cvoid

"""
Gets the message associated with the error as a C-string. The string will be
valid until the error is destroyed.
"""
ireeCompilerErrorGetMessage(error) = ireeCompilerErrorGetMessage(error::Ptr{iree_compiler_error_t})::Cstring

### Global initialization.

"""
Gets the version of the API that this compiler exports.
The version is encoded with the lower 16 bits containing the minor version
and upper bits containing the major version.
The compiler API is versioned. Within a major version, symbols may be
added, but existing symbols must not be removed or changed to alter
previously exposed functionality. A major version bump implies an API
break and no forward or backward compatibility is assumed across major
versions.
"""
ireeCompilerGetAPIVersion() = @cvcall libiree.ireeCompilerGetAPIVersion()::Cint

"""
The compiler must be globally initialized before further use.
It is intended to be called as part of the hosting process's startup
sequence. Any failures that it can encounter are fatal and will abort
the process.
It is legal to call this multiple times, and each call must be balanced
by a call to |ireeCompilerGlobalShutdown|. The final shutdown call will
permanently disable the compiler for the process and subsequent calls
to initialize will fail/abort. If this is not desirable, some higher level
code must hold initialization open with its own call.
"""
ireeCompilerGlobalInitialize() = @cvcall libiree.ireeCompilerGlobalInitialize()::Cvoid

"""
Gets the build revision of the IREE compiler. In official releases, this
will be a string with the build tag. In development builds, it may be an
empty string. The returned is valid for as long as the compiler is
initialized.
Available since: 1.1
"""
ireeCompilerGetRevision() = @cvcall libiree.ireeCompilerGetRevision()::Cstring

"""
Processes an argc/argv from main() using platform specific dark magic,
possibly updating them in-place to cleaned up values. On most systems,
this is a no-op. However, on Windows, this will do various processing
to extract the original command line and decode it properly as UTF-8.
It is therefore essential that this be passed the actual argc/argv to main
and not some permutation of it. Any updated argv will persist until a
final call to |ireeCompilerGlobalShutdown|. This must be called after
|ireeCompilerGlobalInitialize|.

We're really sorry about this. Old platforms are annoying.
This API is not yet considered version-stable. If using out of tree, please
contact the developers.
"""
ireeCompilerGetProcessCLArgs(argc, argv) =
    @cvcall libiree.ireeCompilerGetProcessCLArgs(argc::Ptr{Cint}, argv::Ptr{Cstring})::Cvoid

"""
Initializes the command line environment from an explicit argc/argv
(typically the result of a prior call to ireeCompilerGetProcessCLArgs).
This uses dark magic to setup the usual array of expected signal handlers.
This API is not yet considered version-stable. If using out of tree, please
contact the developers.

Note that there is as yet no facility to register new global command line
options from the C API. However, this facility should be sufficient for
subordinating builtin command line options to a higher level integration
by tunneling global options into the initialization sequence.
"""
ireeCompilerSetupGlobalCL(argc, argv, banner, installSignalHandlers) =
    @cvcall libiree.ireeCompilerSetupGlobalCL(argc::Cint, argv::Ptr{Cstring}, banner::Cstring, installSignalHandlers::Bool)::Cvoid

"""
// Destroys any process level resources that the compiler may have created.
// This must be called prior to library unloading.
"""
ireeCompilerGlobalShutdown() = @cvcall libiree.ireeCompilerGlobalShutdown()::Cvoid

"""
Invokes a callback with each registered HAL target backend.

This is really only suitable for global CLI-like tools, as plugins may
register target backends and plugins are activated at the session level.
"""
ireeCompilerEnumerateRegisteredHALTargetBackends(callback, userData) =
    @cvcall libiree.ireeCompilerEnumerateRegisteredHALTargetBackends(callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid

"""
Enumerates all plugins that are linked into the compiler.
Available since: 1.2
"""
ireeCompilerEnumeratePlugins(callback, userData) = @cvcall libiree.ireeCompilerEnumeratePlugins(callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid

# Session management.
# A session represents a scope where one or more runs can be executed.
# Internally, it consists of an MLIRContext and a private set of session
# options. If the CL environment was initialized, session options will be
# bootstrapped from global flags.
#
# Session creation cannot fail in a non-fatal way.

"""
Creates a new session (which must be destroyed by
ireeCompilerSessionDestroy).
"""
ireeCompilerSessionCreate() = @cvcall libiree.ireeCompilerSessionCreate()::Ptr{iree_compiler_session_t}

"""
Destroys a session.
"""
ireeCompilerSessionDestroy(session) = @cvcall libiree.ireeCompilerSessionDestroy(session::Ptr{iree_compiler_session_t})::Cvoid

"""
Sets session-local flags. These are a subset of flags supported by CLI
tools and are privately scoped.
"""

ireeCompilerSessionSetFlags(session, argc, argv) =
    @cvcall libiree.ireeCompilerSessionSetFlags(session::Ptr{iree_compiler_session_t}, argc::Cint, argv::Ptr{Cstring})::Ptr{iree_compiler_error_t}

"""
Gets textual flags actually in effect from any source. Optionally, only
calls back for non-default valued flags.
"""
ireeCompilerSessionGetFlags(session, nonDefaultOnly, onFlag, userData) =
    @cvcall libiree.ireeCompilerSessionGetFlags(session::Ptr{iree_compiler_session_t}, nonDefaultOnly::Bool, onFlag::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid

# Run management.
# Runs execute against a session and represent a discrete invocation of the
# compiler.

@cenum iree_compiler_diagnostic_severity_t begin
    IREE_COMPILER_DIAGNOSTIC_SEVERITY_NOTE = 0
    IREE_COMPILER_DIAGNOSTIC_SEVERITY_WARNING = 1
    IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR = 2
    IREE_COMPILER_DIAGNOSTIC_SEVERITY_REMARK = 3
end

ireeCompilerInvocationCreate(session) =
    @cvcall libiree.ireeCompilerInvocationCreate(session::Ptr{iree_compiler_session_t})::Ptr{iree_compiler_invocation_t}

"""
Enables a callback to receive diagnostics. This is targeted at API use of
the compiler, allowing fine grained collection of formatted diagnostic
records. It is not completely identical to
|ireeCompilerInvocationEnableConsoleDiagnostics| which produces output
suitable for an interactive stream (including color detection, etc) and has
additional features for reading source files, etc. With default flags, no
system state outside of the session will be used (i.e. no debug information
loaded from files, etc).
The |flags| parameter is reserved for the future and must be 0.
The |callback| may be invoked from any thread at any time prior to
destruction of the invocation. The callback should not make any calls back
into compiler APIs.
The |message| passes to the callback is only valid for the duration of
the callback and the |messageSize| does not include a terminator nul.
"""
ireeCompilerInvocationEnableCallbackDiagnostics(
    inv, flags,
    callback,
    userData) = @cvcall libiree.ireeCompilerInvocationEnableCallbackDiagnostics(inv::Ptr{iree_compiler_invocation_t}, flags::Cint, callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid

"""
Enables default, pretty-printed diagnostics to the console. This is usually
the right thing to do for command-line tools, but other mechanisms are
preferred for library use.
"""
ireeCompilerInvocationEnableConsoleDiagnostics(inv) = @cvcall libiree.ireeCompilerInvocationEnableConsoleDiagnostics(inv::Ptr{iree_compiler_invocation_t})::Cvoid

"""
Destroys a run.
"""
ireeCompilerInvocationDestroy(inv) = @cvcall libiree.ireeCompilerInvocationDestroy(inv::Ptr{iree_compiler_invocation_t})::Cvoid

"""
Sets a crash handler on the invocation. In the event of a crash, the callback
will be invoked to create an output which will receive the crash dump.
The callback should either set |*outOutput| to a new |iree_compiler_output_t|
or return an error. Ownership of the output is passed to the caller.
The implementation implicitly calls |ireeCompilerOutputKeep| on the
output.
"""
ireeCompilerInvocationSetCrashHandler(
    inv, genLocalReproducer,
    callback, userData) = @cvcall libiree.ireeCompilerInvocationSetCrashHandler(inv::Ptr{iree_compiler_invocation_t}, genLocalReproducer::Bool, callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid

"""
Parses a source into this instance in preparation for performing a
compilation action.
Returns false and emits diagnostics on failure.
"""
ireeCompilerInvocationParseSource(inv, source) =
    @cvcall libiree.ireeCompilerInvocationParseSource(inv::Ptr{iree_compiler_invocation_t},
        source::Ptr{iree_compiler_source_t})::Bool

"""
Sets a mnemonic phase name to run compilation from. Default is "input".
The meaning of this is pipeline specific. See IREEVMPipelinePhase
for the standard pipeline.
Available since: 1.3
"""
ireeCompilerInvocationSetCompileFromPhase(inv, phase) =
    @cvcall libiree.ireeCompilerInvocationSetCompileFromPhase(inv::Ptr{iree_compiler_invocation_t},
        phase::Cstring)::Cvoid

"""
Sets a mnemonic phase name to run compilation to. Default is "end".
The meaning of this is pipeline specific. See IREEVMPipelinePhase
for the standard pipeline.
"""
ireeCompilerInvocationSetCompileToPhase(inv, phase) =
    @cvcall libiree.ireeCompilerInvocationSetCompileToPhase(inv::Ptr{iree_compiler_invocation_t}, phase::Cstring)::Cvoid

"""
Enables/disables verification of IR after each pass. Defaults to enabled.
"""
ireeCompilerInvocationSetVerifyIR(inv, enable) =
    @cvcall libiree.ireeCompilerInvocationSetVerifyIR(inv::Ptr{iree_compiler_invocation_t}, enable::Bool)::Cvoid

"""
Runs a compilation pipeline.
Returns false and emits diagnostics on failure.
"""
@cenum iree_compiler_pipeline_t begin
    # IREE's full compilation pipeline.
    IREE_COMPILER_PIPELINE_STD = 0
    # Pipeline to translate a single hal.executable into a target-specific
    # binary form (such as an ELF file or a flatbuffer containing a SPIR-V
    # blob).
    IREE_COMPILER_PIPELINE_HAL_EXECUTABLE = 1
    # IREE's precompilation pipeline, which does input preprocessing and
    # pre-fusion global optimization.
    # This is experimental and this should be changed as we move to a more
    # cohesive approach for managing compilation phases.
    IREE_COMPILER_PIPELINE_PRECOMPILE = 2
end

ireeCompilerInvocationPipeline(inv, pipeline) =
    @cvcall libiree.ireeCompilerInvocationPipeline(inv::Ptr{iree_compiler_invocation_t},
        pipeline::iree_compiler_pipeline_t)::Bool

"""
Runs an arbitrary pass pipeline.
Returns false and emits diagnostics on failure.
Available since: 1.4
"""
ireeCompilerInvocationRunPassPipeline(inv, textPassPipeline) =
    @cvcall libiree.ireeCompilerInvocationRunPassPipeline(
        inv::Ptr{iree_compiler_invocation_t},
        textPassPipeline::Cstring,
    )::Bool

"""
Outputs the current compiler state as textual IR to the output.
"""
ireeCompilerInvocationOutputIR(inv, output) =
    @cvcall libiree.ireeCompilerInvocationOutputIR(inv::Ptr{iree_compiler_invocation_t},
        output::Ptr{iree_compiler_output_t})::Ptr{iree_compiler_error_t}

"""
Outputs the current compiler state as bytecode IR to the output.
Emits as the given bytecode version or most recent if -1.
Available since: 1.4
"""
ireeCompilerInvocationOutputIRBytecode(inv, output, bytecodeVersion) =
    @cvcall libiree.ireeCompilerInvocationOutputIRBytecode(inv::Ptr{iree_compiler_invocation_t},
        output::Ptr{iree_compiler_output_t},
        bytecodeVersion::Cint)::Ptr{iree_compiler_error_t}

"""
Assuming that the compiler has produced VM IR, converts it to bytecode
and outputs it. This is a valid next step after running the
IREE_COMPILER_PIPELINE_STD pipeline.
"""
ireeCompilerInvocationOutputVMBytecode(inv, output) =
    @cvcall libiree.ireeCompilerInvocationOutputVMBytecode(inv::Ptr{iree_compiler_invocation_t},
        output::Ptr{iree_compiler_output_t})::Ptr{iree_compiler_error_t}


"""
Assuming that the compiler has produced VM IR, converts it to textual
C source and output it. This is a valid next step after running the
IREE_COMPILER_PIPELINE_STD pipeline.
"""
ireeCompilerInvocationOutputVMCSource(inv, output) =
    @cvcall libiree.ireeCompilerInvocationOutputVMCSource(inv::Ptr{iree_compiler_invocation_t},
        output::Ptr{iree_compiler_output_t})::Ptr{iree_compiler_error_t}

"""
Outputs the contents of a single HAL executable as binary data.
This is a valid next step after running the
IREE_COMPILER_PIPELINE_HAL_EXECUTABLE pipeline.
"""
ireeCompilerInvocationOutputHALExecutable(inv, output) =
    @cvcall libiree.ireeCompilerInvocationOutputHALExecutable(inv::Ptr{iree_compiler_invocation_t},
        output::Ptr{iree_compiler_output_t})::Ptr{iree_compiler_error_t}

# Sources.
# Compilation sources are loaded into iree_compiler_source_t instances. These
# instances reference an originating session and may contain a concrete
# buffer of memory. Generally, when processing a source, its backing buffer
# will be transferred out from under it (i.e. sources are single-use).
# The actual source instance must be kept live until all processing is
# completed as some methods of loading a source involve maintaining
# references to backing resources.

"""
Destroy source instances.
"""
ireeCompilerSourceDestroy(source) = @cvcall libiree.ireeCompilerSourceDestroy(source::Ptr{iree_compiler_source_t})::Cvoid

"""
Opens the source from a file. This is used for normal text assembly file
sources.
Must be destroyed with ireeCompilerSourceDestroy().
"""
ireeCompilerSourceOpenFile(session, filePath, out_source) =
    @cvcall libiree.ireeCompilerSourceOpenFile(session::Ptr{iree_compiler_session_t},
        filePath::Cstring,
        out_source::Ptr{Ptr{iree_compiler_source_t}})::Ptr{iree_compiler_error_t}

"""
Wraps an existing buffer in memory.
If |isNullTerminated| is true, then the null must be accounted for in the
length. This is required for text buffers and it is permitted for binary
buffers.
Must be destroyed with ireeCompilerSourceDestroy().
"""

ireeCompilerSourceWrapBuffer(session, bufferName, buffer,
    length, isNullTerminated, out_source) =
    @cvcall libiree.ireeCompilerSourceWrapBuffer(session::Ptr{iree_compiler_session_t},
        bufferName::Cstring, buffer::Cstring,
        length::Csize_t, isNullTerminated::Bool,
        out_source::Ptr{Ptr{iree_compiler_source_t}})::Ptr{iree_compiler_error_t}

"""
Splits the current source buffer, invoking a callback for each "split"
within it. This is per the usual MLIR split rules (see
splitAndProcessBuffer): which split on `// -----`.
Both the original source and all yielded sources must be destroyed by the
caller eventually (split buffers are allowed to escape the callback).
"""
ireeCompilerSourceSplit(source, callback, userData) =
    @cvcall libiree.ireeCompilerSourceSplit(source::Ptr{iree_compiler_source_t},
        callback::Ptr{Cvoid},
        userData::Ptr{Cvoid})::Ptr{iree_compiler_error_t}

# Outputs.
# Compilation outputs are standalone instances that are used to collect
# final compilation artifacts. In their most basic form, they are just
# wrappers around an output stream over some file. However, more advanced
# things can be enabled via additional APIs (i.e. allocating efficient
# temporary file handles, etc).
#
# Outputs are not bound to a session as they can outlive it or be disconnected
# from the actual process of compilation in arbitrary ways.

"""
Destroy output instances.
"""
ireeCompilerOutputDestroy(output) = @cvcall libiree.ireeCompilerOutputDestroy(output::Ptr{iree_compiler_output_t})::Cvoid

"""
Opens a file for the output.
Must be destroyed via ireeCompilerOutputDestroy().
"""
ireeCompilerOutputOpenFile(filePath, out_output) =
    @cvcall libiree.ireeCompilerOutputOpenFile(filePath::Cstring, out_output::Ptr{Ptr{iree_compiler_output_t}})::Ptr{iree_compiler_error_t}

"""
Opens a file descriptor for output.
Must be destroyed via ireeCompilerOutputDestroy().
"""
ireeCompilerOutputOpenFD(fd, out_output) = @cvcall libiree.ireeCompilerOutputOpenFD(fd::Cint, out_output::Ptr{Ptr{iree_compiler_output_t}})::Ptr{iree_compiler_error_t}

"""
Opens an output to in-memory storage. Use the API
|ireeCompilerOutputMapMemory| to access the mapped contents once all
output has been written.
"""
ireeCompilerOutputOpenMembuffer(out_output) = @cvcall libiree.ireeCompilerOutputOpenMembuffer(out_output::Ptr{Ptr{iree_compiler_output_t}})::Ptr{iree_compiler_error_t}

"""
Maps the contents of a compiler output opened via
|ireeCompilerOutputOpenMembuffer|. This may be something obtained via
mmap or a more ordinary temporary buffer. This may fail in platform
specific ways unless if the output was created via
|ireeCompilerOutputOpenMembuffer|.
"""
ireeCompilerOutputMapMemory(output, contents, size) =
    @cvcall libiree.ireeCompilerOutputMapMemory(output::Ptr{iree_compiler_output_t}, contents::Ptr{Ptr{Cvoid}},
        size::Ptr{UInt64})::Ptr{iree_compiler_error_t}

"""
For file or other persistent outputs, by default they will be deleted on
|ireeCompilerOutputDestroy| (or exit). It is necessary to call
|ireeCompilerOutputKeep| in order to have them committed to their accessible
place.
"""
ireeCompilerOutputKeep(output) = @cvcall libiree.ireeCompilerOutputKeep(output::Ptr{iree_compiler_output_t})::Cvoid

"""
Writes arbitrary data to the output.
"""
ireeCompilerOutputWrite(output, data, length) =
    @cvcall libiree.ireeCompilerOutputWrite(output::Ptr{iree_compiler_output_t}, data::Ptr{Cvoid},
        length::Csize_t)::Ptr{iree_compiler_error_t}

### Mlir Interop

"""
Registers all dialects and extensions known to the IREE compiler.
"""
ireeCompilerRegisterDialects(registry) =
    @cvcall libiree.ireeCompilerRegisterDialects(registry::MlirDialectRegistry)::Cvoid

"""
Performs post-creation initialization of an externally derived context.
This configures things such as threading behavior. Dialect registration
is done via ireeCompilerRegisterDialects.
"""
ireeCompilerInitializeContext(context) =
    @cvcall libiree.ireeCompilerInitializeContext(context::MlirContext)::Cvoid

"""
Gets the MlirContext that the session manages. The context is owned by the
session and valid until it is destroyed.
This implicitly "activates" the session: make sure that any configuration
(flags, etc) has been done prior. Activation is lazy and is usually done
on the first use of the context (i.e. for parsing a source in an
invocation) but API access like this forces it.
Returns a NULL context if it has already been stolen or if activation fails.
"""
ireeCompilerSessionBorrowContext(session) =
    @cvcall libiree.ireeCompilerSessionBorrowContext(session::Ptr{iree_compiler_session_t})::MlirContext

"""
Gets the MlirContext that the session manages, releasing it for external
management of its lifetime. If the context has already been released,
then {nullptr} is returned. Upon return, it is up to the caller to destroy
the context and ensure that its lifetime extends at least as long as the
session remains in use.
This implicitly "activates" the session: make sure that any configuration
(flags, etc) has been done prior. Activation is lazy and is usually done
on the first use of the context (i.e. for parsing a source in an
invocation) but API access like this forces it.
Returns a NULL context if it has already been stolen or if activation fails.
"""
ireeCompilerSessionStealContext(session) =
    @cvcall libiree.ireeCompilerSessionStealContext(session::Ptr{iree_compiler_session_t})::MlirContext

"""
Same as |ireeCompilerInvocationStealModule| but ownership of the module
remains with the caller, who is responsible for ensuring that it is not
destroyed before the invocation is destroyed.
"""
ireeCompilerInvocationImportBorrowModule(inv, moduleOp) =
    @cvcall libiree.ireeCompilerInvocationImportBorrowModule(inv::Ptr{iree_compiler_invocation_t}, moduleOp::MlirOperation)::Bool


"""
Imports an externally built MlirModule into the invocation as an alternative
to parsing with |ireeCompilerInvocationParseSource|.
Ownership of the moduleOp is transferred to the invocation, regardless of
whether the call succeeds or fails.
On failure, returns false and issues diagnostics.
"""
ireeCompilerInvocationImportStealModule(inv, moduleOp) =
    @cvcall libiree.ireeCompilerInvocationImportStealModule(inv::Ptr{iree_compiler_invocation_t}, moduleOp::MlirOperation)::Bool

end # module Compiler
