module IREE

import Base.Libc.Libdl

const libiree = expanduser("~/Projects/iree-build/lib/libIREECompiler.so.0")
const libiree_runtime = (expanduser("~/Projects/iree-build/runtime/src/iree/runtime/libiree_runtime_runtime_shared.so"))

export
    CompilerOptions,
    Instance,
    InstanceOptions,
    SessionOptions,
    Session,
    Device,
    Call,
    BufferView

export
    append_bytecode!

include("./compiler.jl")
include("./runtime.jl")

import ..MLIR
using .Compiler:
    IreeCompilerOptions

const iree_registered_targets_and_passes = Ref(false)

function register_all_dialects!(context)
    if !iree_registered_targets_and_passes[]
        Compiler.ireeCompilerRegisterTargetBackends()
        Compiler.ireeCompilerRegisterAllPasses()
        iree_registered_targets_and_passes[] = true
    end

    Compiler.ireeCompilerRegisterAllDialects(context)

    context
end

mutable struct CompilerOptions
    options::IreeCompilerOptions

    CompilerOptions(options::IreeCompilerOptions) = begin
        finalizer(new(options)) do options
            Compiler.ireeCompilerOptionsDestroy(options.options)
        end
    end
end

CompilerOptions(flags) = begin
    if !iree_registered_targets_and_passes[]
        Compiler.ireeCompilerRegisterTargetBackends()
        Compiler.ireeCompilerRegisterAllPasses()
        iree_registered_targets_and_passes[] = true
    end

    options = CompilerOptions(Compiler.ireeCompilerOptionsCreate())

    io = IOBuffer()
    c_print_callback = @cfunction(MLIR.print_callback, Cvoid, (MLIR.MlirStringRef, Any))
    ref = Ref(io)

    result = GC.@preserve ref Compiler.ireeCompilerOptionsSetFlags(
        options,
        length(flags), flags,
        c_print_callback, Base.pointer_from_objref(ref),
    )

    if MLIR.LibMLIR.mlirLogicalResultIsFailure(result)
        msg = String(take!(io))
        throw("failed to create CompilerOptions: $msg")
    end

    return options
end

Base.convert(::Type{IreeCompilerOptions}, options::CompilerOptions) = options.options

### Pass Manager

build_vm_pass_pipeline!(op_pass, options) =
    Compiler.ireeCompilerBuildIREEVMPassPipeline(options, op_pass)

function translate_module_to_vm_bytecode(module_, options)
    io = IOBuffer()
    c_print_callback = @cfunction(MLIR.print_callback, Cvoid, (MLIR.MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref Compiler.ireeCompilerTranslateModuletoVMBytecode(
        options,
        MLIR.get_operation(module_),
        c_print_callback,
        ref,
    )
    take!(io)
end

## Runtime

using .Runtime:
    iree_hal_buffer_view_t,
    iree_runtime_instance_options_t,
    iree_hal_driver_registry_t,
    iree_runtime_instance_t,
    iree_hal_device_t,
    iree_string_view_t,
    iree_runtime_session_options_t,
    iree_runtime_session_t,
    iree_allocator_t,
    iree_vm_list_t,
    iree_status_t,
    iree_vm_value_t

Base.String(string_view::iree_string_view_t) =
    unsafe_string(Ptr{UInt8}(string_view.data), string_view.size)

### Status

mutable struct IREEException <: Exception
    status::iree_status_t

    IREEException(status) = begin
        finalizer(new(status)) do exception
            Runtime.iree_status_free(exception.status)
        end
    end
end

const IREE_STATUS_OK = C_NULL

function Base.show(io::IO, exception::IREEException)
    (; status) = exception
    @assert status != IREE_STATUS_OK

    out_buffer = Ref(Ptr{Cchar}())
    out_buffer_size = Ref(zero(Runtime.iree_host_size_t))
    allocator = Ref(iree_allocator_system())

    GC.@preserve out_buffer out_buffer_size allocator Runtime.iree_status_to_string(
        status,
        Base.pointer_from_objref(allocator),
        Base.pointer_from_objref(out_buffer),
        Base.pointer_from_objref(out_buffer_size),
    )

    if out_buffer[] != C_NULL && out_buffer_size[] > 0
        msg = unsafe_string(out_buffer[], out_buffer_size[])
        Runtime.iree_allocator_free(allocator[], out_buffer[])
        write(io, msg)
    end
end

function check_status(status::iree_status_t)
    if status == IREE_STATUS_OK
        return
    end
    throw(IREEException(status))
end

macro iree_check(ex)
    quote
        local status = $(esc(ex))
        check_status(status)
    end
end

### Runtime extensions

function iree_allocator_system()
    libiree_runtime_handle = Libdl.dlopen(libiree_runtime)
    iree_allocator_t(C_NULL, Libdl.dlsym(libiree_runtime_handle, :iree_allocator_system_ctl))
end

iree_allocator_null() =
    iree_allocator_t(Ptr{Cvoid}(), Runtime.iree_allocator_ctl_fn_t())

function iree_make_string_view(s)
    iree_string_view_t(
        Base.unsafe_convert(Ptr{Cchar}, s),
        sizeof(s)
    )
end

### Runtime Instance Options

struct InstanceOptions
    options::iree_runtime_instance_options_t
end

function InstanceOptions()
    iree_options = iree_runtime_instance_options_t(Ptr{iree_hal_driver_registry_t}())
    options = InstanceOptions(iree_options)
    Runtime.iree_runtime_instance_options_initialize(options)
    use_all_available_drivers!(options)
    @assert options.options.driver_registry != C_NULL "failed to initialize InstanceOptions"
    options
end

use_all_available_drivers!(options) =
    Runtime.iree_runtime_instance_options_use_all_available_drivers(options)

Base.unsafe_convert(::Type{Ptr{iree_runtime_instance_options_t}}, options::InstanceOptions) =
    Base.unsafe_convert(Ptr{iree_runtime_instance_options_t}, Base.pointer_from_objref(options.options))

### String view

Base.convert(::Type{iree_string_view_t}, str::String) =
    iree_string_view_t(Base.unsafe_convert(Ptr{Cchar}, str), sizeof(str))

### Runtime Instance

mutable struct Instance
    instance::Ptr{iree_runtime_instance_t}
end

function Instance(options::InstanceOptions=InstanceOptions())
    instance = Instance(Ptr{iree_runtime_instance_t}())
    @iree_check begin
        GC.@preserve instance Runtime.iree_runtime_instance_create(
            options,
            iree_allocator_system(),
            Base.unsafe_convert(Ptr{Ptr{iree_runtime_instance_t}}, Base.pointer_from_objref(instance)),
        )
    end
    @assert instance.instance != C_NULL "failed to create Instance"
    instance
end

Base.unsafe_convert(::Type{Ptr{iree_runtime_instance_t}}, instance::Instance) = instance.instance

### Runtime Device

mutable struct Device
    device::Ptr{iree_hal_device_t}

    Device(device) = finalizer(Runtime.iree_hal_device_release, new(device))
end

function Device(instance::Instance, driver_name="local-task")
    device = Device(Ptr{iree_hal_device_t}())
    @iree_check begin
        GC.@preserve device Runtime.iree_runtime_instance_try_create_default_device(
            instance,
            driver_name,
            Base.unsafe_convert(Ptr{Ptr{iree_runtime_instance_t}}, Base.pointer_from_objref(device)),
        )
    end
    @assert device.device != C_NULL "failed to create Device"
    device
end

Base.unsafe_convert(::Type{Ptr{iree_hal_device_t}}, device::Device) = device.device

device_id(device) = Runtime.iree_hal_device_id(device)

### Session Options

mutable struct SessionOptions
    options::iree_runtime_session_options_t
end

function SessionOptions()
    options = SessionOptions(iree_runtime_session_options_t(
        zero(Runtime.iree_vm_context_flags_t),
        zero(Runtime.iree_runtime_session_builtins_t)
    ))
    Runtime.iree_runtime_session_options_initialize(options)
    options
end

Base.unsafe_convert(::Type{Ptr{iree_runtime_session_options_t}}, options::SessionOptions) =
    Base.unsafe_convert(Ptr{iree_runtime_session_options_t}, Base.pointer_from_objref(options.options))


### Session

mutable struct Session
    session::Ptr{iree_runtime_session_t}
    bytecodes::Vector{Vector{UInt8}}

    Session(session) = begin
        finalizer(new(session, [])) do session
            Runtime.iree_runtime_session_release(session.session)
        end
    end
end

function Session(instance, options, device)
    session = Session(Ptr{iree_runtime_session_t}())
    allocator = Runtime.iree_runtime_instance_host_allocator(instance)
    @iree_check begin
        GC.@preserve session Runtime.iree_runtime_session_create_with_device(
            instance,
            options,
            device,
            allocator,
            Base.pointer_from_objref(session),
        )
    end
    @assert session.session != C_NULL "failed to create Session"
    session
end

Base.unsafe_convert(::Type{Ptr{iree_runtime_session_t}}, session::Session) =
    session.session

function append_bytecode!(session, bytecode)
    bytecode = collect(bytecode)
    push!(session.bytecodes, bytecode) # keep the buffer around

    @iree_check begin
        GC.@preserve bytecode Runtime.iree_runtime_session_append_bytecode_module_from_memory(
            session,
            Runtime.iree_const_byte_span_t(
                Base.pointer(bytecode), length(bytecode)
            ),
            iree_allocator_null()
        )
    end

    session
end

### Runtime Call

mutable struct iree_vm_module_t end

struct iree_vm_function_t
    module_::Ptr{iree_vm_module_t}
    linkage::UInt16
    ordinal::UInt16
end

mutable struct iree_runtime_call_t
    session::Ptr{iree_runtime_session_t}
    function_::iree_vm_function_t
    inputs::Ptr{iree_vm_list_t}
    outputs::Ptr{iree_vm_list_t}
end

iree_vm_function_signature(func) = @ccall libiree_runtime.iree_vm_function_signature(func::Ptr{iree_vm_function_t})::Runtime.iree_vm_function_signature_t

iree_runtime_call_deinitialize(call) =
    @ccall libiree_runtime.iree_runtime_call_deinitialize(call::Ptr{iree_runtime_call_t})::Cvoid
iree_runtime_call_initialize_by_name(session, full_name, out_call) =
    @ccall libiree_runtime.iree_runtime_call_initialize_by_name(
        session::Ptr{iree_runtime_session_t},
        full_name::iree_string_view_t,
        out_call::Ptr{iree_runtime_call_t},
    )::iree_status_t
iree_runtime_call_invoke(call, flags) =
    @ccall libiree_runtime.iree_runtime_call_invoke(call::Ptr{iree_runtime_call_t}, flags::Runtime.iree_runtime_call_flags_t)::iree_status_t
iree_runtime_call_reset(call) =
    @ccall libiree_runtime.iree_runtime_call_reset(call::Ptr{iree_runtime_call_t})::Cvoid
iree_runtime_call_inputs_push_back_buffer_view(call, buffer_view) =
    @ccall libiree_runtime.iree_runtime_call_inputs_push_back_buffer_view(
        call::Ptr{iree_runtime_call_t},
        buffer_view::Ptr{iree_hal_buffer_view_t},
    )::iree_status_t
iree_runtime_call_outputs_pop_front_buffer_view(call, ret) =
    @ccall libiree_runtime.iree_runtime_call_outputs_pop_front_buffer_view(call::Ptr{iree_runtime_call_t}, ret::Ptr{Ptr{iree_hal_buffer_view_t}})::iree_status_t

### Call

"""
    Call(session::Session, full_name::String)

A Call struct is a callable referencing a function inside the module. Therefore,
the provided full name must start with the prefix "module.". Call arguments can
be passed either as values (primitive types) or as references (`AbstractArray`
which are then turned into `BufferView`).
"""
struct Call
    call::iree_runtime_call_t
    session::Session

    Call(call, session) = begin
        new(finalizer(call) do call
                iree_runtime_call_deinitialize(
                    Base.pointer_from_objref(call)
                )
            end, session)
    end
end

function Base.show(io::IO, call::Call)
    arguments_count, results_count = get_arity(call)
    print(io, "Call(#= $arguments_count => $results_count =#)")
end

function Call(session::Session, full_name)
    out_call = iree_runtime_call_t(
        Ptr{iree_runtime_session_t}(),
        iree_vm_function_t(Ptr{iree_vm_module_t}(), zero(UInt16), zero(UInt16)),
        Ptr{iree_vm_list_t}(),
        Ptr{iree_vm_list_t}(),
    )
    @iree_check begin
        GC.@preserve out_call iree_runtime_call_initialize_by_name(
            session,
            full_name,
            Base.unsafe_convert(Ptr{iree_runtime_call_t}, Base.pointer_from_objref(out_call)),
        )
    end
    @assert out_call.session == session.session
    Call(out_call, session)
end

Base.unsafe_convert(::Type{Ptr{iree_runtime_call_t}}, call::Call) =
    Ptr{iree_runtime_call_t}(Base.pointer_from_objref(call.call))

"""
    get_arity(::Call)::Tuple{Int,Int}

Returns the argument and result counts for this `Call`.

See also `calling_convention(::Call)`
"""
function get_arity(call)
    GC.@preserve call begin
        call_ptr = Ptr{Cvoid}(Base.unsafe_convert(Ptr{iree_runtime_call_t}, call))
        signature = iree_vm_function_signature(
            Ptr{iree_vm_function_t}(call_ptr + sizeof(Ptr{Cvoid})))
    end

    signature = Ref(signature)
    out_argument_count = Ref(zero(Runtime.iree_host_size_t))
    out_result_count = Ref(zero(Runtime.iree_host_size_t))

    @iree_check GC.@preserve signature out_argument_count out_result_count begin
        Runtime.iree_vm_function_call_count_arguments_and_results(
            Base.unsafe_convert(Ptr{Runtime.iree_vm_function_signature_t}, Base.pointer_from_objref(signature)),
            Base.unsafe_convert(Ptr{Runtime.iree_host_size_t}, Base.pointer_from_objref(out_argument_count)),
            Base.unsafe_convert(Ptr{Runtime.iree_host_size_t}, Base.pointer_from_objref(out_result_count)),
        )
    end

    (out_argument_count[], out_result_count[])
end

"""
    calling_convention(call::Call)::Pair{Vector{Char},Vector{Char}}

Returns details about the expected way to pass or retrieve arguments from the provided
runtime call. `IREE.IREE_RUNTIME_CALLING_CONVENTION_REF` can be used to check wether or not
an argument should be passed as a reference or as a value.
"""
function calling_convention(call)
    cconv = GC.@preserve call begin
        call_ptr = Ptr{Cvoid}(Base.unsafe_convert(Ptr{iree_runtime_call_t}, call))
        signature = iree_vm_function_signature(
            Ptr{iree_vm_function_t}(call_ptr + sizeof(Ptr{Cvoid})))
        String(signature.calling_convention)
    end

    args, results = split(chopprefix(cconv, r"\d+"), '_')
    collect(args) => collect(results)
end

iree_hal_buffer_view_type_id() = @ccall libiree_runtime.iree_hal_buffer_view_type_id()::Runtime.iree_vm_ref_type_t

function set_ref!(list, i, view)
    value = Ref(Runtime.iree_vm_ref_t(Tuple(zero(UInt8) for _ in 1:16)))
    @iree_check GC.@preserve value Runtime.iree_vm_ref_wrap_assign(
        view, iree_hal_buffer_view_type_id(),
        Base.pointer_from_objref(value),
    )

    ref_type_name = String(Runtime.iree_vm_ref_type_name(value[].type))

    @iree_check GC.@preserve value Runtime.iree_vm_list_set_ref_retain(
        list, i - 1, Base.pointer_from_objref(value)
    )
end

const IREE_RUNTIME_CALLING_CONVENTION_REF = Char('r')

function (call::Call)(args...)
    args_count, results_count = get_arity(call)
    args_cconv, results_cconv = calling_convention(call)
    @assert length(args) == args_count "expected $(args_count) arguments but got $(length(args))"

    @iree_check Runtime.iree_vm_list_resize(call.call.inputs, length(args))

    # Push args
    for (i, (arg, arg_cconv)) in enumerate(zip(args, args_cconv))
        if arg isa AbstractArray && arg_cconv == IREE_RUNTIME_CALLING_CONVENTION_REF
            set_ref!(
                call.call.inputs, i,
                BufferView(call.session, arg),
            )
        elseif arg_cconv != IREE_RUNTIME_CALLING_CONVENTION_REF
            in_value = Ref(to_vm_value(arg))
            @iree_check GC.@preserve in_value begin
                Runtime.iree_vm_list_set_value(
                    call.call.inputs,
                    i - 1,
                    Base.pointer_from_objref(in_value),
                )
            end
        else
            expected_type = arg_cconv == IREE_RUNTIME_CALLING_CONVENTION_REF ?
                            "AbstractArray" : "Value"
            throw("invalid arg #$i type $(typeof(arg)) expected $expected_type")
        end
    end
    @assert Runtime.iree_vm_list_size(call.call.inputs) == args_count == length(args)

    # Invoke
    @iree_check iree_runtime_call_invoke(call, zero(Runtime.iree_runtime_call_flags_t))

    result_size = Runtime.iree_vm_list_size(call.call.outputs)
    @assert results_count == result_size

    # Pull return values
    res = map(enumerate(Iterators.reverse(results_cconv))) do (i, result_cconv)
        if result_cconv == IREE_RUNTIME_CALLING_CONVENTION_REF
            ret = Ref(Ptr{iree_hal_buffer_view_t}())
            @iree_check GC.@preserve ret begin
                iree_runtime_call_outputs_pop_front_buffer_view(
                    call, Base.pointer_from_objref(ret))
            end
            buffer_view = ret[]
            N = Runtime.iree_hal_buffer_view_shape_rank(buffer_view)
            iree_type = Runtime.iree_hal_buffer_view_element_type(buffer_view)
            T = julia_type(iree_type)
            return BufferView{T,N}(buffer_view, call.session)
        end

        out_value = Ref(iree_vm_value_t(Tuple(zero(UInt8) for _ in 1:16)))
        @iree_check begin
            GC.@preserve out_value Runtime.iree_vm_list_get_value(
                call.call.outputs,
                i - 1,
                Base.pointer_from_objref(out_value),
            )
        end
        val = out_value[]

        from_vm_value(val)
    end

    iree_runtime_call_reset(call)

    if isempty(res)
        nothing
    elseif length(res) == 1
        first(res)
    else
        Tuple(res)
    end
end

### IREE Buffer View

"""
    BufferView(session::Session, a::AbstractArray)
    BufferView{T,N} <: AbstractArray{T,N}

A BufferView copies the data to an iree owned data buffer which can then be used
as a parameter to runtime calls when the calling convention requires a reference
to a buffer view.
"""
mutable struct BufferView{T,N} <: AbstractArray{T,N}
    view::Ptr{iree_hal_buffer_view_t}
    session::Session

    BufferView{T,N}(view, session) where {T,N} = begin
        @assert view != C_NULL
        finalizer(new{T,N}(view, session)) do view
            Runtime.iree_hal_buffer_view_release(view.view)
        end
    end
end

using .Runtime:
    # IREE_HAL_ELEMENT_TYPE_INT_4
    # IREE_HAL_ELEMENT_TYPE_SINT_4
    # IREE_HAL_ELEMENT_TYPE_UINT_4
    IREE_HAL_ELEMENT_TYPE_BOOL_8,
    IREE_HAL_ELEMENT_TYPE_INT_8,
    IREE_HAL_ELEMENT_TYPE_SINT_8,
    IREE_HAL_ELEMENT_TYPE_UINT_8,
    IREE_HAL_ELEMENT_TYPE_INT_16,
    IREE_HAL_ELEMENT_TYPE_SINT_16,
    IREE_HAL_ELEMENT_TYPE_UINT_16,
    IREE_HAL_ELEMENT_TYPE_INT_32,
    IREE_HAL_ELEMENT_TYPE_SINT_32,
    IREE_HAL_ELEMENT_TYPE_UINT_32,
    IREE_HAL_ELEMENT_TYPE_INT_64,
    IREE_HAL_ELEMENT_TYPE_SINT_64,
    IREE_HAL_ELEMENT_TYPE_UINT_64,
    IREE_HAL_ELEMENT_TYPE_FLOAT_16,
    IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    IREE_HAL_ELEMENT_TYPE_FLOAT_64

function julia_type(T)
    T == IREE_HAL_ELEMENT_TYPE_BOOL_8 && return Bool
    # IREE_HAL_ELEMENT_TYPE_INT_4
    # IREE_HAL_ELEMENT_TYPE_SINT_4
    # IREE_HAL_ELEMENT_TYPE_UINT_4
    T == IREE_HAL_ELEMENT_TYPE_INT_8 && return Int8
    T == IREE_HAL_ELEMENT_TYPE_SINT_8 && return Int8
    T == IREE_HAL_ELEMENT_TYPE_UINT_8 && return UInt8
    T == IREE_HAL_ELEMENT_TYPE_INT_16 && return Int16
    T == IREE_HAL_ELEMENT_TYPE_SINT_16 && return Int16
    T == IREE_HAL_ELEMENT_TYPE_UINT_16 && return UInt16
    T == IREE_HAL_ELEMENT_TYPE_INT_32 && return Int32
    T == IREE_HAL_ELEMENT_TYPE_SINT_32 && return Int32
    T == IREE_HAL_ELEMENT_TYPE_UINT_32 && return UInt32
    T == IREE_HAL_ELEMENT_TYPE_INT_64 && return Int64
    T == IREE_HAL_ELEMENT_TYPE_SINT_64 && return Int64
    T == IREE_HAL_ELEMENT_TYPE_UINT_64 && return UInt64
    T == IREE_HAL_ELEMENT_TYPE_FLOAT_16 && return Float16
    T == IREE_HAL_ELEMENT_TYPE_FLOAT_32 && return Float32
    T == IREE_HAL_ELEMENT_TYPE_FLOAT_64 && return Float64
    throw("invalid IREE type $T")
end

function iree_type(T)
    T == Bool && return IREE_HAL_ELEMENT_TYPE_BOOL_8
    # IREE_HAL_ELEMENT_TYPE_INT_4
    # IREE_HAL_ELEMENT_TYPE_SINT_4
    # IREE_HAL_ELEMENT_TYPE_UINT_4
    T == Int8 && return IREE_HAL_ELEMENT_TYPE_INT_8
    T == Int8 && return IREE_HAL_ELEMENT_TYPE_SINT_8
    T == UInt8 && return IREE_HAL_ELEMENT_TYPE_UINT_8
    T == Int16 && return IREE_HAL_ELEMENT_TYPE_INT_16
    T == Int16 && return IREE_HAL_ELEMENT_TYPE_SINT_16
    T == UInt16 && return IREE_HAL_ELEMENT_TYPE_UINT_16
    T == Int32 && return IREE_HAL_ELEMENT_TYPE_INT_32
    T == Int32 && return IREE_HAL_ELEMENT_TYPE_SINT_32
    T == UInt32 && return IREE_HAL_ELEMENT_TYPE_UINT_32
    T == Int64 && return IREE_HAL_ELEMENT_TYPE_INT_64
    T == Int64 && return IREE_HAL_ELEMENT_TYPE_SINT_64
    T == UInt64 && return IREE_HAL_ELEMENT_TYPE_UINT_64
    T == Float16 && return IREE_HAL_ELEMENT_TYPE_FLOAT_16
    T == Float32 && return IREE_HAL_ELEMENT_TYPE_FLOAT_32
    T == Float64 && return IREE_HAL_ELEMENT_TYPE_FLOAT_64
    throw("invalid IREE type $T")
end

mutable struct iree_hal_buffer_t end

function row_majorize(a::AbstractArray{T,N}) where {T,N}
    strides = zeros(Int, N)
    prev = 1
    for i in N:-1:1
        strides[i] = prev
        prev *= size(a,i)
    end

    out = similar(a)
    for I in CartesianIndices(a)
        offset = 1 + sum(strides .* (Tuple(I) .- 1))
        out[offset] = a[I]
    end
    return out
end

BufferView(_, view::BufferView) = view
function BufferView(session, a::AbstractArray{T,N}) where {T,N}
    # Unfortunately, IREE does not support IREE_HAL_ENCODING_TYPE_DENSE_COLUMN_MAJOR
    # we therefore have to load the buffers in such a way that the data is in row major
    row_major_a = N > 1 ?
        row_majorize(a) : a

    out_buffer = Ref(Ptr{iree_hal_buffer_view_t}())

    @iree_check GC.@preserve a out_buffer Runtime.iree_hal_buffer_view_allocate_buffer(
        Runtime.iree_runtime_session_device_allocator(session),
        N, collect(size(row_major_a)),
        iree_type(T), Runtime.IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        Runtime.iree_hal_buffer_params_t(
            Runtime.IREE_HAL_BUFFER_USAGE_DEFAULT | Runtime.IREE_HAL_BUFFER_USAGE_MAPPING,
            Runtime.IREE_HAL_MEMORY_ACCESS_READ,
            Runtime.IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            zero(Runtime.iree_hal_queue_affinity_t),
            zero(Runtime.iree_device_size_t),
        ),
        Runtime.iree_const_byte_span_t(pointer(row_major_a), sizeof(row_major_a)),
        Base.unsafe_convert(Ptr{Ptr{Runtime.iree_hal_buffer_view_t}}, Base.pointer_from_objref(out_buffer),)
    )

    BufferView{T,N}(out_buffer[], session)
end

function Base.size(view::BufferView)
    ndims = Runtime.iree_hal_buffer_view_shape_rank(view)
    dims_ptr = Runtime.iree_hal_buffer_view_shape_dims(view)
    dims = unsafe_wrap(Array, dims_ptr, ndims; own=false)
    Tuple(dims)
end

iree_hal_buffer_view_append_to_builder(view, max_element_count, builder) =
    @ccall libiree_runtime.iree_hal_buffer_view_append_to_builder(
        view::Ptr{iree_hal_buffer_view_t},
        max_element_count::Runtime.iree_host_size_t,
        builder::Ptr{Runtime.iree_string_builder_t},
    )::iree_status_t
iree_hal_buffer_view_fprint(fd, view, max_element_count, allocator) =
    @ccall libiree_runtime.iree_hal_buffer_view_fprint(
        fd::Ptr{Cvoid},
        view::Ptr{iree_hal_buffer_view_t},
        max_element_count::Runtime.iree_host_size_t,
        allocator::iree_allocator_t,
    )::iree_status_t
iree_hal_buffer_view_buffer(view) =
    @ccall libiree_runtime.iree_hal_buffer_view_buffer(view::Ptr{iree_hal_buffer_view_t})::Ptr{Cvoid}
iree_hal_buffer_map_read(buffer, offset, target_buffer, data_length) =
    @ccall libiree_runtime.iree_hal_buffer_map_read(
        buffer::Ptr{iree_hal_buffer_t}, offset::Runtime.iree_device_size_t,
        target_buffer::Ptr{Cvoid}, data_length::Runtime.iree_device_size_t)::iree_status_t
iree_hal_buffer_allocated_buffer(buffer) =
    @ccall libiree_runtime.iree_hal_buffer_allocated_buffer(buffer::Ptr{iree_hal_buffer_t})::Ptr{iree_hal_buffer_t}

"Util to bypass the default Base.show implemented for arrays"
function dump_view(buffer_view::Ptr{iree_hal_buffer_view_t})
    builder = Runtime.iree_string_builder_t(
        iree_allocator_system(),
        Cstring(Ptr{UInt8}()),
        zero(Runtime.iree_host_size_t),
        zero(Runtime.iree_host_size_t),
    )
    @iree_check GC.@preserve builder iree_hal_buffer_view_append_to_builder(
        buffer_view, typemax(Runtime.iree_host_size_t),
        Base.pointer_from_objref(builder),
    )
    str = unsafe_string(Ptr{UInt8}(builder.buffer), builder.size)
    GC.@preserve builder Runtime.iree_string_builder_deinitialize(
        Base.pointer_from_objref(builder),
    )
    write(stdout, str)
    nothing
end

Base.getindex(view::BufferView, I::CartesianIndex) = Base.getindex(view, Tuple(I)...)
function Base.getindex(view::BufferView{T,N}, I...) where {T,N}
    Base.checkbounds(view, I...)

    buffer = Base.unsafe_convert(
        Ptr{T},
        iree_hal_buffer_view_buffer(view),
    )
    @assert buffer != C_NULL

    n_dims = ndims(view)
    indices = [
        Runtime.iree_hal_dim_t(idx - 1)
        for (i, idx) in enumerate(I)
        if i <= n_dims
    ]

    offset = Ref(zero(Runtime.iree_device_size_t))
    @iree_check GC.@preserve offset Runtime.iree_hal_buffer_view_compute_offset(
        view, length(indices), indices,
        Base.unsafe_convert(Ptr{Runtime.iree_device_size_t},
            Base.pointer_from_objref(offset)),
    )

    val = [zero(T)]
    @iree_check GC.@preserve val iree_hal_buffer_map_read(
        buffer, offset[],
        pointer(val),
        sizeof(T),
    )

    only(val)
end

Base.unsafe_convert(::Type{Ptr{T}}, view::BufferView) where {T<:Union{Cvoid,iree_hal_buffer_view_t}} = Ptr{T}(view.view)

function vm_value_with_type(f, type)
    value = Ref(iree_vm_value_t(Tuple(zero(UInt8) for _ in 1:16)))
    GC.@preserve value begin
        val_ptr = Ptr{iree_vm_value_t}(Base.pointer_from_objref(value))
        val_ptr.type = type
        f(val_ptr)
    end
    value[]
end

using .Runtime:
    IREE_VM_VALUE_TYPE_NONE,
    IREE_VM_VALUE_TYPE_I8,
    IREE_VM_VALUE_TYPE_I16,
    IREE_VM_VALUE_TYPE_I32,
    IREE_VM_VALUE_TYPE_I64,
    IREE_VM_VALUE_TYPE_F32,
    IREE_VM_VALUE_TYPE_F64

to_vm_value(::Nothing) = vm_value_with_type(_ -> nothing, IREE_VM_VALUE_TYPE_NONE)
to_vm_value(i8::Int8) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_I8) do val
        val.i8 = i8
    end
to_vm_value(i16::Int16) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_I16) do val
        val.i16 = i16
    end
to_vm_value(i32::Int32) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_I32) do val
        val.i32 = i32
    end
to_vm_value(i64::Int64) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_I64) do val
        val.i64 = i64
    end
to_vm_value(f32::Float32) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_F32) do val
        val.f32 = f32
    end
to_vm_value(f64::Float64) =
    vm_value_with_type(IREE_VM_VALUE_TYPE_F64) do val
        val.f64 = f64
    end
to_vm_value(::T) where {T} = throw("cannot convert value of type $T to a iree_vm_value_t")

function from_vm_value(val)
    if val.type == IREE_VM_VALUE_TYPE_NONE
        nothing
    elseif val.type == IREE_VM_VALUE_TYPE_I8
        val.i8
    elseif val.type == IREE_VM_VALUE_TYPE_I16
        val.i16
    elseif val.type == IREE_VM_VALUE_TYPE_I32
        val.i32
    elseif val.type == IREE_VM_VALUE_TYPE_I64
        val.i64
    elseif val.type == IREE_VM_VALUE_TYPE_F32
        val.f32
    elseif val.type == IREE_VM_VALUE_TYPE_F64
        val.f64
    else
        throw("unsupported result type $(val.type)")
    end
end

end # module IREE
