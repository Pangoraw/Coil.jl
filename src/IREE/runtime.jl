module Runtime

using CEnum

using ..IREE: libiree_runtime as libiree

const intptr_t = Clong

const iree_host_size_t = signed(Csize_t)

const iree_device_size_t = signed(Csize_t)

function iree_host_align(value, alignment)
    ccall((:iree_host_align, libiree), iree_host_size_t, (iree_host_size_t, iree_host_size_t), value, alignment)
end

struct iree_string_view_t
    data::Cstring
    size::iree_host_size_t
end

function iree_make_cstring_view(str)
    ccall((:iree_make_cstring_view, libiree), iree_string_view_t, (Cstring,), str)
end

mutable struct iree_status_handle_t end

const iree_status_t = Ptr{iree_status_handle_t}

@cenum iree_status_code_e::UInt32 begin
    IREE_STATUS_OK = 0
    IREE_STATUS_CANCELLED = 1
    IREE_STATUS_UNKNOWN = 2
    IREE_STATUS_INVALID_ARGUMENT = 3
    IREE_STATUS_DEADLINE_EXCEEDED = 4
    IREE_STATUS_NOT_FOUND = 5
    IREE_STATUS_ALREADY_EXISTS = 6
    IREE_STATUS_PERMISSION_DENIED = 7
    IREE_STATUS_RESOURCE_EXHAUSTED = 8
    IREE_STATUS_FAILED_PRECONDITION = 9
    IREE_STATUS_ABORTED = 10
    IREE_STATUS_OUT_OF_RANGE = 11
    IREE_STATUS_UNIMPLEMENTED = 12
    IREE_STATUS_INTERNAL = 13
    IREE_STATUS_UNAVAILABLE = 14
    IREE_STATUS_DATA_LOSS = 15
    IREE_STATUS_UNAUTHENTICATED = 16
    IREE_STATUS_DEFERRED = 17
    IREE_STATUS_CODE_MASK = 31
end

const iree_status_code_t = iree_status_code_e

function iree_status_allocate(code, file, line, message)
    ccall((:iree_status_allocate, libiree), iree_status_t, (iree_status_code_t, Cstring, UInt32, iree_string_view_t), code, file, line, message)
end

function iree_string_view_empty()
    ccall((:iree_string_view_empty, libiree), iree_string_view_t, ())
end

function iree_status_annotate(base_status, message)
    ccall((:iree_status_annotate, libiree), iree_status_t, (iree_status_t, iree_string_view_t), base_status, message)
end

function iree_status_ignore(status)
    ccall((:iree_status_ignore, libiree), iree_status_t, (iree_status_t,), status)
end

function iree_status_abort(status)
    ccall((:iree_status_abort, libiree), Cvoid, (iree_status_t,), status)
end

function iree_abort()
    ccall((:iree_abort, libiree), Cvoid, ())
end

# no prototype is found for this function at alignment.h:34:1, please use with caution
function static_assert()
    ccall((:static_assert, libiree), Cint, ())
end

function iree_hal_resource_is(resource, vtable)
    ccall((:iree_hal_resource_is, libiree), Bool, (Ptr{Cvoid}, Ptr{Cvoid}), resource, vtable)
end

const iree_hal_queue_affinity_t = Int64

const iree_hal_element_type_t = UInt32

const iree_hal_numerical_type_t = UInt8

const iree_vm_ref_type_t = UInt32

struct iree_vm_ref_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_ref_t}, f::Symbol)
    f === :ptr && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :offsetof_counter && return (Ptr{UInt32}(x + 8), 0, 8)
    f === :type && return (Ptr{iree_vm_ref_type_t}(x + 8), 8, 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_ref_t, f::Symbol)
    r = Ref{iree_vm_ref_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_ref_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{iree_vm_ref_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

# typedef void ( IREE_API_PTR * iree_vm_ref_destroy_t ) ( void * ptr )
const iree_vm_ref_destroy_t = Ptr{Cvoid}

struct iree_vm_ref_type_descriptor_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_ref_type_descriptor_t}, f::Symbol)
    f === :destroy && return Ptr{iree_vm_ref_destroy_t}(x + 0)
    f === :offsetof_counter && return (Ptr{UInt32}(x + 8), 0, 8)
    f === :type && return (Ptr{iree_vm_ref_type_t}(x + 8), 8, 24)
    f === :type_name && return Ptr{iree_string_view_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_ref_type_descriptor_t, f::Symbol)
    r = Ref{iree_vm_ref_type_descriptor_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_ref_type_descriptor_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{iree_vm_ref_type_descriptor_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

function iree_vm_ref_wrap_retain(ptr, type, out_ref)
    ccall((:iree_vm_ref_wrap_retain, libiree), iree_status_t, (Ptr{Cvoid}, iree_vm_ref_type_t, Ptr{iree_vm_ref_t}), ptr, type, out_ref)
end

function iree_vm_ref_wrap_assign(ptr, type, out_ref)
    ccall((:iree_vm_ref_wrap_assign, libiree), iree_status_t, (Ptr{Cvoid}, iree_vm_ref_type_t, Ptr{iree_vm_ref_t}), ptr, type, out_ref)
end

function iree_vm_ref_check(ref, type)
    ccall((:iree_vm_ref_check, libiree), iree_status_t, (iree_vm_ref_t, iree_vm_ref_type_t), ref, type)
end

const iree_zone_id_t = UInt32

struct iree_byte_span_t
    data::Ptr{UInt8}
    data_length::iree_host_size_t
end

function iree_make_byte_span(data, data_length)
    ccall((:iree_make_byte_span, libiree), iree_byte_span_t, (Ptr{Cvoid}, iree_host_size_t), data, data_length)
end

mutable struct iree_vm_stack_t end

const iree_vm_invocation_flags_t = UInt32

mutable struct iree_vm_state_resolver_t
    self::Ptr{Cvoid}
    query_module_state::Ptr{Cvoid}
end

# typedef iree_status_t ( IREE_API_PTR * iree_allocator_ctl_fn_t ) ( void * self , iree_allocator_command_t command , const void * params , void * * inout_ptr )
const iree_allocator_ctl_fn_t = Ptr{Cvoid}

struct iree_allocator_t
    self::Ptr{Cvoid}
    ctl::iree_allocator_ctl_fn_t
end

function iree_vm_stack_initialize(storage, flags, state_resolver, allocator, out_stack)
    ccall((:iree_vm_stack_initialize, libiree), iree_status_t, (iree_byte_span_t, iree_vm_invocation_flags_t, iree_vm_state_resolver_t, iree_allocator_t, Ptr{Ptr{iree_vm_stack_t}}), storage, flags, state_resolver, allocator, out_stack)
end

function iree_vm_stack_annotate_backtrace(stack, base_status)
    ccall((:iree_vm_stack_annotate_backtrace, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_status_t), stack, base_status)
end

const iree_vm_native_function_flags_t = UInt32

# typedef iree_status_t ( IREE_API_PTR * iree_vm_native_function_target2_t ) ( iree_vm_stack_t * IREE_RESTRICT stack , void * IREE_RESTRICT module , void * IREE_RESTRICT module_state , const void * IREE_RESTRICT args , void * IREE_RESTRICT rets )
const iree_vm_native_function_target2_t = Ptr{Cvoid}

function iree_host_size_has_alignment(value, alignment)
    ccall((:iree_host_size_has_alignment, libiree), Bool, (iree_host_size_t, iree_host_size_t), value, alignment)
end

function iree_device_align(value, alignment)
    ccall((:iree_device_align, libiree), iree_device_size_t, (iree_device_size_t, iree_device_size_t), value, alignment)
end

function iree_device_size_has_alignment(value, alignment)
    ccall((:iree_device_size_has_alignment, libiree), Bool, (iree_device_size_t, iree_device_size_t), value, alignment)
end

function iree_make_string_view(str, str_length)
    ccall((:iree_make_string_view, libiree), iree_string_view_t, (Cstring, iree_host_size_t), str, str_length)
end

struct iree_string_pair_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_string_pair_t}, f::Symbol)
    f === :first && return Ptr{iree_string_view_t}(x + 0)
    f === :key && return Ptr{iree_string_view_t}(x + 0)
    f === :second && return Ptr{iree_string_view_t}(x + 16)
    f === :value && return Ptr{iree_string_view_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_string_pair_t, f::Symbol)
    r = Ref{iree_string_pair_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_string_pair_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_string_pair_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_string_pair_empty()
    ccall((:iree_string_pair_empty, libiree), iree_string_pair_t, ())
end

function iree_make_string_pair(first, second)
    ccall((:iree_make_string_pair, libiree), iree_string_pair_t, (iree_string_view_t, iree_string_view_t), first, second)
end

function iree_make_cstring_pair(first, second)
    ccall((:iree_make_cstring_pair, libiree), iree_string_pair_t, (Cstring, Cstring), first, second)
end

function iree_string_view_equal(lhs, rhs)
    ccall((:iree_string_view_equal, libiree), Bool, (iree_string_view_t, iree_string_view_t), lhs, rhs)
end

function iree_string_view_compare(lhs, rhs)
    ccall((:iree_string_view_compare, libiree), Cint, (iree_string_view_t, iree_string_view_t), lhs, rhs)
end

function iree_string_view_find_char(value, c, pos)
    ccall((:iree_string_view_find_char, libiree), iree_host_size_t, (iree_string_view_t, Cchar, iree_host_size_t), value, c, pos)
end

function iree_string_view_find_first_of(value, s, pos)
    ccall((:iree_string_view_find_first_of, libiree), iree_host_size_t, (iree_string_view_t, iree_string_view_t, iree_host_size_t), value, s, pos)
end

function iree_string_view_find_last_of(value, s, pos)
    ccall((:iree_string_view_find_last_of, libiree), iree_host_size_t, (iree_string_view_t, iree_string_view_t, iree_host_size_t), value, s, pos)
end

function iree_string_view_starts_with(value, prefix)
    ccall((:iree_string_view_starts_with, libiree), Bool, (iree_string_view_t, iree_string_view_t), value, prefix)
end

function iree_string_view_ends_with(value, suffix)
    ccall((:iree_string_view_ends_with, libiree), Bool, (iree_string_view_t, iree_string_view_t), value, suffix)
end

function iree_string_view_remove_prefix(value, n)
    ccall((:iree_string_view_remove_prefix, libiree), iree_string_view_t, (iree_string_view_t, iree_host_size_t), value, n)
end

function iree_string_view_remove_suffix(value, n)
    ccall((:iree_string_view_remove_suffix, libiree), iree_string_view_t, (iree_string_view_t, iree_host_size_t), value, n)
end

function iree_string_view_strip_prefix(value, prefix)
    ccall((:iree_string_view_strip_prefix, libiree), iree_string_view_t, (iree_string_view_t, iree_string_view_t), value, prefix)
end

function iree_string_view_strip_suffix(value, suffix)
    ccall((:iree_string_view_strip_suffix, libiree), iree_string_view_t, (iree_string_view_t, iree_string_view_t), value, suffix)
end

function iree_string_view_consume_prefix(value, prefix)
    ccall((:iree_string_view_consume_prefix, libiree), Bool, (Ptr{iree_string_view_t}, iree_string_view_t), value, prefix)
end

function iree_string_view_consume_suffix(value, suffix)
    ccall((:iree_string_view_consume_suffix, libiree), Bool, (Ptr{iree_string_view_t}, iree_string_view_t), value, suffix)
end

function iree_string_view_trim(value)
    ccall((:iree_string_view_trim, libiree), iree_string_view_t, (iree_string_view_t,), value)
end

function iree_string_view_substr(value, pos, n)
    ccall((:iree_string_view_substr, libiree), iree_string_view_t, (iree_string_view_t, iree_host_size_t, iree_host_size_t), value, pos, n)
end

function iree_string_view_split(value, split_char, out_lhs, out_rhs)
    ccall((:iree_string_view_split, libiree), intptr_t, (iree_string_view_t, Cchar, Ptr{iree_string_view_t}, Ptr{iree_string_view_t}), value, split_char, out_lhs, out_rhs)
end

function iree_string_view_replace_char(value, old_char, new_char)
    ccall((:iree_string_view_replace_char, libiree), Cvoid, (iree_string_view_t, Cchar, Cchar), value, old_char, new_char)
end

function iree_string_view_match_pattern(value, pattern)
    ccall((:iree_string_view_match_pattern, libiree), Bool, (iree_string_view_t, iree_string_view_t), value, pattern)
end

function iree_string_view_append_to_buffer(source_value, target_value, buffer)
    ccall((:iree_string_view_append_to_buffer, libiree), iree_host_size_t, (iree_string_view_t, Ptr{iree_string_view_t}, Cstring), source_value, target_value, buffer)
end

function iree_string_view_atoi_int32(value, out_value)
    ccall((:iree_string_view_atoi_int32, libiree), Bool, (iree_string_view_t, Ptr{Int32}), value, out_value)
end

function iree_string_view_atoi_uint32(value, out_value)
    ccall((:iree_string_view_atoi_uint32, libiree), Bool, (iree_string_view_t, Ptr{UInt32}), value, out_value)
end

function iree_string_view_atoi_int64(value, out_value)
    ccall((:iree_string_view_atoi_int64, libiree), Bool, (iree_string_view_t, Ptr{Int64}), value, out_value)
end

function iree_string_view_atoi_uint64(value, out_value)
    ccall((:iree_string_view_atoi_uint64, libiree), Bool, (iree_string_view_t, Ptr{UInt64}), value, out_value)
end

function iree_string_view_atof(value, out_value)
    ccall((:iree_string_view_atof, libiree), Bool, (iree_string_view_t, Ptr{Cfloat}), value, out_value)
end

function iree_string_view_atod(value, out_value)
    ccall((:iree_string_view_atod, libiree), Bool, (iree_string_view_t, Ptr{Cdouble}), value, out_value)
end

function iree_string_view_parse_hex_bytes(value, buffer_length, out_buffer)
    ccall((:iree_string_view_parse_hex_bytes, libiree), Bool, (iree_string_view_t, iree_host_size_t, Ptr{UInt8}), value, buffer_length, out_buffer)
end

function iree_status_code_from_errno(error_number)
    ccall((:iree_status_code_from_errno, libiree), iree_status_code_t, (Cint,), error_number)
end

function iree_status_code_string(code)
    ccall((:iree_status_code_string, libiree), Cstring, (iree_status_code_t,), code)
end

function iree_status_clone(status)
    ccall((:iree_status_clone, libiree), iree_status_t, (iree_status_t,), status)
end

function iree_status_free(status)
    ccall((:iree_status_free, libiree), Cvoid, (iree_status_t,), status)
end

function iree_status_join(base_status, new_status)
    ccall((:iree_status_join, libiree), iree_status_t, (iree_status_t, iree_status_t), base_status, new_status)
end

function iree_status_consume_code(status)
    ccall((:iree_status_consume_code, libiree), iree_status_code_t, (iree_status_t,), status)
end

function iree_status_format(status, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_status_format, libiree), Bool, (iree_status_t, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), status, buffer_capacity, buffer, out_buffer_length)
end

function iree_status_to_string(status, allocator, out_buffer, out_buffer_length)
    ccall((:iree_status_to_string, libiree), Bool, (iree_status_t, Ptr{iree_allocator_t}, Ptr{Cstring}, Ptr{iree_host_size_t}), status, allocator, out_buffer, out_buffer_length)
end

function iree_status_fprint(file, status)
    ccall((:iree_status_fprint, libiree), Cvoid, (Ptr{Libc.FILE}, iree_status_t), file, status)
end

# no prototype is found for this function at allocator.h:72:32, please use with caution
function iree_byte_span_empty()
    ccall((:iree_byte_span_empty, libiree), iree_byte_span_t, ())
end

function iree_byte_span_is_empty(span)
    ccall((:iree_byte_span_is_empty, libiree), Bool, (iree_byte_span_t,), span)
end

struct iree_const_byte_span_t
    data::Ptr{UInt8}
    data_length::iree_host_size_t
end

function iree_make_const_byte_span(data, data_length)
    ccall((:iree_make_const_byte_span, libiree), iree_const_byte_span_t, (Ptr{Cvoid}, iree_host_size_t), data, data_length)
end

# no prototype is found for this function at allocator.h:93:38, please use with caution
function iree_const_byte_span_empty()
    ccall((:iree_const_byte_span_empty, libiree), iree_const_byte_span_t, ())
end

function iree_const_byte_span_is_empty(span)
    ccall((:iree_const_byte_span_is_empty, libiree), Bool, (iree_const_byte_span_t,), span)
end

@cenum iree_allocator_command_e::UInt32 begin
    IREE_ALLOCATOR_COMMAND_MALLOC = 0
    IREE_ALLOCATOR_COMMAND_CALLOC = 1
    IREE_ALLOCATOR_COMMAND_REALLOC = 2
    IREE_ALLOCATOR_COMMAND_FREE = 3
end

const iree_allocator_command_t = iree_allocator_command_e

mutable struct iree_allocator_alloc_params_t
    byte_length::iree_host_size_t
end

function iree_allocator_malloc(allocator, byte_length, out_ptr)
    ccall((:iree_allocator_malloc, libiree), iree_status_t, (iree_allocator_t, iree_host_size_t, Ptr{Ptr{Cvoid}}), allocator, byte_length, out_ptr)
end

function iree_allocator_malloc_uninitialized(allocator, byte_length, out_ptr)
    ccall((:iree_allocator_malloc_uninitialized, libiree), iree_status_t, (iree_allocator_t, iree_host_size_t, Ptr{Ptr{Cvoid}}), allocator, byte_length, out_ptr)
end

function iree_allocator_realloc(allocator, byte_length, inout_ptr)
    ccall((:iree_allocator_realloc, libiree), iree_status_t, (iree_allocator_t, iree_host_size_t, Ptr{Ptr{Cvoid}}), allocator, byte_length, inout_ptr)
end

function iree_allocator_clone(allocator, source_bytes, out_ptr)
    ccall((:iree_allocator_clone, libiree), iree_status_t, (iree_allocator_t, iree_const_byte_span_t, Ptr{Ptr{Cvoid}}), allocator, source_bytes, out_ptr)
end

function iree_allocator_free(allocator, ptr)
    ccall((:iree_allocator_free, libiree), Cvoid, (iree_allocator_t, Ptr{Cvoid}), allocator, ptr)
end

function iree_allocator_system_ctl(self, command, params, inout_ptr)
    ccall((:iree_allocator_system_ctl, libiree), iree_status_t, (Ptr{Cvoid}, iree_allocator_command_t, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), self, command, params, inout_ptr)
end

function iree_allocator_system()
    ccall((:iree_allocator_system, libiree), iree_allocator_t, ())
end

function iree_allocator_null()
    ccall((:iree_allocator_null, libiree), iree_allocator_t, ())
end

function iree_allocator_is_null(allocator)
    ccall((:iree_allocator_is_null, libiree), Bool, (iree_allocator_t,), allocator)
end

function iree_allocator_malloc_aligned(allocator, byte_length, min_alignment, offset, out_ptr)
    ccall((:iree_allocator_malloc_aligned, libiree), iree_status_t, (iree_allocator_t, iree_host_size_t, iree_host_size_t, iree_host_size_t, Ptr{Ptr{Cvoid}}), allocator, byte_length, min_alignment, offset, out_ptr)
end

function iree_allocator_realloc_aligned(allocator, byte_length, min_alignment, offset, inout_ptr)
    ccall((:iree_allocator_realloc_aligned, libiree), iree_status_t, (iree_allocator_t, iree_host_size_t, iree_host_size_t, iree_host_size_t, Ptr{Ptr{Cvoid}}), allocator, byte_length, min_alignment, offset, inout_ptr)
end

function iree_allocator_free_aligned(allocator, ptr)
    ccall((:iree_allocator_free_aligned, libiree), Cvoid, (iree_allocator_t, Ptr{Cvoid}), allocator, ptr)
end

mutable struct iree_string_builder_t
    allocator::iree_allocator_t
    buffer::Cstring
    size::iree_host_size_t
    capacity::iree_host_size_t
end

function iree_string_builder_initialize(allocator, out_builder)
    ccall((:iree_string_builder_initialize, libiree), Cvoid, (iree_allocator_t, Ptr{iree_string_builder_t}), allocator, out_builder)
end

function iree_string_builder_initialize_with_storage(buffer, buffer_capacity, out_builder)
    ccall((:iree_string_builder_initialize_with_storage, libiree), Cvoid, (Cstring, iree_host_size_t, Ptr{iree_string_builder_t}), buffer, buffer_capacity, out_builder)
end

function iree_string_builder_deinitialize(builder)
    ccall((:iree_string_builder_deinitialize, libiree), Cvoid, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_buffer(builder)
    ccall((:iree_string_builder_buffer, libiree), Cstring, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_size(builder)
    ccall((:iree_string_builder_size, libiree), iree_host_size_t, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_capacity(builder)
    ccall((:iree_string_builder_capacity, libiree), iree_host_size_t, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_view(builder)
    ccall((:iree_string_builder_view, libiree), iree_string_view_t, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_take_storage(builder)
    ccall((:iree_string_builder_take_storage, libiree), Cstring, (Ptr{iree_string_builder_t},), builder)
end

function iree_string_builder_reserve(builder, minimum_capacity)
    ccall((:iree_string_builder_reserve, libiree), iree_status_t, (Ptr{iree_string_builder_t}, iree_host_size_t), builder, minimum_capacity)
end

function iree_string_builder_append_inline(builder, count, out_head)
    ccall((:iree_string_builder_append_inline, libiree), iree_status_t, (Ptr{iree_string_builder_t}, iree_host_size_t, Ptr{Cstring}), builder, count, out_head)
end

function iree_string_builder_append_string(builder, value)
    ccall((:iree_string_builder_append_string, libiree), iree_status_t, (Ptr{iree_string_builder_t}, iree_string_view_t), builder, value)
end

function iree_string_builder_append_cstring(builder, value)
    ccall((:iree_string_builder_append_cstring, libiree), iree_status_t, (Ptr{iree_string_builder_t}, Cstring), builder, value)
end

struct iree_bitfield_string_mapping_t
    bits::UInt32
    string::iree_string_view_t
end

function iree_bitfield_format(value, mapping_count, mappings, string_builder)
    ccall((:iree_bitfield_format, libiree), iree_status_t, (UInt32, iree_host_size_t, Ptr{iree_bitfield_string_mapping_t}, Ptr{iree_string_builder_t}), value, mapping_count, mappings, string_builder)
end

struct iree_bitfield_string_temp_t
    buffer::NTuple{128, Cchar}
end

function iree_bitfield_format_inline(value, mapping_count, mappings, out_temp)
    ccall((:iree_bitfield_format_inline, libiree), iree_string_view_t, (UInt32, iree_host_size_t, Ptr{iree_bitfield_string_mapping_t}, Ptr{iree_bitfield_string_temp_t}), value, mapping_count, mappings, out_temp)
end

const iree_time_t = Int64

const iree_duration_t = Int64

function iree_time_now()
    ccall((:iree_time_now, libiree), iree_time_t, ())
end

function iree_relative_timeout_to_deadline_ns(timeout_ns)
    ccall((:iree_relative_timeout_to_deadline_ns, libiree), iree_time_t, (iree_duration_t,), timeout_ns)
end

function iree_absolute_deadline_to_timeout_ns(deadline_ns)
    ccall((:iree_absolute_deadline_to_timeout_ns, libiree), iree_duration_t, (iree_time_t,), deadline_ns)
end

function iree_absolute_deadline_to_timeout_ms(deadline_ns)
    ccall((:iree_absolute_deadline_to_timeout_ms, libiree), UInt32, (iree_time_t,), deadline_ns)
end

@cenum iree_timeout_type_e::UInt32 begin
    IREE_TIMEOUT_ABSOLUTE = 0
    IREE_TIMEOUT_RELATIVE = 1
end

const iree_timeout_type_t = iree_timeout_type_e

struct iree_timeout_t
    type::iree_timeout_type_t
    nanos::iree_time_t
end

function iree_immediate_timeout()
    ccall((:iree_immediate_timeout, libiree), iree_timeout_t, ())
end

function iree_timeout_is_immediate(timeout)
    ccall((:iree_timeout_is_immediate, libiree), Bool, (iree_timeout_t,), timeout)
end

function iree_infinite_timeout()
    ccall((:iree_infinite_timeout, libiree), iree_timeout_t, ())
end

function iree_timeout_is_infinite(timeout)
    ccall((:iree_timeout_is_infinite, libiree), Bool, (iree_timeout_t,), timeout)
end

function iree_make_deadline(deadline_ns)
    ccall((:iree_make_deadline, libiree), iree_timeout_t, (iree_time_t,), deadline_ns)
end

function iree_make_timeout_ns(timeout_ns)
    ccall((:iree_make_timeout_ns, libiree), iree_timeout_t, (iree_duration_t,), timeout_ns)
end

function iree_make_timeout_ms(timeout_ms)
    ccall((:iree_make_timeout_ms, libiree), iree_timeout_t, (iree_duration_t,), timeout_ms)
end

function iree_convert_timeout_to_absolute(timeout)
    ccall((:iree_convert_timeout_to_absolute, libiree), Cvoid, (Ptr{iree_timeout_t},), timeout)
end

function iree_timeout_as_deadline_ns(timeout)
    ccall((:iree_timeout_as_deadline_ns, libiree), iree_time_t, (iree_timeout_t,), timeout)
end

function iree_timeout_min(lhs, rhs)
    ccall((:iree_timeout_min, libiree), iree_timeout_t, (iree_timeout_t, iree_timeout_t), lhs, rhs)
end

function iree_wait_until(deadline_ns)
    ccall((:iree_wait_until, libiree), Bool, (iree_time_t,), deadline_ns)
end

@cenum iree_wait_primitive_type_bits_t::UInt32 begin
    IREE_WAIT_PRIMITIVE_TYPE_NONE = 0
    IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD = 1
    IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE = 2
    IREE_WAIT_PRIMITIVE_TYPE_PIPE = 3
    IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE = 4
    IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX = 5
    IREE_WAIT_PRIMITIVE_TYPE_ANY = 255
end

const iree_wait_primitive_type_t = UInt8

struct iree_wait_primitive_value_t
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{iree_wait_primitive_value_t}, f::Symbol)
    f === :reserved && return Ptr{Cint}(x + 0)
    f === :event && return Ptr{__JL_Ctag_50}(x + 0)
    f === :pipe && return Ptr{__JL_Ctag_51}(x + 0)
    f === :local_futex && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::iree_wait_primitive_value_t, f::Symbol)
    r = Ref{iree_wait_primitive_value_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_wait_primitive_value_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_wait_primitive_value_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

mutable struct iree_wait_primitive_t
    type::iree_wait_primitive_type_t
    value::iree_wait_primitive_value_t
end

function iree_make_wait_primitive(type, value)
    ccall((:iree_make_wait_primitive, libiree), iree_wait_primitive_t, (iree_wait_primitive_type_t, iree_wait_primitive_value_t), type, value)
end

function iree_wait_primitive_immediate()
    ccall((:iree_wait_primitive_immediate, libiree), iree_wait_primitive_t, ())
end

function iree_wait_primitive_is_immediate(wait_primitive)
    ccall((:iree_wait_primitive_is_immediate, libiree), Bool, (iree_wait_primitive_t,), wait_primitive)
end

# typedef iree_status_t ( IREE_API_PTR * iree_wait_source_ctl_fn_t ) ( iree_wait_source_t wait_source , iree_wait_source_command_t command , const void * params , void * * inout_ptr )
const iree_wait_source_ctl_fn_t = Ptr{Cvoid}

struct iree_wait_source_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_wait_source_t}, f::Symbol)
    f === :self && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :data && return Ptr{UInt64}(x + 8)
    f === :storage && return Ptr{NTuple{2, UInt64}}(x + 0)
    f === :ctl && return Ptr{iree_wait_source_ctl_fn_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_wait_source_t, f::Symbol)
    r = Ref{iree_wait_source_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_wait_source_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_wait_source_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

@cenum iree_wait_source_command_e::UInt32 begin
    IREE_WAIT_SOURCE_COMMAND_QUERY = 0
    IREE_WAIT_SOURCE_COMMAND_WAIT_ONE = 1
    IREE_WAIT_SOURCE_COMMAND_EXPORT = 2
end

const iree_wait_source_command_t = iree_wait_source_command_e

mutable struct iree_wait_source_wait_params_t
    timeout::iree_timeout_t
end

mutable struct iree_wait_source_export_params_t
    target_type::iree_wait_primitive_type_t
    timeout::iree_timeout_t
end

function iree_wait_source_immediate()
    ccall((:iree_wait_source_immediate, libiree), iree_wait_source_t, ())
end

function iree_wait_source_is_immediate(wait_source)
    ccall((:iree_wait_source_is_immediate, libiree), Bool, (iree_wait_source_t,), wait_source)
end

function iree_wait_source_delay_ctl(wait_source, command, params, inout_ptr)
    ccall((:iree_wait_source_delay_ctl, libiree), iree_status_t, (iree_wait_source_t, iree_wait_source_command_t, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), wait_source, command, params, inout_ptr)
end

function iree_wait_source_delay(deadline_ns)
    ccall((:iree_wait_source_delay, libiree), iree_wait_source_t, (iree_time_t,), deadline_ns)
end

function iree_wait_source_is_delay(wait_source)
    ccall((:iree_wait_source_is_delay, libiree), Bool, (iree_wait_source_t,), wait_source)
end

function iree_wait_source_import(wait_primitive, out_wait_source)
    ccall((:iree_wait_source_import, libiree), iree_status_t, (iree_wait_primitive_t, Ptr{iree_wait_source_t}), wait_primitive, out_wait_source)
end

function iree_wait_source_export(wait_source, target_type, timeout, out_wait_primitive)
    ccall((:iree_wait_source_export, libiree), iree_status_t, (iree_wait_source_t, iree_wait_primitive_type_t, iree_timeout_t, Ptr{iree_wait_primitive_t}), wait_source, target_type, timeout, out_wait_primitive)
end

function iree_wait_source_query(wait_source, out_wait_status_code)
    ccall((:iree_wait_source_query, libiree), iree_status_t, (iree_wait_source_t, Ptr{iree_status_code_t}), wait_source, out_wait_status_code)
end

function iree_wait_source_wait_one(wait_source, timeout)
    ccall((:iree_wait_source_wait_one, libiree), iree_status_t, (iree_wait_source_t, iree_timeout_t), wait_source, timeout)
end

# typedef iree_status_t ( IREE_API_PTR * iree_loop_ctl_fn_t ) ( void * self , iree_loop_command_t command , const void * params , void * * inout_ptr )
const iree_loop_ctl_fn_t = Ptr{Cvoid}

mutable struct iree_loop_t
    self::Ptr{Cvoid}
    ctl::iree_loop_ctl_fn_t
end

const iree_loop_command_t = UInt32

@cenum iree_loop_priority_e::UInt32 begin
    IREE_LOOP_PRIORITY_DEFAULT = 0
end

const iree_loop_priority_t = iree_loop_priority_e

# typedef iree_status_t ( IREE_API_PTR * iree_loop_callback_fn_t ) ( void * user_data , iree_loop_t loop , iree_status_t status )
const iree_loop_callback_fn_t = Ptr{Cvoid}

# typedef iree_status_t ( IREE_API_PTR * iree_loop_workgroup_fn_t ) ( void * user_data , iree_loop_t loop , uint32_t workgroup_x , uint32_t workgroup_y , uint32_t workgroup_z )
const iree_loop_workgroup_fn_t = Ptr{Cvoid}

# no prototype is found for this function at loop.h:117:27, please use with caution
function iree_loop_null()
    ccall((:iree_loop_null, libiree), iree_loop_t, ())
end

function iree_loop_call(loop, priority, callback, user_data)
    ccall((:iree_loop_call, libiree), iree_status_t, (iree_loop_t, iree_loop_priority_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, priority, callback, user_data)
end

function iree_loop_dispatch(loop, workgroup_count_xyz, workgroup_callback, completion_callback, user_data)
    ccall((:iree_loop_dispatch, libiree), iree_status_t, (iree_loop_t, Ptr{UInt32}, iree_loop_workgroup_fn_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, workgroup_count_xyz, workgroup_callback, completion_callback, user_data)
end

function iree_loop_wait_until(loop, timeout, callback, user_data)
    ccall((:iree_loop_wait_until, libiree), iree_status_t, (iree_loop_t, iree_timeout_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, timeout, callback, user_data)
end

function iree_loop_wait_one(loop, wait_source, timeout, callback, user_data)
    ccall((:iree_loop_wait_one, libiree), iree_status_t, (iree_loop_t, iree_wait_source_t, iree_timeout_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, wait_source, timeout, callback, user_data)
end

function iree_loop_wait_any(loop, count, wait_sources, timeout, callback, user_data)
    ccall((:iree_loop_wait_any, libiree), iree_status_t, (iree_loop_t, iree_host_size_t, Ptr{iree_wait_source_t}, iree_timeout_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, count, wait_sources, timeout, callback, user_data)
end

function iree_loop_wait_all(loop, count, wait_sources, timeout, callback, user_data)
    ccall((:iree_loop_wait_all, libiree), iree_status_t, (iree_loop_t, iree_host_size_t, Ptr{iree_wait_source_t}, iree_timeout_t, iree_loop_callback_fn_t, Ptr{Cvoid}), loop, count, wait_sources, timeout, callback, user_data)
end

function iree_loop_drain(loop, timeout)
    ccall((:iree_loop_drain, libiree), iree_status_t, (iree_loop_t, iree_timeout_t), loop, timeout)
end

@cenum iree_loop_command_e::UInt32 begin
    IREE_LOOP_COMMAND_CALL = 0
    IREE_LOOP_COMMAND_DISPATCH = 1
    IREE_LOOP_COMMAND_WAIT_UNTIL = 2
    IREE_LOOP_COMMAND_WAIT_ONE = 3
    IREE_LOOP_COMMAND_WAIT_ANY = 4
    IREE_LOOP_COMMAND_WAIT_ALL = 5
    IREE_LOOP_COMMAND_DRAIN = 6
    IREE_LOOP_COMMAND_MAX = 6
end

struct iree_loop_callback_t
    fn::iree_loop_callback_fn_t
    user_data::Ptr{Cvoid}
end

mutable struct iree_loop_call_params_t
    callback::iree_loop_callback_t
    priority::iree_loop_priority_t
end

mutable struct iree_loop_dispatch_params_t
    callback::iree_loop_callback_t
    workgroup_fn::iree_loop_workgroup_fn_t
    workgroup_count_xyz::NTuple{3, UInt32}
end

mutable struct iree_loop_wait_until_params_t
    callback::iree_loop_callback_t
    deadline_ns::iree_time_t
end

mutable struct iree_loop_wait_one_params_t
    callback::iree_loop_callback_t
    deadline_ns::iree_time_t
    wait_source::iree_wait_source_t
end

mutable struct iree_loop_wait_multi_params_t
    callback::iree_loop_callback_t
    deadline_ns::iree_time_t
    count::iree_host_size_t
    wait_sources::Ptr{iree_wait_source_t}
end

mutable struct iree_loop_drain_params_t
    deadline_ns::iree_time_t
end

function iree_loop_inline_ctl(self, command, params, inout_ptr)
    ccall((:iree_loop_inline_ctl, libiree), iree_status_t, (Ptr{Cvoid}, iree_loop_command_t, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), self, command, params, inout_ptr)
end

function iree_loop_inline_using_storage_ctl(self, command, params, inout_ptr)
    ccall((:iree_loop_inline_using_storage_ctl, libiree), iree_status_t, (Ptr{Cvoid}, iree_loop_command_t, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), self, command, params, inout_ptr)
end

function iree_loop_inline(out_status)
    ccall((:iree_loop_inline, libiree), iree_loop_t, (Ptr{iree_status_t},), out_status)
end

mutable struct iree_loop_inline_storage_t
    opaque::NTuple{512, UInt8}
    status::iree_status_t
end

function iree_loop_inline_initialize(storage)
    ccall((:iree_loop_inline_initialize, libiree), iree_loop_t, (Ptr{iree_loop_inline_storage_t},), storage)
end

function iree_loop_inline_deinitialize(storage)
    ccall((:iree_loop_inline_deinitialize, libiree), Cvoid, (Ptr{iree_loop_inline_storage_t},), storage)
end

@cenum iree_memory_order_e::UInt32 begin
    iree_memory_order_relaxed = 0
    iree_memory_order_consume = 1
    iree_memory_order_acquire = 2
    iree_memory_order_release = 3
    iree_memory_order_acq_rel = 4
    iree_memory_order_seq_cst = 5
end

const iree_memory_order_t = iree_memory_order_e

mutable struct iree_hal_resource_vtable_t
    destroy::Ptr{Cvoid}
end

function iree_hal_resource_retain(any_resource)
    ccall((:iree_hal_resource_retain, libiree), Cvoid, (Ptr{Cvoid},), any_resource)
end

function iree_hal_resource_release(any_resource)
    ccall((:iree_hal_resource_release, libiree), Cvoid, (Ptr{Cvoid},), any_resource)
end

mutable struct iree_hal_allocator_t end

@cenum iree_hal_memory_type_bits_t::UInt32 begin
    IREE_HAL_MEMORY_TYPE_NONE = 0
    IREE_HAL_MEMORY_TYPE_OPTIMAL = 1
    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE = 2
    IREE_HAL_MEMORY_TYPE_HOST_COHERENT = 4
    IREE_HAL_MEMORY_TYPE_HOST_CACHED = 8
    IREE_HAL_MEMORY_TYPE_HOST_LOCAL = 38
    IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST = 39
    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE = 16
    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL = 48
    IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE = 49
end

const iree_hal_memory_type_t = UInt32

@cenum iree_hal_memory_access_bits_t::UInt32 begin
    IREE_HAL_MEMORY_ACCESS_NONE = 0
    IREE_HAL_MEMORY_ACCESS_READ = 1
    IREE_HAL_MEMORY_ACCESS_WRITE = 2
    IREE_HAL_MEMORY_ACCESS_DISCARD = 4
    IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE = 6
    IREE_HAL_MEMORY_ACCESS_MAY_ALIAS = 8
    IREE_HAL_MEMORY_ACCESS_ANY = 16
    IREE_HAL_MEMORY_ACCESS_ALL = 7
end

const iree_hal_memory_access_t = UInt16

@cenum iree_hal_buffer_usage_bits_t::UInt32 begin
    IREE_HAL_BUFFER_USAGE_NONE = 0
    IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE = 1
    IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET = 2
    IREE_HAL_BUFFER_USAGE_TRANSFER = 3
    IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMS = 256
    IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ = 512
    IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ = 1024
    IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE = 2048
    IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE = 3072
    IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ = 4096
    IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE = 8192
    IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE = 12288
    IREE_HAL_BUFFER_USAGE_SHARING_EXPORT = 65536
    IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE = 131072
    IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT = 262144
    IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE = 524288
    IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED = 16777216
    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT = 33554432
    IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL = 67108864
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM = 134217728
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE = 268435456
    IREE_HAL_BUFFER_USAGE_MAPPING = 150994944
    IREE_HAL_BUFFER_USAGE_DEFAULT = 3075
end

const iree_hal_buffer_usage_t = UInt32

@cenum iree_hal_buffer_overlap_e::UInt32 begin
    IREE_HAL_BUFFER_OVERLAP_DISJOINT = 0
    IREE_HAL_BUFFER_OVERLAP_PARTIAL = 1
    IREE_HAL_BUFFER_OVERLAP_COMPLETE = 2
end

const iree_hal_buffer_overlap_t = iree_hal_buffer_overlap_e

@cenum iree_hal_transfer_buffer_flag_bits_t::UInt32 begin
    IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT = 0
end

const iree_hal_transfer_buffer_flags_t = UInt32

@cenum iree_hal_mapping_mode_bits_t::UInt32 begin
    IREE_HAL_MAPPING_MODE_SCOPED = 1
    IREE_HAL_MAPPING_MODE_PERSISTENT = 2
end

const iree_hal_mapping_mode_t = UInt32

struct iree_hal_buffer_mapping_impl_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_hal_buffer_mapping_impl_t}, f::Symbol)
    f === :byte_offset && return Ptr{iree_device_size_t}(x + 0)
    f === :allowed_access && return Ptr{iree_hal_memory_access_t}(x + 8)
    f === :is_persistent && return (Ptr{UInt32}(x + 8), 16, 1)
    f === :reserved_flags && return (Ptr{UInt32}(x + 12), 0, 31)
    f === :reserved && return Ptr{NTuple{1, UInt64}}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_hal_buffer_mapping_impl_t, f::Symbol)
    r = Ref{iree_hal_buffer_mapping_impl_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_hal_buffer_mapping_impl_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{iree_hal_buffer_mapping_impl_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

function iree_hal_memory_type_format(value, out_temp)
    ccall((:iree_hal_memory_type_format, libiree), iree_string_view_t, (iree_hal_memory_type_t, Ptr{iree_bitfield_string_temp_t}), value, out_temp)
end

function iree_hal_memory_access_format(value, out_temp)
    ccall((:iree_hal_memory_access_format, libiree), iree_string_view_t, (iree_hal_memory_access_t, Ptr{iree_bitfield_string_temp_t}), value, out_temp)
end

function iree_hal_buffer_usage_format(value, out_temp)
    ccall((:iree_hal_buffer_usage_format, libiree), iree_string_view_t, (iree_hal_buffer_usage_t, Ptr{iree_bitfield_string_temp_t}), value, out_temp)
end

function iree_hal_buffer_validate_memory_type(actual_memory_type, expected_memory_type)
    ccall((:iree_hal_buffer_validate_memory_type, libiree), iree_status_t, (iree_hal_memory_type_t, iree_hal_memory_type_t), actual_memory_type, expected_memory_type)
end

function iree_hal_buffer_validate_access(allowed_memory_access, required_memory_access)
    ccall((:iree_hal_buffer_validate_access, libiree), iree_status_t, (iree_hal_memory_access_t, iree_hal_memory_access_t), allowed_memory_access, required_memory_access)
end

function iree_hal_buffer_validate_usage(allowed_usage, required_usage)
    ccall((:iree_hal_buffer_validate_usage, libiree), iree_status_t, (iree_hal_buffer_usage_t, iree_hal_buffer_usage_t), allowed_usage, required_usage)
end

mutable struct iree_hal_buffer_vtable_t
    recycle::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
    map_range::Ptr{Cvoid}
    unmap_range::Ptr{Cvoid}
    invalidate_range::Ptr{Cvoid}
    flush_range::Ptr{Cvoid}
end

@cenum iree_hal_allocator_pool_bits_t::UInt32 begin
    IREE_HAL_ALLOCATOR_POOL_DEFAULT = 0
end

const iree_hal_allocator_pool_t = UInt32

mutable struct iree_hal_buffer_params_t
    usage::iree_hal_buffer_usage_t
    access::iree_hal_memory_access_t
    type::iree_hal_memory_type_t
    queue_affinity::iree_hal_queue_affinity_t
    min_alignment::iree_device_size_t
end

function iree_hal_buffer_params_canonicalize(params)
    ccall((:iree_hal_buffer_params_canonicalize, libiree), Cvoid, (Ptr{iree_hal_buffer_params_t},), params)
end

function iree_hal_buffer_params_with_usage(params, usage)
    ccall((:iree_hal_buffer_params_with_usage, libiree), iree_hal_buffer_params_t, (iree_hal_buffer_params_t, iree_hal_buffer_usage_t), params, usage)
end

@cenum iree_hal_buffer_compatibility_bits_t::UInt32 begin
    IREE_HAL_BUFFER_COMPATIBILITY_NONE = 0
    IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE = 1
    IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE = 2
    IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE = 4
    IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER = 1024
    IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH = 2048
end

const iree_hal_buffer_compatibility_t = UInt32

@cenum iree_hal_external_buffer_type_e::UInt32 begin
    IREE_HAL_EXTERNAL_BUFFER_TYPE_NONE = 0
    IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION = 1
    IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD = 2
    IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32 = 3
end

const iree_hal_external_buffer_type_t = iree_hal_external_buffer_type_e

@cenum iree_hal_external_buffer_flag_bits_t::UInt32 begin
    IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE = 0
end

const iree_hal_external_buffer_flags_t = UInt32

struct __JL_Ctag_54
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{__JL_Ctag_54}, f::Symbol)
    f === :host_allocation && return Ptr{__JL_Ctag_55}(x + 0)
    f === :opaque_fd && return Ptr{__JL_Ctag_56}(x + 0)
    f === :opaque_win32 && return Ptr{__JL_Ctag_57}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_54, f::Symbol)
    r = Ref{__JL_Ctag_54}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_54}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_54}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct iree_hal_external_buffer_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_hal_external_buffer_t}, f::Symbol)
    f === :type && return Ptr{iree_hal_external_buffer_type_t}(x + 0)
    f === :flags && return Ptr{iree_hal_external_buffer_flags_t}(x + 4)
    f === :size && return Ptr{iree_device_size_t}(x + 8)
    f === :handle && return Ptr{__JL_Ctag_54}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_hal_external_buffer_t, f::Symbol)
    r = Ref{iree_hal_external_buffer_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_hal_external_buffer_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_hal_external_buffer_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

# typedef void ( IREE_API_PTR * iree_hal_buffer_release_fn_t ) ( void * user_data , iree_hal_buffer_t * buffer )
const iree_hal_buffer_release_fn_t = Ptr{Cvoid}

mutable struct iree_hal_buffer_release_callback_t
    fn::iree_hal_buffer_release_fn_t
    user_data::Ptr{Cvoid}
end

function iree_hal_buffer_release_callback_null()
    ccall((:iree_hal_buffer_release_callback_null, libiree), iree_hal_buffer_release_callback_t, ())
end

mutable struct iree_hal_allocator_statistics_t
    host_bytes_peak::iree_device_size_t
    host_bytes_allocated::iree_device_size_t
    host_bytes_freed::iree_device_size_t
    device_bytes_peak::iree_device_size_t
    device_bytes_allocated::iree_device_size_t
    device_bytes_freed::iree_device_size_t
end

function iree_hal_allocator_statistics_format(statistics, builder)
    ccall((:iree_hal_allocator_statistics_format, libiree), iree_status_t, (Ptr{iree_hal_allocator_statistics_t}, Ptr{iree_string_builder_t}), statistics, builder)
end

function iree_hal_allocator_retain(allocator)
    ccall((:iree_hal_allocator_retain, libiree), Cvoid, (Ptr{iree_hal_allocator_t},), allocator)
end

function iree_hal_allocator_release(allocator)
    ccall((:iree_hal_allocator_release, libiree), Cvoid, (Ptr{iree_hal_allocator_t},), allocator)
end

function iree_hal_allocator_host_allocator(allocator)
    ccall((:iree_hal_allocator_host_allocator, libiree), iree_allocator_t, (Ptr{iree_hal_allocator_t},), allocator)
end

function iree_hal_allocator_trim(allocator)
    ccall((:iree_hal_allocator_trim, libiree), iree_status_t, (Ptr{iree_hal_allocator_t},), allocator)
end

function iree_hal_allocator_query_statistics(allocator, out_statistics)
    ccall((:iree_hal_allocator_query_statistics, libiree), Cvoid, (Ptr{iree_hal_allocator_t}, Ptr{iree_hal_allocator_statistics_t}), allocator, out_statistics)
end

function iree_hal_allocator_statistics_fprint(file, allocator)
    ccall((:iree_hal_allocator_statistics_fprint, libiree), iree_status_t, (Ptr{Libc.FILE}, Ptr{iree_hal_allocator_t}), file, allocator)
end

function iree_hal_allocator_query_compatibility(allocator, params, allocation_size)
    ccall((:iree_hal_allocator_query_compatibility, libiree), iree_hal_buffer_compatibility_t, (Ptr{iree_hal_allocator_t}, iree_hal_buffer_params_t, iree_device_size_t), allocator, params, allocation_size)
end

function iree_hal_allocator_create_heap(identifier, data_allocator, host_allocator, out_allocator)
    ccall((:iree_hal_allocator_create_heap, libiree), iree_status_t, (iree_string_view_t, iree_allocator_t, iree_allocator_t, Ptr{Ptr{iree_hal_allocator_t}}), identifier, data_allocator, host_allocator, out_allocator)
end

mutable struct iree_hal_allocator_vtable_t
    destroy::Ptr{Cvoid}
    host_allocator::Ptr{Cvoid}
    trim::Ptr{Cvoid}
    query_statistics::Ptr{Cvoid}
    query_compatibility::Ptr{Cvoid}
    allocate_buffer::Ptr{Cvoid}
    deallocate_buffer::Ptr{Cvoid}
    import_buffer::Ptr{Cvoid}
    export_buffer::Ptr{Cvoid}
end

function iree_hal_allocator_destroy(allocator)
    ccall((:iree_hal_allocator_destroy, libiree), Cvoid, (Ptr{iree_hal_allocator_t},), allocator)
end

function iree_hal_allocator_statistics_record_alloc(statistics, memory_type, allocation_size)
    ccall((:iree_hal_allocator_statistics_record_alloc, libiree), Cvoid, (Ptr{iree_hal_allocator_statistics_t}, iree_hal_memory_type_t, iree_device_size_t), statistics, memory_type, allocation_size)
end

function iree_hal_allocator_statistics_record_free(statistics, memory_type, allocation_size)
    ccall((:iree_hal_allocator_statistics_record_free, libiree), Cvoid, (Ptr{iree_hal_allocator_statistics_t}, iree_hal_memory_type_t, iree_device_size_t), statistics, memory_type, allocation_size)
end

@cenum iree_hal_numerical_type_bits_t::UInt32 begin
    IREE_HAL_NUMERICAL_TYPE_UNKNOWN = 0
    IREE_HAL_NUMERICAL_TYPE_INTEGER = 16
    IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED = 17
    IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED = 18
    IREE_HAL_NUMERICAL_TYPE_BOOLEAN = 19
    IREE_HAL_NUMERICAL_TYPE_FLOAT = 32
    IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE = 33
    IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN = 34
    IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX = 35
end

@cenum iree_hal_element_types_t::UInt32 begin
    IREE_HAL_ELEMENT_TYPE_NONE = 0
    IREE_HAL_ELEMENT_TYPE_OPAQUE_8 = 8
    IREE_HAL_ELEMENT_TYPE_OPAQUE_16 = 16
    IREE_HAL_ELEMENT_TYPE_OPAQUE_32 = 32
    IREE_HAL_ELEMENT_TYPE_OPAQUE_64 = 64
    IREE_HAL_ELEMENT_TYPE_BOOL_8 = 318767112
    IREE_HAL_ELEMENT_TYPE_INT_4 = 268435460
    IREE_HAL_ELEMENT_TYPE_SINT_4 = 285212676
    IREE_HAL_ELEMENT_TYPE_UINT_4 = 301989892
    IREE_HAL_ELEMENT_TYPE_INT_8 = 268435464
    IREE_HAL_ELEMENT_TYPE_SINT_8 = 285212680
    IREE_HAL_ELEMENT_TYPE_UINT_8 = 301989896
    IREE_HAL_ELEMENT_TYPE_INT_16 = 268435472
    IREE_HAL_ELEMENT_TYPE_SINT_16 = 285212688
    IREE_HAL_ELEMENT_TYPE_UINT_16 = 301989904
    IREE_HAL_ELEMENT_TYPE_INT_32 = 268435488
    IREE_HAL_ELEMENT_TYPE_SINT_32 = 285212704
    IREE_HAL_ELEMENT_TYPE_UINT_32 = 301989920
    IREE_HAL_ELEMENT_TYPE_INT_64 = 268435520
    IREE_HAL_ELEMENT_TYPE_SINT_64 = 285212736
    IREE_HAL_ELEMENT_TYPE_UINT_64 = 301989952
    IREE_HAL_ELEMENT_TYPE_FLOAT_16 = 553648144
    IREE_HAL_ELEMENT_TYPE_FLOAT_32 = 553648160
    IREE_HAL_ELEMENT_TYPE_FLOAT_64 = 553648192
    IREE_HAL_ELEMENT_TYPE_BFLOAT_16 = 570425360
    IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64 = 587202624
    IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128 = 587202688
end

@cenum iree_hal_encoding_types_t::UInt32 begin
    IREE_HAL_ENCODING_TYPE_OPAQUE = 0
    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR = 1
end

const iree_hal_encoding_type_t = UInt32

const iree_hal_dim_t = iree_device_size_t

mutable struct iree_hal_buffer_view_t end

function iree_hal_buffer_view_retain(buffer_view)
    ccall((:iree_hal_buffer_view_retain, libiree), Cvoid, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_release(buffer_view)
    ccall((:iree_hal_buffer_view_release, libiree), Cvoid, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_shape_rank(buffer_view)
    ccall((:iree_hal_buffer_view_shape_rank, libiree), iree_host_size_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_shape_dims(buffer_view)
    ccall((:iree_hal_buffer_view_shape_dims, libiree), Ptr{iree_hal_dim_t}, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_shape_dim(buffer_view, index)
    ccall((:iree_hal_buffer_view_shape_dim, libiree), iree_hal_dim_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t), buffer_view, index)
end

function iree_hal_buffer_view_shape(buffer_view, rank_capacity, out_shape, out_shape_rank)
    ccall((:iree_hal_buffer_view_shape, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_host_size_t}), buffer_view, rank_capacity, out_shape, out_shape_rank)
end

function iree_hal_buffer_view_reshape(buffer_view, shape, shape_rank)
    ccall((:iree_hal_buffer_view_reshape, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, Ptr{iree_hal_dim_t}, iree_host_size_t), buffer_view, shape, shape_rank)
end

function iree_hal_buffer_view_element_count(buffer_view)
    ccall((:iree_hal_buffer_view_element_count, libiree), iree_host_size_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_element_type(buffer_view)
    ccall((:iree_hal_buffer_view_element_type, libiree), iree_hal_element_type_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_element_size(buffer_view)
    ccall((:iree_hal_buffer_view_element_size, libiree), iree_host_size_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_encoding_type(buffer_view)
    ccall((:iree_hal_buffer_view_encoding_type, libiree), iree_hal_encoding_type_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_byte_length(buffer_view)
    ccall((:iree_hal_buffer_view_byte_length, libiree), iree_device_size_t, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_view_compute_offset(buffer_view, indices_count, indices, out_offset)
    ccall((:iree_hal_buffer_view_compute_offset, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_device_size_t}), buffer_view, indices_count, indices, out_offset)
end

function iree_hal_buffer_view_compute_range(buffer_view, indices_count, start_indices, lengths_count, lengths, out_start_offset, out_length)
    ccall((:iree_hal_buffer_view_compute_range, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_device_size_t}, Ptr{iree_device_size_t}), buffer_view, indices_count, start_indices, lengths_count, lengths, out_start_offset, out_length)
end

function iree_hal_buffer_view_destroy(buffer_view)
    ccall((:iree_hal_buffer_view_destroy, libiree), Cvoid, (Ptr{iree_hal_buffer_view_t},), buffer_view)
end

function iree_hal_buffer_compute_view_size(shape_rank, shape, element_type, encoding_type, out_allocation_size)
    ccall((:iree_hal_buffer_compute_view_size, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_hal_encoding_type_t, Ptr{iree_device_size_t}), shape_rank, shape, element_type, encoding_type, out_allocation_size)
end

function iree_hal_buffer_compute_view_offset(shape_rank, shape, element_type, encoding_type, indices_count, indices, out_offset)
    ccall((:iree_hal_buffer_compute_view_offset, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_hal_encoding_type_t, iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_device_size_t}), shape_rank, shape, element_type, encoding_type, indices_count, indices, out_offset)
end

function iree_hal_buffer_compute_view_range(shape_rank, shape, element_type, encoding_type, indices_count, start_indices, lengths_count, lengths, out_start_offset, out_length)
    ccall((:iree_hal_buffer_compute_view_range, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_hal_encoding_type_t, iree_host_size_t, Ptr{iree_hal_dim_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_device_size_t}, Ptr{iree_device_size_t}), shape_rank, shape, element_type, encoding_type, indices_count, start_indices, lengths_count, lengths, out_start_offset, out_length)
end

function iree_hal_buffer_view_allocate_buffer(allocator, shape_rank, shape, element_type, encoding_type, buffer_params, initial_data, out_buffer_view)
    ccall((:iree_hal_buffer_view_allocate_buffer, libiree), iree_status_t, (Ptr{iree_hal_allocator_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_hal_encoding_type_t, iree_hal_buffer_params_t, iree_const_byte_span_t, Ptr{Ptr{iree_hal_buffer_view_t}}), allocator, shape_rank, shape, element_type, encoding_type, buffer_params, initial_data, out_buffer_view)
end

# typedef iree_status_t ( IREE_API_PTR * iree_hal_buffer_view_generator_callback_t ) ( iree_hal_buffer_mapping_t * mapping , void * user_data )
const iree_hal_buffer_view_generator_callback_t = Ptr{Cvoid}

function iree_hal_buffer_view_generate_buffer(allocator, shape_rank, shape, element_type, encoding_type, buffer_params, callback, user_data, out_buffer_view)
    ccall((:iree_hal_buffer_view_generate_buffer, libiree), iree_status_t, (Ptr{iree_hal_allocator_t}, iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_hal_encoding_type_t, iree_hal_buffer_params_t, iree_hal_buffer_view_generator_callback_t, Ptr{Cvoid}, Ptr{Ptr{iree_hal_buffer_view_t}}), allocator, shape_rank, shape, element_type, encoding_type, buffer_params, callback, user_data, out_buffer_view)
end

function iree_hal_buffer_view_parse(value, buffer_allocator, out_buffer_view)
    ccall((:iree_hal_buffer_view_parse, libiree), iree_status_t, (iree_string_view_t, Ptr{iree_hal_allocator_t}, Ptr{Ptr{iree_hal_buffer_view_t}}), value, buffer_allocator, out_buffer_view)
end

function iree_hal_buffer_view_format(buffer_view, max_element_count, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_hal_buffer_view_format, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), buffer_view, max_element_count, buffer_capacity, buffer, out_buffer_length)
end

function iree_hal_buffer_view_fprint(file, buffer_view, max_element_count, host_allocator)
    ccall((:iree_hal_buffer_view_fprint, libiree), iree_status_t, (Ptr{Libc.FILE}, Ptr{iree_hal_buffer_view_t}, iree_host_size_t, iree_allocator_t), file, buffer_view, max_element_count, host_allocator)
end

function iree_hal_buffer_view_append_to_builder(buffer_view, max_element_count, builder)
    ccall((:iree_hal_buffer_view_append_to_builder, libiree), iree_status_t, (Ptr{iree_hal_buffer_view_t}, iree_host_size_t, Ptr{iree_string_builder_t}), buffer_view, max_element_count, builder)
end

mutable struct iree_hal_device_t end

@cenum iree_hal_channel_flag_bits_t::UInt32 begin
    IREE_HAL_CHANNEL_FLAG_NONE = 0
end

const iree_hal_channel_flags_t = UInt32

mutable struct iree_hal_channel_params_t
    flags::iree_hal_channel_flags_t
    id::iree_const_byte_span_t
    rank::Int32
    count::Int32
end

mutable struct iree_hal_channel_t end

function iree_hal_channel_create(device, queue_affinity, params, out_channel)
    ccall((:iree_hal_channel_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_hal_queue_affinity_t, iree_hal_channel_params_t, Ptr{Ptr{iree_hal_channel_t}}), device, queue_affinity, params, out_channel)
end

function iree_hal_channel_retain(channel)
    ccall((:iree_hal_channel_retain, libiree), Cvoid, (Ptr{iree_hal_channel_t},), channel)
end

function iree_hal_channel_release(channel)
    ccall((:iree_hal_channel_release, libiree), Cvoid, (Ptr{iree_hal_channel_t},), channel)
end

function iree_hal_channel_query_rank_and_count(channel, out_rank, out_count)
    ccall((:iree_hal_channel_query_rank_and_count, libiree), Cvoid, (Ptr{iree_hal_channel_t}, Ptr{Int32}, Ptr{Int32}), channel, out_rank, out_count)
end

function iree_hal_channel_rank(channel)
    ccall((:iree_hal_channel_rank, libiree), Int32, (Ptr{iree_hal_channel_t},), channel)
end

function iree_hal_channel_count(channel)
    ccall((:iree_hal_channel_count, libiree), Int32, (Ptr{iree_hal_channel_t},), channel)
end

mutable struct iree_hal_channel_vtable_t
    destroy::Ptr{Cvoid}
    query_rank_and_count::Ptr{Cvoid}
end

function iree_hal_channel_destroy(channel)
    ccall((:iree_hal_channel_destroy, libiree), Cvoid, (Ptr{iree_hal_channel_t},), channel)
end

mutable struct iree_hal_event_t end

function iree_hal_event_create(device, out_event)
    ccall((:iree_hal_event_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, Ptr{Ptr{iree_hal_event_t}}), device, out_event)
end

function iree_hal_event_retain(event)
    ccall((:iree_hal_event_retain, libiree), Cvoid, (Ptr{iree_hal_event_t},), event)
end

function iree_hal_event_release(event)
    ccall((:iree_hal_event_release, libiree), Cvoid, (Ptr{iree_hal_event_t},), event)
end

mutable struct iree_hal_event_vtable_t
    destroy::Ptr{Cvoid}
end

function iree_hal_event_destroy(event)
    ccall((:iree_hal_event_destroy, libiree), Cvoid, (Ptr{iree_hal_event_t},), event)
end

mutable struct iree_hal_executable_t end

function iree_hal_executable_retain(executable)
    ccall((:iree_hal_executable_retain, libiree), Cvoid, (Ptr{iree_hal_executable_t},), executable)
end

function iree_hal_executable_release(executable)
    ccall((:iree_hal_executable_release, libiree), Cvoid, (Ptr{iree_hal_executable_t},), executable)
end

mutable struct iree_hal_executable_vtable_t
    destroy::Ptr{Cvoid}
end

function iree_hal_executable_destroy(executable)
    ccall((:iree_hal_executable_destroy, libiree), Cvoid, (Ptr{iree_hal_executable_t},), executable)
end

@cenum iree_hal_descriptor_set_layout_flag_bits_t::UInt32 begin
    IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE = 0
end

const iree_hal_descriptor_set_layout_flags_t = UInt32

@cenum iree_hal_descriptor_type_e::UInt32 begin
    IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6
    IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
end

const iree_hal_descriptor_type_t = iree_hal_descriptor_type_e

@cenum iree_hal_descriptor_flag_bits_t::UInt32 begin
    IREE_HAL_DESCRIPTOR_FLAG_NONE = 0
    IREE_HAL_DESCRIPTOR_FLAG_READ_ONLY = 1
end

const iree_hal_descriptor_flags_t = UInt32

mutable struct iree_hal_descriptor_set_layout_binding_t
    binding::UInt32
    type::iree_hal_descriptor_type_t
    flags::iree_hal_descriptor_flags_t
end

mutable struct iree_hal_descriptor_set_layout_t end

function iree_hal_descriptor_set_layout_create(device, flags, binding_count, bindings, out_descriptor_set_layout)
    ccall((:iree_hal_descriptor_set_layout_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_hal_descriptor_set_layout_flags_t, iree_host_size_t, Ptr{iree_hal_descriptor_set_layout_binding_t}, Ptr{Ptr{iree_hal_descriptor_set_layout_t}}), device, flags, binding_count, bindings, out_descriptor_set_layout)
end

function iree_hal_descriptor_set_layout_retain(descriptor_set_layout)
    ccall((:iree_hal_descriptor_set_layout_retain, libiree), Cvoid, (Ptr{iree_hal_descriptor_set_layout_t},), descriptor_set_layout)
end

function iree_hal_descriptor_set_layout_release(descriptor_set_layout)
    ccall((:iree_hal_descriptor_set_layout_release, libiree), Cvoid, (Ptr{iree_hal_descriptor_set_layout_t},), descriptor_set_layout)
end

mutable struct iree_hal_pipeline_layout_t end

function iree_hal_pipeline_layout_create(device, push_constants, set_layout_count, set_layouts, out_pipeline_layout)
    ccall((:iree_hal_pipeline_layout_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_host_size_t, iree_host_size_t, Ptr{Ptr{iree_hal_descriptor_set_layout_t}}, Ptr{Ptr{iree_hal_pipeline_layout_t}}), device, push_constants, set_layout_count, set_layouts, out_pipeline_layout)
end

function iree_hal_pipeline_layout_retain(pipeline_layout)
    ccall((:iree_hal_pipeline_layout_retain, libiree), Cvoid, (Ptr{iree_hal_pipeline_layout_t},), pipeline_layout)
end

function iree_hal_pipeline_layout_release(pipeline_layout)
    ccall((:iree_hal_pipeline_layout_release, libiree), Cvoid, (Ptr{iree_hal_pipeline_layout_t},), pipeline_layout)
end

mutable struct iree_hal_descriptor_set_layout_vtable_t
    destroy::Ptr{Cvoid}
end

function iree_hal_descriptor_set_layout_destroy(descriptor_set_layout)
    ccall((:iree_hal_descriptor_set_layout_destroy, libiree), Cvoid, (Ptr{iree_hal_descriptor_set_layout_t},), descriptor_set_layout)
end

mutable struct iree_hal_pipeline_layout_vtable_t
    destroy::Ptr{Cvoid}
end

function iree_hal_pipeline_layout_destroy(pipeline_layout)
    ccall((:iree_hal_pipeline_layout_destroy, libiree), Cvoid, (Ptr{iree_hal_pipeline_layout_t},), pipeline_layout)
end

@cenum iree_hal_command_buffer_mode_bits_t::UInt32 begin
    IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT = 1
    IREE_HAL_COMMAND_BUFFER_MODE_NESTED = 2
    IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION = 16
    IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED = 32
end

const iree_hal_command_buffer_mode_t = UInt32

@cenum iree_hal_command_category_bits_t::UInt32 begin
    IREE_HAL_COMMAND_CATEGORY_TRANSFER = 1
    IREE_HAL_COMMAND_CATEGORY_DISPATCH = 2
    IREE_HAL_COMMAND_CATEGORY_ANY = 3
end

const iree_hal_command_category_t = UInt32

@cenum iree_hal_execution_stage_bits_t::UInt32 begin
    IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE = 1
    IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS = 2
    IREE_HAL_EXECUTION_STAGE_DISPATCH = 4
    IREE_HAL_EXECUTION_STAGE_TRANSFER = 8
    IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE = 16
    IREE_HAL_EXECUTION_STAGE_HOST = 32
end

const iree_hal_execution_stage_t = UInt32

@cenum iree_hal_execution_barrier_flag_bits_t::UInt32 begin
    IREE_HAL_EXECUTION_BARRIER_FLAG_NONE = 0
end

const iree_hal_execution_barrier_flags_t = UInt32

@cenum iree_hal_access_scope_bits_t::UInt32 begin
    IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ = 1
    IREE_HAL_ACCESS_SCOPE_CONSTANT_READ = 2
    IREE_HAL_ACCESS_SCOPE_DISPATCH_READ = 4
    IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE = 8
    IREE_HAL_ACCESS_SCOPE_TRANSFER_READ = 16
    IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE = 32
    IREE_HAL_ACCESS_SCOPE_HOST_READ = 64
    IREE_HAL_ACCESS_SCOPE_HOST_WRITE = 128
    IREE_HAL_ACCESS_SCOPE_MEMORY_READ = 256
    IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE = 512
end

const iree_hal_access_scope_t = UInt32

mutable struct iree_hal_memory_barrier_t
    source_scope::iree_hal_access_scope_t
    target_scope::iree_hal_access_scope_t
end

@cenum iree_hal_collective_kind_e::UInt32 begin
    IREE_HAL_COLLECTIVE_KIND_ALL_GATHER = 0
    IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE = 1
    IREE_HAL_COLLECTIVE_KIND_BROADCAST = 2
    IREE_HAL_COLLECTIVE_KIND_REDUCE = 3
    IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER = 4
    IREE_HAL_COLLECTIVE_KIND_SEND = 5
    IREE_HAL_COLLECTIVE_KIND_RECV = 6
    IREE_HAL_COLLECTIVE_KIND_MAX_VALUE = 6
end

const iree_hal_collective_kind_t = UInt8

@cenum iree_hal_collective_reduction_e::UInt32 begin
    IREE_HAL_COLLECTIVE_REDUCTION_SUM = 0
    IREE_HAL_COLLECTIVE_REDUCTION_PRODUCT = 1
    IREE_HAL_COLLECTIVE_REDUCTION_MINIMUM = 2
    IREE_HAL_COLLECTIVE_REDUCTION_MAXIMUM = 3
    IREE_HAL_COLLECTIVE_REDUCTION_AVERAGE = 4
end

const iree_hal_collective_reduction_t = UInt8

@cenum iree_hal_collective_element_type_e::UInt32 begin
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8 = 0
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8 = 1
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16 = 2
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16 = 3
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32 = 4
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32 = 5
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64 = 6
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64 = 7
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16 = 8
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32 = 9
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64 = 10
    IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16 = 11
end

const iree_hal_collective_element_type_t = UInt8

struct iree_hal_collective_op_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_hal_collective_op_t}, f::Symbol)
    f === :packed && return Ptr{UInt32}(x + 0)
    f === :kind && return Ptr{iree_hal_collective_kind_t}(x + 0)
    f === :reduction && return Ptr{iree_hal_collective_reduction_t}(x + 1)
    f === :element_type && return Ptr{iree_hal_collective_element_type_t}(x + 2)
    f === :reserved && return Ptr{UInt8}(x + 3)
    return getfield(x, f)
end

function Base.getproperty(x::iree_hal_collective_op_t, f::Symbol)
    r = Ref{iree_hal_collective_op_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_hal_collective_op_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_hal_collective_op_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

mutable struct iree_hal_label_color_t
    r::UInt8
    g::UInt8
    b::UInt8
    a::UInt8
end

mutable struct iree_hal_label_location_t
    file::iree_string_view_t
    line::Cint
end

# no prototype is found for this function at command_buffer.h:391:38, please use with caution
function iree_hal_label_color_unspecified()
    ccall((:iree_hal_label_color_unspecified, libiree), iree_hal_label_color_t, ())
end

function iree_hal_command_buffer_mode_format(value, out_temp)
    ccall((:iree_hal_command_buffer_mode_format, libiree), iree_string_view_t, (iree_hal_command_buffer_mode_t, Ptr{iree_bitfield_string_temp_t}), value, out_temp)
end

function iree_hal_command_category_format(value, out_temp)
    ccall((:iree_hal_command_category_format, libiree), iree_string_view_t, (iree_hal_command_category_t, Ptr{iree_bitfield_string_temp_t}), value, out_temp)
end

struct iree_hal_command_buffer_validation_state_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_hal_command_buffer_validation_state_t}, f::Symbol)
    f === :device && return Ptr{Ptr{iree_hal_device_t}}(x + 0)
    f === :is_recording && return (Ptr{UInt32}(x + 8), 0, 1)
    f === :debug_group_depth && return Ptr{Int32}(x + 12)
    return getfield(x, f)
end

function Base.getproperty(x::iree_hal_command_buffer_validation_state_t, f::Symbol)
    r = Ref{iree_hal_command_buffer_validation_state_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_hal_command_buffer_validation_state_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{iree_hal_command_buffer_validation_state_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

@cenum iree_hal_transfer_command_type_t::UInt32 begin
    IREE_HAL_TRANSFER_COMMAND_TYPE_FILL = 0
    IREE_HAL_TRANSFER_COMMAND_TYPE_COPY = 1
    IREE_HAL_TRANSFER_COMMAND_TYPE_UPDATE = 2
end

struct iree_hal_transfer_command_t
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{iree_hal_transfer_command_t}, f::Symbol)
    f === :type && return Ptr{iree_hal_transfer_command_type_t}(x + 0)
    f === :fill && return Ptr{Cvoid}(x + 8)
    f === :copy && return Ptr{Cvoid}(x + 8)
    f === :update && return Ptr{Cvoid}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_hal_transfer_command_t, f::Symbol)
    r = Ref{iree_hal_transfer_command_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_hal_transfer_command_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_hal_transfer_command_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

mutable struct iree_hal_command_buffer_vtable_t
    destroy::Ptr{Cvoid}
    dyn_cast::Ptr{Cvoid}
    _begin::Ptr{Cvoid}
    _end::Ptr{Cvoid}
    begin_debug_group::Ptr{Cvoid}
    end_debug_group::Ptr{Cvoid}
    execution_barrier::Ptr{Cvoid}
    signal_event::Ptr{Cvoid}
    reset_event::Ptr{Cvoid}
    wait_events::Ptr{Cvoid}
    discard_buffer::Ptr{Cvoid}
    fill_buffer::Ptr{Cvoid}
    update_buffer::Ptr{Cvoid}
    copy_buffer::Ptr{Cvoid}
    collective::Ptr{Cvoid}
    push_constants::Ptr{Cvoid}
    push_descriptor_set::Ptr{Cvoid}
    dispatch::Ptr{Cvoid}
    dispatch_indirect::Ptr{Cvoid}
    execute_commands::Ptr{Cvoid}
end

@cenum iree_hal_executable_caching_mode_bits_t::UInt32 begin
    IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA = 1
    IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING = 2
    IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION = 4
    IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_DEBUGGING = 8
    IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_COVERAGE = 16
    IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING = 32
    IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION = 64
end

const iree_hal_executable_caching_mode_t = UInt32

mutable struct iree_hal_executable_params_t
    caching_mode::iree_hal_executable_caching_mode_t
    executable_format::iree_string_view_t
    executable_data::iree_const_byte_span_t
    pipeline_layout_count::iree_host_size_t
    pipeline_layouts::Ptr{Ptr{iree_hal_pipeline_layout_t}}
    constant_count::iree_host_size_t
    constants::Ptr{UInt32}
end

function iree_hal_executable_params_initialize(out_executable_params)
    ccall((:iree_hal_executable_params_initialize, libiree), Cvoid, (Ptr{iree_hal_executable_params_t},), out_executable_params)
end

mutable struct iree_hal_executable_cache_t end

function iree_hal_executable_cache_create(device, identifier, loop, out_executable_cache)
    ccall((:iree_hal_executable_cache_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_string_view_t, iree_loop_t, Ptr{Ptr{iree_hal_executable_cache_t}}), device, identifier, loop, out_executable_cache)
end

function iree_hal_executable_cache_retain(executable_cache)
    ccall((:iree_hal_executable_cache_retain, libiree), Cvoid, (Ptr{iree_hal_executable_cache_t},), executable_cache)
end

function iree_hal_executable_cache_release(executable_cache)
    ccall((:iree_hal_executable_cache_release, libiree), Cvoid, (Ptr{iree_hal_executable_cache_t},), executable_cache)
end

function iree_hal_executable_cache_can_prepare_format(executable_cache, caching_mode, executable_format)
    ccall((:iree_hal_executable_cache_can_prepare_format, libiree), Bool, (Ptr{iree_hal_executable_cache_t}, iree_hal_executable_caching_mode_t, iree_string_view_t), executable_cache, caching_mode, executable_format)
end

function iree_hal_executable_cache_prepare_executable(executable_cache, executable_params, out_executable)
    ccall((:iree_hal_executable_cache_prepare_executable, libiree), iree_status_t, (Ptr{iree_hal_executable_cache_t}, Ptr{iree_hal_executable_params_t}, Ptr{Ptr{iree_hal_executable_t}}), executable_cache, executable_params, out_executable)
end

mutable struct iree_hal_executable_cache_vtable_t
    destroy::Ptr{Cvoid}
    can_prepare_format::Ptr{Cvoid}
    prepare_executable::Ptr{Cvoid}
end

function iree_hal_executable_cache_destroy(executable_cache)
    ccall((:iree_hal_executable_cache_destroy, libiree), Cvoid, (Ptr{iree_hal_executable_cache_t},), executable_cache)
end

mutable struct iree_hal_semaphore_t end

function iree_hal_semaphore_create(device, initial_value, out_semaphore)
    ccall((:iree_hal_semaphore_create, libiree), iree_status_t, (Ptr{iree_hal_device_t}, UInt64, Ptr{Ptr{iree_hal_semaphore_t}}), device, initial_value, out_semaphore)
end

function iree_hal_semaphore_retain(semaphore)
    ccall((:iree_hal_semaphore_retain, libiree), Cvoid, (Ptr{iree_hal_semaphore_t},), semaphore)
end

function iree_hal_semaphore_release(semaphore)
    ccall((:iree_hal_semaphore_release, libiree), Cvoid, (Ptr{iree_hal_semaphore_t},), semaphore)
end

function iree_hal_semaphore_query(semaphore, out_value)
    ccall((:iree_hal_semaphore_query, libiree), iree_status_t, (Ptr{iree_hal_semaphore_t}, Ptr{UInt64}), semaphore, out_value)
end

function iree_hal_semaphore_signal(semaphore, new_value)
    ccall((:iree_hal_semaphore_signal, libiree), iree_status_t, (Ptr{iree_hal_semaphore_t}, UInt64), semaphore, new_value)
end

function iree_hal_semaphore_fail(semaphore, status)
    ccall((:iree_hal_semaphore_fail, libiree), Cvoid, (Ptr{iree_hal_semaphore_t}, iree_status_t), semaphore, status)
end

function iree_hal_semaphore_wait(semaphore, value, timeout)
    ccall((:iree_hal_semaphore_wait, libiree), iree_status_t, (Ptr{iree_hal_semaphore_t}, UInt64, iree_timeout_t), semaphore, value, timeout)
end

function iree_hal_semaphore_await(semaphore, value)
    ccall((:iree_hal_semaphore_await, libiree), iree_wait_source_t, (Ptr{iree_hal_semaphore_t}, UInt64), semaphore, value)
end

mutable struct iree_hal_semaphore_list_t
    count::iree_host_size_t
    semaphores::Ptr{Ptr{iree_hal_semaphore_t}}
    payload_values::Ptr{UInt64}
end

function iree_hal_semaphore_list_empty()
    ccall((:iree_hal_semaphore_list_empty, libiree), iree_hal_semaphore_list_t, ())
end

function iree_hal_semaphore_list_signal(semaphore_list)
    ccall((:iree_hal_semaphore_list_signal, libiree), iree_status_t, (iree_hal_semaphore_list_t,), semaphore_list)
end

function iree_hal_semaphore_list_fail(semaphore_list, signal_status)
    ccall((:iree_hal_semaphore_list_fail, libiree), Cvoid, (iree_hal_semaphore_list_t, iree_status_t), semaphore_list, signal_status)
end

function iree_hal_semaphore_list_wait(semaphore_list, timeout)
    ccall((:iree_hal_semaphore_list_wait, libiree), iree_status_t, (iree_hal_semaphore_list_t, iree_timeout_t), semaphore_list, timeout)
end

mutable struct iree_hal_semaphore_vtable_t
    destroy::Ptr{Cvoid}
    query::Ptr{Cvoid}
    signal::Ptr{Cvoid}
    fail::Ptr{Cvoid}
    wait::Ptr{Cvoid}
end

function iree_hal_semaphore_destroy(semaphore)
    ccall((:iree_hal_semaphore_destroy, libiree), Cvoid, (Ptr{iree_hal_semaphore_t},), semaphore)
end

const iree_hal_device_id_t = Csize_t

@cenum iree_hal_device_feature_bits_t::UInt32 begin
    IREE_HAL_DEVICE_FEATURE_NONE = 0
    IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING = 1
    IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE = 2
    IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING = 4
end

const iree_hal_device_feature_t = UInt32

mutable struct iree_hal_device_info_t
    device_id::iree_hal_device_id_t
    path::iree_string_view_t
    name::iree_string_view_t
end

@cenum iree_hal_device_profiling_mode_bits_t::UInt32 begin
    IREE_HAL_DEVICE_PROFILING_MODE_NONE = 0
    IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS = 1
    IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS = 2
    IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS = 4
end

const iree_hal_device_profiling_mode_t = UInt32

mutable struct iree_hal_device_profiling_options_t
    mode::iree_hal_device_profiling_mode_t
    file_path::Cstring
end

@cenum iree_hal_semaphore_compatibility_bits_t::UInt32 begin
    IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE = 0
    IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT = 1
    IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT = 2
    IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL = 4
    IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_SIGNAL = 8
    IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY = 5
    IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL = 15
end

const iree_hal_semaphore_compatibility_t = UInt32

@cenum iree_hal_wait_mode_e::UInt32 begin
    IREE_HAL_WAIT_MODE_ALL = 0
    IREE_HAL_WAIT_MODE_ANY = 1
end

const iree_hal_wait_mode_t = iree_hal_wait_mode_e

function iree_hal_device_retain(device)
    ccall((:iree_hal_device_retain, libiree), Cvoid, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_release(device)
    ccall((:iree_hal_device_release, libiree), Cvoid, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_id(device)
    ccall((:iree_hal_device_id, libiree), iree_string_view_t, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_host_allocator(device)
    ccall((:iree_hal_device_host_allocator, libiree), iree_allocator_t, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_allocator(device)
    ccall((:iree_hal_device_allocator, libiree), Ptr{iree_hal_allocator_t}, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_trim(device)
    ccall((:iree_hal_device_trim, libiree), iree_status_t, (Ptr{iree_hal_device_t},), device)
end

function iree_hal_device_query_i64(device, category, key, out_value)
    ccall((:iree_hal_device_query_i64, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_string_view_t, iree_string_view_t, Ptr{Int64}), device, category, key, out_value)
end

function iree_hal_device_query_semaphore_compatibility(device, semaphore)
    ccall((:iree_hal_device_query_semaphore_compatibility, libiree), iree_hal_semaphore_compatibility_t, (Ptr{iree_hal_device_t}, Ptr{iree_hal_semaphore_t}), device, semaphore)
end

function iree_hal_device_queue_barrier(device, queue_affinity, wait_semaphore_list, signal_semaphore_list)
    ccall((:iree_hal_device_queue_barrier, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_hal_queue_affinity_t, iree_hal_semaphore_list_t, iree_hal_semaphore_list_t), device, queue_affinity, wait_semaphore_list, signal_semaphore_list)
end

function iree_hal_device_queue_flush(device, queue_affinity)
    ccall((:iree_hal_device_queue_flush, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_hal_queue_affinity_t), device, queue_affinity)
end

function iree_hal_device_wait_semaphores(device, wait_mode, semaphore_list, timeout)
    ccall((:iree_hal_device_wait_semaphores, libiree), iree_status_t, (Ptr{iree_hal_device_t}, iree_hal_wait_mode_t, iree_hal_semaphore_list_t, iree_timeout_t), device, wait_mode, semaphore_list, timeout)
end

function iree_hal_device_profiling_begin(device, options)
    ccall((:iree_hal_device_profiling_begin, libiree), iree_status_t, (Ptr{iree_hal_device_t}, Ptr{iree_hal_device_profiling_options_t}), device, options)
end

function iree_hal_device_profiling_end(device)
    ccall((:iree_hal_device_profiling_end, libiree), iree_status_t, (Ptr{iree_hal_device_t},), device)
end

mutable struct iree_hal_device_vtable_t
    destroy::Ptr{Cvoid}
    id::Ptr{Cvoid}
    host_allocator::Ptr{Cvoid}
    device_allocator::Ptr{Cvoid}
    trim::Ptr{Cvoid}
    query_i64::Ptr{Cvoid}
    create_channel::Ptr{Cvoid}
    create_command_buffer::Ptr{Cvoid}
    create_descriptor_set_layout::Ptr{Cvoid}
    create_event::Ptr{Cvoid}
    create_executable_cache::Ptr{Cvoid}
    create_pipeline_layout::Ptr{Cvoid}
    create_semaphore::Ptr{Cvoid}
    query_semaphore_compatibility::Ptr{Cvoid}
    transfer_range::Ptr{Cvoid}
    queue_alloca::Ptr{Cvoid}
    queue_dealloca::Ptr{Cvoid}
    queue_execute::Ptr{Cvoid}
    queue_flush::Ptr{Cvoid}
    wait_semaphores::Ptr{Cvoid}
    profiling_begin::Ptr{Cvoid}
    profiling_end::Ptr{Cvoid}
end

function iree_hal_device_destroy(device)
    ccall((:iree_hal_device_destroy, libiree), Cvoid, (Ptr{iree_hal_device_t},), device)
end

mutable struct iree_hal_driver_info_t
    driver_name::iree_string_view_t
    full_name::iree_string_view_t
end

mutable struct iree_hal_driver_t end

function iree_hal_driver_retain(driver)
    ccall((:iree_hal_driver_retain, libiree), Cvoid, (Ptr{iree_hal_driver_t},), driver)
end

function iree_hal_driver_release(driver)
    ccall((:iree_hal_driver_release, libiree), Cvoid, (Ptr{iree_hal_driver_t},), driver)
end

function iree_hal_driver_query_available_devices(driver, host_allocator, out_device_info_count, out_device_infos)
    ccall((:iree_hal_driver_query_available_devices, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_allocator_t, Ptr{iree_host_size_t}, Ptr{Ptr{iree_hal_device_info_t}}), driver, host_allocator, out_device_info_count, out_device_infos)
end

function iree_hal_driver_dump_device_info(driver, device_id, builder)
    ccall((:iree_hal_driver_dump_device_info, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_hal_device_id_t, Ptr{iree_string_builder_t}), driver, device_id, builder)
end

function iree_hal_driver_create_device_by_ordinal(driver, device_ordinal, param_count, params, host_allocator, out_device)
    ccall((:iree_hal_driver_create_device_by_ordinal, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_host_size_t, iree_host_size_t, Ptr{iree_string_pair_t}, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), driver, device_ordinal, param_count, params, host_allocator, out_device)
end

function iree_hal_driver_create_device_by_id(driver, device_id, param_count, params, host_allocator, out_device)
    ccall((:iree_hal_driver_create_device_by_id, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_hal_device_id_t, iree_host_size_t, Ptr{iree_string_pair_t}, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), driver, device_id, param_count, params, host_allocator, out_device)
end

function iree_hal_driver_create_device_by_path(driver, driver_name, device_path, param_count, params, host_allocator, out_device)
    ccall((:iree_hal_driver_create_device_by_path, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_string_view_t, iree_string_view_t, iree_host_size_t, Ptr{iree_string_pair_t}, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), driver, driver_name, device_path, param_count, params, host_allocator, out_device)
end

function iree_hal_driver_create_device_by_uri(driver, device_uri, host_allocator, out_device)
    ccall((:iree_hal_driver_create_device_by_uri, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_string_view_t, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), driver, device_uri, host_allocator, out_device)
end

function iree_hal_driver_create_default_device(driver, host_allocator, out_device)
    ccall((:iree_hal_driver_create_default_device, libiree), iree_status_t, (Ptr{iree_hal_driver_t}, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), driver, host_allocator, out_device)
end

mutable struct iree_hal_driver_vtable_t
    destroy::Ptr{Cvoid}
    query_available_devices::Ptr{Cvoid}
    dump_device_info::Ptr{Cvoid}
    create_device_by_id::Ptr{Cvoid}
    create_device_by_path::Ptr{Cvoid}
end

function iree_hal_driver_destroy(driver)
    ccall((:iree_hal_driver_destroy, libiree), Cvoid, (Ptr{iree_hal_driver_t},), driver)
end

mutable struct iree_hal_driver_factory_t
    self::Ptr{Cvoid}
    enumerate::Ptr{Cvoid}
    try_create::Ptr{Cvoid}
end

mutable struct iree_hal_driver_registry_t end

function iree_hal_driver_registry_default()
    ccall((:iree_hal_driver_registry_default, libiree), Ptr{iree_hal_driver_registry_t}, ())
end

function iree_hal_driver_registry_allocate(host_allocator, out_registry)
    ccall((:iree_hal_driver_registry_allocate, libiree), iree_status_t, (iree_allocator_t, Ptr{Ptr{iree_hal_driver_registry_t}}), host_allocator, out_registry)
end

function iree_hal_driver_registry_free(registry)
    ccall((:iree_hal_driver_registry_free, libiree), Cvoid, (Ptr{iree_hal_driver_registry_t},), registry)
end

function iree_hal_driver_registry_register_factory(registry, factory)
    ccall((:iree_hal_driver_registry_register_factory, libiree), iree_status_t, (Ptr{iree_hal_driver_registry_t}, Ptr{iree_hal_driver_factory_t}), registry, factory)
end

function iree_hal_driver_registry_unregister_factory(registry, factory)
    ccall((:iree_hal_driver_registry_unregister_factory, libiree), iree_status_t, (Ptr{iree_hal_driver_registry_t}, Ptr{iree_hal_driver_factory_t}), registry, factory)
end

function iree_hal_driver_registry_enumerate(registry, host_allocator, out_driver_info_count, out_driver_infos)
    ccall((:iree_hal_driver_registry_enumerate, libiree), iree_status_t, (Ptr{iree_hal_driver_registry_t}, iree_allocator_t, Ptr{iree_host_size_t}, Ptr{Ptr{iree_hal_driver_info_t}}), registry, host_allocator, out_driver_info_count, out_driver_infos)
end

function iree_hal_driver_registry_try_create(registry, driver_name, host_allocator, out_driver)
    ccall((:iree_hal_driver_registry_try_create, libiree), iree_status_t, (Ptr{iree_hal_driver_registry_t}, iree_string_view_t, iree_allocator_t, Ptr{Ptr{iree_hal_driver_t}}), registry, driver_name, host_allocator, out_driver)
end

function iree_hal_create_device(registry, device_uri, host_allocator, out_device)
    ccall((:iree_hal_create_device, libiree), iree_status_t, (Ptr{iree_hal_driver_registry_t}, iree_string_view_t, iree_allocator_t, Ptr{Ptr{iree_hal_device_t}}), registry, device_uri, host_allocator, out_device)
end

function iree_hal_parse_shape(value, shape_capacity, out_shape_rank, out_shape)
    ccall((:iree_hal_parse_shape, libiree), iree_status_t, (iree_string_view_t, iree_host_size_t, Ptr{iree_host_size_t}, Ptr{iree_hal_dim_t}), value, shape_capacity, out_shape_rank, out_shape)
end

function iree_hal_format_shape(shape_rank, shape, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_hal_format_shape, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), shape_rank, shape, buffer_capacity, buffer, out_buffer_length)
end

function iree_hal_append_shape_string(shape_rank, shape, string_builder)
    ccall((:iree_hal_append_shape_string, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, Ptr{iree_string_builder_t}), shape_rank, shape, string_builder)
end

function iree_hal_parse_element_type(value, out_element_type)
    ccall((:iree_hal_parse_element_type, libiree), iree_status_t, (iree_string_view_t, Ptr{iree_hal_element_type_t}), value, out_element_type)
end

function iree_hal_format_element_type(element_type, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_hal_format_element_type, libiree), iree_status_t, (iree_hal_element_type_t, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), element_type, buffer_capacity, buffer, out_buffer_length)
end

function iree_hal_append_element_type_string(element_type, string_builder)
    ccall((:iree_hal_append_element_type_string, libiree), iree_status_t, (iree_hal_element_type_t, Ptr{iree_string_builder_t}), element_type, string_builder)
end

function iree_hal_parse_shape_and_element_type(value, shape_capacity, out_shape_rank, out_shape, out_element_type)
    ccall((:iree_hal_parse_shape_and_element_type, libiree), iree_status_t, (iree_string_view_t, iree_host_size_t, Ptr{iree_host_size_t}, Ptr{iree_hal_dim_t}, Ptr{iree_hal_element_type_t}), value, shape_capacity, out_shape_rank, out_shape, out_element_type)
end

function iree_hal_append_shape_and_element_type_string(shape_rank, shape, element_type, string_builder)
    ccall((:iree_hal_append_shape_and_element_type_string, libiree), iree_status_t, (iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, Ptr{iree_string_builder_t}), shape_rank, shape, element_type, string_builder)
end

function iree_hal_parse_element(data_str, element_type, data_ptr)
    ccall((:iree_hal_parse_element, libiree), iree_status_t, (iree_string_view_t, iree_hal_element_type_t, iree_byte_span_t), data_str, element_type, data_ptr)
end

function iree_hal_format_element(data, element_type, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_hal_format_element, libiree), iree_status_t, (iree_const_byte_span_t, iree_hal_element_type_t, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), data, element_type, buffer_capacity, buffer, out_buffer_length)
end

function iree_hal_parse_buffer_elements(data_str, element_type, data_ptr)
    ccall((:iree_hal_parse_buffer_elements, libiree), iree_status_t, (iree_string_view_t, iree_hal_element_type_t, iree_byte_span_t), data_str, element_type, data_ptr)
end

function iree_hal_format_buffer_elements(data, shape_rank, shape, element_type, max_element_count, buffer_capacity, buffer, out_buffer_length)
    ccall((:iree_hal_format_buffer_elements, libiree), iree_status_t, (iree_const_byte_span_t, iree_host_size_t, Ptr{iree_hal_dim_t}, iree_hal_element_type_t, iree_host_size_t, iree_host_size_t, Cstring, Ptr{iree_host_size_t}), data, shape_rank, shape, element_type, max_element_count, buffer_capacity, buffer, out_buffer_length)
end

@cenum iree_vm_ref_type_bits_t::UInt32 begin
    IREE_VM_REF_TYPE_NULL = 0
    IREE_VM_REF_TYPE_MAX_VALUE = 16777214
    IREE_VM_REF_TYPE_ANY = 16777215
end

function iree_vm_ref_object_retain(ptr, type_descriptor)
    ccall((:iree_vm_ref_object_retain, libiree), Cvoid, (Ptr{Cvoid}, Ptr{iree_vm_ref_type_descriptor_t}), ptr, type_descriptor)
end

function iree_vm_ref_object_release(ptr, type_descriptor)
    ccall((:iree_vm_ref_object_release, libiree), Cvoid, (Ptr{Cvoid}, Ptr{iree_vm_ref_type_descriptor_t}), ptr, type_descriptor)
end

function iree_vm_ref_register_type(descriptor)
    ccall((:iree_vm_ref_register_type, libiree), iree_status_t, (Ptr{iree_vm_ref_type_descriptor_t},), descriptor)
end

function iree_vm_ref_type_name(type)
    ccall((:iree_vm_ref_type_name, libiree), iree_string_view_t, (iree_vm_ref_type_t,), type)
end

function iree_vm_ref_lookup_registered_type(full_name)
    ccall((:iree_vm_ref_lookup_registered_type, libiree), Ptr{iree_vm_ref_type_descriptor_t}, (iree_string_view_t,), full_name)
end

function iree_vm_ref_null()
    ccall((:iree_vm_ref_null, libiree), iree_vm_ref_t, ())
end

function iree_vm_ref_retain_inplace(ref)
    ccall((:iree_vm_ref_retain_inplace, libiree), Cvoid, (Ptr{iree_vm_ref_t},), ref)
end

function iree_vm_ref_retain(ref, out_ref)
    ccall((:iree_vm_ref_retain, libiree), Cvoid, (Ptr{iree_vm_ref_t}, Ptr{iree_vm_ref_t}), ref, out_ref)
end

function iree_vm_ref_retain_checked(ref, type, out_ref)
    ccall((:iree_vm_ref_retain_checked, libiree), iree_status_t, (Ptr{iree_vm_ref_t}, iree_vm_ref_type_t, Ptr{iree_vm_ref_t}), ref, type, out_ref)
end

function iree_vm_ref_retain_or_move(is_move, ref, out_ref)
    ccall((:iree_vm_ref_retain_or_move, libiree), Cvoid, (Cint, Ptr{iree_vm_ref_t}, Ptr{iree_vm_ref_t}), is_move, ref, out_ref)
end

function iree_vm_ref_retain_or_move_checked(is_move, ref, type, out_ref)
    ccall((:iree_vm_ref_retain_or_move_checked, libiree), iree_status_t, (Cint, Ptr{iree_vm_ref_t}, iree_vm_ref_type_t, Ptr{iree_vm_ref_t}), is_move, ref, type, out_ref)
end

function iree_vm_ref_release(ref)
    ccall((:iree_vm_ref_release, libiree), Cvoid, (Ptr{iree_vm_ref_t},), ref)
end

function iree_vm_ref_assign(ref, out_ref)
    ccall((:iree_vm_ref_assign, libiree), Cvoid, (Ptr{iree_vm_ref_t}, Ptr{iree_vm_ref_t}), ref, out_ref)
end

function iree_vm_ref_move(ref, out_ref)
    ccall((:iree_vm_ref_move, libiree), Cvoid, (Ptr{iree_vm_ref_t}, Ptr{iree_vm_ref_t}), ref, out_ref)
end

function iree_vm_ref_is_null(ref)
    ccall((:iree_vm_ref_is_null, libiree), Bool, (Ptr{iree_vm_ref_t},), ref)
end

function iree_vm_ref_equal(lhs, rhs)
    ccall((:iree_vm_ref_equal, libiree), Bool, (Ptr{iree_vm_ref_t}, Ptr{iree_vm_ref_t}), lhs, rhs)
end

@cenum iree_vm_buffer_access_bits_t::UInt32 begin
    IREE_VM_BUFFER_ACCESS_MUTABLE = 1
    IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE = 2
    IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST = 4
    IREE_VM_BUFFER_ACCESS_ORIGIN_HOST = 8
end

const iree_vm_buffer_access_t = UInt32

# no prototype is found for this function at buffer.h:205:1, please use with caution
function iree_vm_buffer_get_descriptor()
    ccall((:iree_vm_buffer_get_descriptor, libiree), Ptr{iree_vm_ref_type_descriptor_t}, ())
end

function iree_vm_buffer_isa(ref)
    ccall((:iree_vm_buffer_isa, libiree), Bool, (iree_vm_ref_t,), ref)
end

# no prototype is found for this function at buffer.h:205:1, please use with caution
function iree_vm_buffer_type_id()
    ccall((:iree_vm_buffer_type_id, libiree), iree_vm_ref_type_t, ())
end

mutable struct iree_vm_instance_t end

function iree_vm_instance_create(allocator, out_instance)
    ccall((:iree_vm_instance_create, libiree), iree_status_t, (iree_allocator_t, Ptr{Ptr{iree_vm_instance_t}}), allocator, out_instance)
end

function iree_vm_instance_retain(instance)
    ccall((:iree_vm_instance_retain, libiree), Cvoid, (Ptr{iree_vm_instance_t},), instance)
end

function iree_vm_instance_release(instance)
    ccall((:iree_vm_instance_release, libiree), Cvoid, (Ptr{iree_vm_instance_t},), instance)
end

function iree_vm_instance_allocator(instance)
    ccall((:iree_vm_instance_allocator, libiree), iree_allocator_t, (Ptr{iree_vm_instance_t},), instance)
end

@cenum iree_vm_stack_frame_type_e::UInt32 begin
    IREE_VM_STACK_FRAME_EXTERNAL = 0
    IREE_VM_STACK_FRAME_NATIVE = 1
    IREE_VM_STACK_FRAME_BYTECODE = 2
    IREE_VM_STACK_FRAME_WAIT = 3
end

const iree_vm_stack_frame_type_t = iree_vm_stack_frame_type_e

mutable struct iree_vm_module_state_t end

const iree_vm_source_offset_t = Int64

@cenum iree_vm_function_linkage_e::UInt32 begin
    IREE_VM_FUNCTION_LINKAGE_INTERNAL = 0
    IREE_VM_FUNCTION_LINKAGE_IMPORT = 1
    IREE_VM_FUNCTION_LINKAGE_EXPORT = 2
    IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL = 3
end

const iree_vm_function_linkage_t = iree_vm_function_linkage_e

struct iree_vm_function_signature_t
    calling_convention::iree_string_view_t
end

mutable struct iree_vm_module_signature_t
    version::UInt32
    attr_count::iree_host_size_t
    import_function_count::iree_host_size_t
    export_function_count::iree_host_size_t
    internal_function_count::iree_host_size_t
end

@cenum iree_vm_module_dependency_flag_bits_t::UInt32 begin
    IREE_VM_MODULE_DEPENDENCY_FLAG_NONE = 0
    IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED = 1
    IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL = 2
end

const iree_vm_module_dependency_flags_t = UInt32

struct iree_vm_module_dependency_t
    name::iree_string_view_t
    minimum_version::UInt32
    flags::iree_vm_module_dependency_flags_t
end

# typedef iree_status_t ( IREE_API_PTR * iree_vm_module_dependency_callback_t ) ( void * user_data_ptr , const iree_vm_module_dependency_t * dependency )
const iree_vm_module_dependency_callback_t = Ptr{Cvoid}

mutable struct iree_vm_register_list_t
    size::UInt16
    registers::Ptr{UInt16}
end

function iree_vm_function_call_get_cconv_fragments(signature, out_arguments, out_results)
    ccall((:iree_vm_function_call_get_cconv_fragments, libiree), iree_status_t, (Ptr{iree_vm_function_signature_t}, Ptr{iree_string_view_t}, Ptr{iree_string_view_t}), signature, out_arguments, out_results)
end

function iree_vm_function_call_is_variadic_cconv(cconv)
    ccall((:iree_vm_function_call_is_variadic_cconv, libiree), Bool, (iree_string_view_t,), cconv)
end

function iree_vm_function_call_count_arguments_and_results(signature, out_argument_count, out_result_count)
    ccall((:iree_vm_function_call_count_arguments_and_results, libiree), iree_status_t, (Ptr{iree_vm_function_signature_t}, Ptr{iree_host_size_t}, Ptr{iree_host_size_t}), signature, out_argument_count, out_result_count)
end

function iree_vm_function_call_compute_cconv_fragment_size(cconv_fragment, segment_size_list, out_required_size)
    ccall((:iree_vm_function_call_compute_cconv_fragment_size, libiree), iree_status_t, (iree_string_view_t, Ptr{iree_vm_register_list_t}, Ptr{iree_host_size_t}), cconv_fragment, segment_size_list, out_required_size)
end

@cenum iree_vm_source_location_format_flag_bits_e::UInt32 begin
    IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_NONE = 0
    IREE_VM_SOURCE_LOCATION_FORMAT_FLAG_SINGLE_LINE = 1
end

const iree_vm_source_location_format_flags_t = UInt32

mutable struct iree_vm_source_location_t
    self::Ptr{Cvoid}
    data::NTuple{2, UInt64}
    format::Ptr{Cvoid}
end

function iree_vm_source_location_format(source_location, flags, builder)
    ccall((:iree_vm_source_location_format, libiree), iree_status_t, (Ptr{iree_vm_source_location_t}, iree_vm_source_location_format_flags_t, Ptr{iree_string_builder_t}), source_location, flags, builder)
end

@cenum iree_vm_signal_e::UInt32 begin
    IREE_VM_SIGNAL_RESUME = 0
    IREE_VM_SIGNAL_SUSPEND = 1
    IREE_VM_SIGNAL_LOW_MEMORY = 2
end

const iree_vm_signal_t = iree_vm_signal_e

@cenum __JL_Ctag_26::UInt32 begin
    IREE_TRACING_PLOT_TYPE_NUMBER = 0
    IREE_TRACING_PLOT_TYPE_MEMORY = 1
    IREE_TRACING_PLOT_TYPE_PERCENTAGE = 2
end

@cenum __JL_Ctag_27::UInt32 begin
    IREE_TRACING_MESSAGE_LEVEL_ERROR = 16711680
    IREE_TRACING_MESSAGE_LEVEL_WARNING = 16776960
    IREE_TRACING_MESSAGE_LEVEL_INFO = 16777215
    IREE_TRACING_MESSAGE_LEVEL_VERBOSE = 12632256
    IREE_TRACING_MESSAGE_LEVEL_DEBUG = 65280
end

@cenum iree_vm_invocation_flag_bits_t::UInt32 begin
    IREE_VM_INVOCATION_FLAG_NONE = 0
    IREE_VM_INVOCATION_FLAG_TRACE_EXECUTION = 1
    IREE_VM_INVOCATION_FLAG_TRACE_INLINE = 2
end

@cenum iree_vm_wait_type_e::UInt32 begin
    IREE_VM_WAIT_UNTIL = 0
    IREE_VM_WAIT_ANY = 1
    IREE_VM_WAIT_ALL = 2
end

const iree_vm_wait_type_t = UInt8

mutable struct iree_vm_wait_frame_t
    wait_type::iree_vm_wait_type_t
    wait_status::iree_status_t
    deadline_ns::iree_time_t
    count::iree_host_size_t
    wait_sources::Ptr{iree_wait_source_t}
end

mutable struct iree_vm_wait_result_t
    status::iree_status_t
end

# typedef void ( IREE_API_PTR * iree_vm_stack_frame_cleanup_fn_t ) ( iree_vm_stack_frame_t * frame )
const iree_vm_stack_frame_cleanup_fn_t = Ptr{Cvoid}

function iree_vm_stack_reset(stack)
    ccall((:iree_vm_stack_reset, libiree), Cvoid, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_deinitialize(stack)
    ccall((:iree_vm_stack_deinitialize, libiree), Cvoid, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_allocate(flags, state_resolver, allocator, out_stack)
    ccall((:iree_vm_stack_allocate, libiree), iree_status_t, (iree_vm_invocation_flags_t, iree_vm_state_resolver_t, iree_allocator_t, Ptr{Ptr{iree_vm_stack_t}}), flags, state_resolver, allocator, out_stack)
end

function iree_vm_stack_free(stack)
    ccall((:iree_vm_stack_free, libiree), Cvoid, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_invocation_flags(stack)
    ccall((:iree_vm_stack_invocation_flags, libiree), iree_vm_invocation_flags_t, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_wait_enter(stack, wait_type, wait_count, timeout, trace_zone, out_wait_frame)
    ccall((:iree_vm_stack_wait_enter, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_wait_type_t, iree_host_size_t, iree_timeout_t, iree_zone_id_t, Ptr{Ptr{iree_vm_wait_frame_t}}), stack, wait_type, wait_count, timeout, trace_zone, out_wait_frame)
end

function iree_vm_stack_wait_leave(stack, out_wait_result)
    ccall((:iree_vm_stack_wait_leave, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, Ptr{iree_vm_wait_result_t}), stack, out_wait_result)
end

function iree_vm_stack_function_leave(stack)
    ccall((:iree_vm_stack_function_leave, libiree), iree_status_t, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_format_backtrace(stack, builder)
    ccall((:iree_vm_stack_format_backtrace, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, Ptr{iree_string_builder_t}), stack, builder)
end

function iree_vm_stack_suspend_trace_zones(stack)
    ccall((:iree_vm_stack_suspend_trace_zones, libiree), Cvoid, (Ptr{iree_vm_stack_t},), stack)
end

function iree_vm_stack_resume_trace_zones(stack)
    ccall((:iree_vm_stack_resume_trace_zones, libiree), Cvoid, (Ptr{iree_vm_stack_t},), stack)
end

mutable struct iree_vm_context_t end

@cenum iree_vm_context_flag_bits_t::UInt32 begin
    IREE_VM_CONTEXT_FLAG_NONE = 0
    IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION = 1
    IREE_VM_CONTEXT_FLAG_CONCURRENT = 2
end

const iree_vm_context_flags_t = UInt32

const iree_vm_context_id_t = intptr_t

function iree_vm_context_create(instance, flags, allocator, out_context)
    ccall((:iree_vm_context_create, libiree), iree_status_t, (Ptr{iree_vm_instance_t}, iree_vm_context_flags_t, iree_allocator_t, Ptr{Ptr{iree_vm_context_t}}), instance, flags, allocator, out_context)
end

function iree_vm_context_retain(context)
    ccall((:iree_vm_context_retain, libiree), Cvoid, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_release(context)
    ccall((:iree_vm_context_release, libiree), Cvoid, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_id(context)
    ccall((:iree_vm_context_id, libiree), iree_vm_context_id_t, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_flags(context)
    ccall((:iree_vm_context_flags, libiree), iree_vm_context_flags_t, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_freeze(context)
    ccall((:iree_vm_context_freeze, libiree), iree_status_t, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_state_resolver(context)
    ccall((:iree_vm_context_state_resolver, libiree), iree_vm_state_resolver_t, (Ptr{iree_vm_context_t},), context)
end

function iree_vm_context_notify(context, signal)
    ccall((:iree_vm_context_notify, libiree), iree_status_t, (Ptr{iree_vm_context_t}, iree_vm_signal_t), context, signal)
end

const iree_vm_size_t = Int32

@cenum iree_vm_value_type_e::UInt32 begin
    IREE_VM_VALUE_TYPE_NONE = 0
    IREE_VM_VALUE_TYPE_I8 = 1
    IREE_VM_VALUE_TYPE_I16 = 2
    IREE_VM_VALUE_TYPE_I32 = 3
    IREE_VM_VALUE_TYPE_I64 = 4
    IREE_VM_VALUE_TYPE_F32 = 5
    IREE_VM_VALUE_TYPE_F64 = 6
    IREE_VM_VALUE_TYPE_MAX = 6
    IREE_VM_VALUE_TYPE_COUNT = 7
end

const iree_vm_value_type_t = iree_vm_value_type_e

struct iree_vm_value_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_value_t}, f::Symbol)
    f === :type && return Ptr{iree_vm_value_type_t}(x + 0)
    f === :i8 && return Ptr{Int8}(x + 8)
    f === :i16 && return Ptr{Int16}(x + 8)
    f === :i32 && return Ptr{Int32}(x + 8)
    f === :i64 && return Ptr{Int64}(x + 8)
    f === :f32 && return Ptr{Cfloat}(x + 8)
    f === :f64 && return Ptr{Cdouble}(x + 8)
    f === :value_storage && return Ptr{NTuple{8, UInt8}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_value_t, f::Symbol)
    r = Ref{iree_vm_value_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_value_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_value_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

# no prototype is found for this function at value.h:61:31, please use with caution
function iree_vm_value_make_none()
    ccall((:iree_vm_value_make_none, libiree), iree_vm_value_t, ())
end

function iree_vm_value_make_i8(value)
    ccall((:iree_vm_value_make_i8, libiree), iree_vm_value_t, (Int8,), value)
end

function iree_vm_value_make_i16(value)
    ccall((:iree_vm_value_make_i16, libiree), iree_vm_value_t, (Int16,), value)
end

function iree_vm_value_make_i32(value)
    ccall((:iree_vm_value_make_i32, libiree), iree_vm_value_t, (Int32,), value)
end

function iree_vm_value_get_i32(value)
    ccall((:iree_vm_value_get_i32, libiree), Int32, (Ptr{iree_vm_value_t},), value)
end

function iree_vm_value_make_i64(value)
    ccall((:iree_vm_value_make_i64, libiree), iree_vm_value_t, (Int64,), value)
end

function iree_vm_value_get_i64(value)
    ccall((:iree_vm_value_get_i64, libiree), Int64, (Ptr{iree_vm_value_t},), value)
end

function iree_vm_value_make_f32(value)
    ccall((:iree_vm_value_make_f32, libiree), iree_vm_value_t, (Cfloat,), value)
end

function iree_vm_value_get_f32(value)
    ccall((:iree_vm_value_get_f32, libiree), Cfloat, (Ptr{iree_vm_value_t},), value)
end

function iree_vm_value_make_f64(value)
    ccall((:iree_vm_value_make_f64, libiree), iree_vm_value_t, (Cdouble,), value)
end

function iree_vm_value_get_f64(value)
    ccall((:iree_vm_value_get_f64, libiree), Cdouble, (Ptr{iree_vm_value_t},), value)
end

struct iree_vm_type_def_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_type_def_t}, f::Symbol)
    f === :value_type && return (Ptr{iree_vm_value_type_t}(x + 0), 0, 8)
    f === :ref_type && return (Ptr{iree_vm_ref_type_t}(x + 0), 8, 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_type_def_t, f::Symbol)
    r = Ref{iree_vm_type_def_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_type_def_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{iree_vm_type_def_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

function iree_vm_type_def_make_variant_type()
    ccall((:iree_vm_type_def_make_variant_type, libiree), iree_vm_type_def_t, ())
end

function iree_vm_type_def_make_value_type(value_type)
    ccall((:iree_vm_type_def_make_value_type, libiree), iree_vm_type_def_t, (iree_vm_value_type_t,), value_type)
end

function iree_vm_type_def_make_ref_type(ref_type)
    ccall((:iree_vm_type_def_make_ref_type, libiree), iree_vm_type_def_t, (iree_vm_ref_type_t,), ref_type)
end

struct iree_vm_variant_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_variant_t}, f::Symbol)
    f === :type && return Ptr{iree_vm_type_def_t}(x + 0)
    f === :i8 && return Ptr{Int8}(x + 8)
    f === :i16 && return Ptr{Int16}(x + 8)
    f === :i32 && return Ptr{Int32}(x + 8)
    f === :i64 && return Ptr{Int64}(x + 8)
    f === :f32 && return Ptr{Cfloat}(x + 8)
    f === :f64 && return Ptr{Cdouble}(x + 8)
    f === :ref && return Ptr{iree_vm_ref_t}(x + 8)
    f === :value_storage && return Ptr{NTuple{8, UInt8}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_variant_t, f::Symbol)
    r = Ref{iree_vm_variant_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_variant_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_variant_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

mutable struct iree_vm_list_t end

function iree_vm_list_storage_size(element_type, capacity)
    ccall((:iree_vm_list_storage_size, libiree), iree_host_size_t, (Ptr{iree_vm_type_def_t}, iree_host_size_t), element_type, capacity)
end

function iree_vm_list_initialize(storage, element_type, capacity, out_list)
    ccall((:iree_vm_list_initialize, libiree), iree_status_t, (iree_byte_span_t, Ptr{iree_vm_type_def_t}, iree_host_size_t, Ptr{Ptr{iree_vm_list_t}}), storage, element_type, capacity, out_list)
end

function iree_vm_list_deinitialize(list)
    ccall((:iree_vm_list_deinitialize, libiree), Cvoid, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_create(element_type, initial_capacity, allocator, out_list)
    ccall((:iree_vm_list_create, libiree), iree_status_t, (Ptr{iree_vm_type_def_t}, iree_host_size_t, iree_allocator_t, Ptr{Ptr{iree_vm_list_t}}), element_type, initial_capacity, allocator, out_list)
end

function iree_vm_list_clone(source, host_allocator, out_target)
    ccall((:iree_vm_list_clone, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_allocator_t, Ptr{Ptr{iree_vm_list_t}}), source, host_allocator, out_target)
end

function iree_vm_list_retain(list)
    ccall((:iree_vm_list_retain, libiree), Cvoid, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_release(list)
    ccall((:iree_vm_list_release, libiree), Cvoid, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_element_type(list)
    ccall((:iree_vm_list_element_type, libiree), iree_vm_type_def_t, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_capacity(list)
    ccall((:iree_vm_list_capacity, libiree), iree_host_size_t, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_reserve(list, minimum_capacity)
    ccall((:iree_vm_list_reserve, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t), list, minimum_capacity)
end

function iree_vm_list_size(list)
    ccall((:iree_vm_list_size, libiree), iree_host_size_t, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_resize(list, new_size)
    ccall((:iree_vm_list_resize, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t), list, new_size)
end

function iree_vm_list_clear(list)
    ccall((:iree_vm_list_clear, libiree), Cvoid, (Ptr{iree_vm_list_t},), list)
end

function iree_vm_list_get_value(list, i, out_value)
    ccall((:iree_vm_list_get_value, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_value_t}), list, i, out_value)
end

function iree_vm_list_get_value_as(list, i, value_type, out_value)
    ccall((:iree_vm_list_get_value_as, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, iree_vm_value_type_t, Ptr{iree_vm_value_t}), list, i, value_type, out_value)
end

function iree_vm_list_set_value(list, i, value)
    ccall((:iree_vm_list_set_value, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_value_t}), list, i, value)
end

function iree_vm_list_push_value(list, value)
    ccall((:iree_vm_list_push_value, libiree), iree_status_t, (Ptr{iree_vm_list_t}, Ptr{iree_vm_value_t}), list, value)
end

function iree_vm_list_get_ref_deref(list, i, type_descriptor)
    ccall((:iree_vm_list_get_ref_deref, libiree), Ptr{Cvoid}, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_ref_type_descriptor_t}), list, i, type_descriptor)
end

function iree_vm_list_get_ref_assign(list, i, out_value)
    ccall((:iree_vm_list_get_ref_assign, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_ref_t}), list, i, out_value)
end

function iree_vm_list_get_ref_retain(list, i, out_value)
    ccall((:iree_vm_list_get_ref_retain, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_ref_t}), list, i, out_value)
end

function iree_vm_list_set_ref_retain(list, i, value)
    ccall((:iree_vm_list_set_ref_retain, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_ref_t}), list, i, value)
end

function iree_vm_list_push_ref_retain(list, value)
    ccall((:iree_vm_list_push_ref_retain, libiree), iree_status_t, (Ptr{iree_vm_list_t}, Ptr{iree_vm_ref_t}), list, value)
end

function iree_vm_list_set_ref_move(list, i, value)
    ccall((:iree_vm_list_set_ref_move, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_ref_t}), list, i, value)
end

function iree_vm_list_push_ref_move(list, value)
    ccall((:iree_vm_list_push_ref_move, libiree), iree_status_t, (Ptr{iree_vm_list_t}, Ptr{iree_vm_ref_t}), list, value)
end

function iree_vm_list_pop_front_ref_move(list, out_value)
    ccall((:iree_vm_list_pop_front_ref_move, libiree), iree_status_t, (Ptr{iree_vm_list_t}, Ptr{iree_vm_ref_t}), list, out_value)
end

function iree_vm_list_get_variant(list, i, out_value)
    ccall((:iree_vm_list_get_variant, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_variant_t}), list, i, out_value)
end

function iree_vm_list_set_variant(list, i, value)
    ccall((:iree_vm_list_set_variant, libiree), iree_status_t, (Ptr{iree_vm_list_t}, iree_host_size_t, Ptr{iree_vm_variant_t}), list, i, value)
end

function iree_vm_list_push_variant(list, value)
    ccall((:iree_vm_list_push_variant, libiree), iree_status_t, (Ptr{iree_vm_list_t}, Ptr{iree_vm_variant_t}), list, value)
end

function iree_vm_list_retain_ref(value)
    ccall((:iree_vm_list_retain_ref, libiree), iree_vm_ref_t, (Ptr{iree_vm_list_t},), value)
end

function iree_vm_list_move_ref(value)
    ccall((:iree_vm_list_move_ref, libiree), iree_vm_ref_t, (Ptr{iree_vm_list_t},), value)
end

function iree_vm_list_deref(ref)
    ccall((:iree_vm_list_deref, libiree), Ptr{iree_vm_list_t}, (iree_vm_ref_t,), ref)
end

function iree_vm_list_check_deref(ref, out_ptr)
    ccall((:iree_vm_list_check_deref, libiree), iree_status_t, (iree_vm_ref_t, Ptr{Ptr{iree_vm_list_t}}), ref, out_ptr)
end

function iree_vm_list_check_deref_or_null(ref, out_ptr)
    ccall((:iree_vm_list_check_deref_or_null, libiree), iree_status_t, (iree_vm_ref_t, Ptr{Ptr{iree_vm_list_t}}), ref, out_ptr)
end

# no prototype is found for this function at list.h:200:1, please use with caution
function iree_vm_list_get_descriptor()
    ccall((:iree_vm_list_get_descriptor, libiree), Ptr{iree_vm_ref_type_descriptor_t}, ())
end

function iree_vm_list_isa(ref)
    ccall((:iree_vm_list_isa, libiree), Bool, (iree_vm_ref_t,), ref)
end

# no prototype is found for this function at list.h:200:1, please use with caution
function iree_vm_list_type_id()
    ccall((:iree_vm_list_type_id, libiree), iree_vm_ref_type_t, ())
end

mutable struct iree_vm_invocation_t end

mutable struct iree_vm_invocation_policy_t end

mutable struct iree_vm_invoke_state_t
    context::Ptr{iree_vm_context_t}
    status::iree_status_t
    cconv_results::iree_string_view_t
    results::iree_byte_span_t
    stack::Ptr{iree_vm_stack_t}
    stack_storage::NTuple{8192, UInt8}
end

function iree_vm_resume_invoke(state)
    ccall((:iree_vm_resume_invoke, libiree), iree_status_t, (Ptr{iree_vm_invoke_state_t},), state)
end

function iree_vm_wait_invoke(state, wait_frame, deadline_ns)
    ccall((:iree_vm_wait_invoke, libiree), iree_status_t, (Ptr{iree_vm_invoke_state_t}, Ptr{iree_vm_wait_frame_t}, iree_time_t), state, wait_frame, deadline_ns)
end

function iree_vm_end_invoke(state, outputs, out_status)
    ccall((:iree_vm_end_invoke, libiree), iree_status_t, (Ptr{iree_vm_invoke_state_t}, Ptr{iree_vm_list_t}, Ptr{iree_status_t}), state, outputs, out_status)
end

function iree_vm_abort_invoke(state)
    ccall((:iree_vm_abort_invoke, libiree), Cvoid, (Ptr{iree_vm_invoke_state_t},), state)
end

const iree_vm_invocation_id_t = intptr_t

# typedef iree_status_t ( IREE_API_PTR * iree_vm_async_invoke_callback_fn_t ) ( void * user_data , iree_loop_t loop , iree_status_t status , iree_vm_list_t * outputs )
const iree_vm_async_invoke_callback_fn_t = Ptr{Cvoid}

struct iree_vm_async_invoke_state_t
    data::NTuple{8304, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_async_invoke_state_t}, f::Symbol)
    f === :begin_params && return Ptr{Cvoid}(x + 0)
    f === :base && return Ptr{iree_vm_invoke_state_t}(x + 0)
    f === :invocation_id && return Ptr{iree_vm_invocation_id_t}(x + 8248)
    f === :deadline_ns && return Ptr{iree_time_t}(x + 8256)
    f === :host_allocator && return Ptr{iree_allocator_t}(x + 8264)
    f === :outputs && return Ptr{Ptr{iree_vm_list_t}}(x + 8280)
    f === :callback && return Ptr{iree_vm_async_invoke_callback_fn_t}(x + 8288)
    f === :user_data && return Ptr{Ptr{Cvoid}}(x + 8296)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_async_invoke_state_t, f::Symbol)
    r = Ref{iree_vm_async_invoke_state_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_async_invoke_state_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_async_invoke_state_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_invocation_retain(invocation)
    ccall((:iree_vm_invocation_retain, libiree), iree_status_t, (Ptr{iree_vm_invocation_t},), invocation)
end

function iree_vm_invocation_release(invocation)
    ccall((:iree_vm_invocation_release, libiree), iree_status_t, (Ptr{iree_vm_invocation_t},), invocation)
end

function iree_vm_invocation_query_status(invocation)
    ccall((:iree_vm_invocation_query_status, libiree), iree_status_t, (Ptr{iree_vm_invocation_t},), invocation)
end

function iree_vm_invocation_outputs(invocation)
    ccall((:iree_vm_invocation_outputs, libiree), Ptr{iree_vm_list_t}, (Ptr{iree_vm_invocation_t},), invocation)
end

function iree_vm_invocation_await(invocation, deadline)
    ccall((:iree_vm_invocation_await, libiree), iree_status_t, (Ptr{iree_vm_invocation_t}, iree_time_t), invocation, deadline)
end

function iree_vm_invocation_cancel(invocation)
    ccall((:iree_vm_invocation_cancel, libiree), Cvoid, (Ptr{iree_vm_invocation_t},), invocation)
end

@cenum iree_vm_native_import_flag_bits_e::UInt32 begin
    IREE_VM_NATIVE_IMPORT_REQUIRED = 1
    IREE_VM_NATIVE_IMPORT_OPTIONAL = 2
end

const iree_vm_native_import_flags_t = UInt32

struct iree_vm_native_import_descriptor_t
    flags::iree_vm_native_import_flags_t
    full_name::iree_string_view_t
end

struct iree_vm_native_export_descriptor_t
    local_name::iree_string_view_t
    calling_convention::iree_string_view_t
    attr_count::iree_host_size_t
    attrs::Ptr{iree_string_pair_t}
end

# typedef iree_status_t ( IREE_API_PTR * iree_vm_native_function_target_t ) ( iree_vm_stack_t * stack , void * module , void * module_state )
const iree_vm_native_function_target_t = Ptr{Cvoid}

@cenum iree_vm_native_function_flag_bits_t::UInt32 begin
    IREE_VM_NATIVE_FUNCTION_CALL_BEGIN = 1
    IREE_VM_NATIVE_FUNCTION_CALL_RESUME = 2
end

# typedef iree_status_t ( IREE_API_PTR * iree_vm_native_function_shim_t ) ( iree_vm_stack_t * stack , iree_vm_native_function_flags_t flags , iree_byte_span_t args_storage , iree_byte_span_t rets_storage , iree_vm_native_function_target_t target_fn , void * module , void * module_state )
const iree_vm_native_function_shim_t = Ptr{Cvoid}

struct iree_vm_native_function_ptr_t
    shim::iree_vm_native_function_shim_t
    target::iree_vm_native_function_target_t
end

mutable struct iree_vm_native_module_descriptor_t
    name::iree_string_view_t
    version::UInt32
    attr_count::iree_host_size_t
    attrs::Ptr{iree_string_pair_t}
    dependency_count::iree_host_size_t
    dependencies::Ptr{iree_vm_module_dependency_t}
    import_count::iree_host_size_t
    imports::Ptr{iree_vm_native_import_descriptor_t}
    export_count::iree_host_size_t
    exports::Ptr{iree_vm_native_export_descriptor_t}
    function_count::iree_host_size_t
    functions::Ptr{iree_vm_native_function_ptr_t}
end

function iree_vm_native_module_size()
    ccall((:iree_vm_native_module_size, libiree), iree_host_size_t, ())
end

mutable struct iree_vm_abi_v_t
    unused::Cint
end

function iree_vm_abi_v_checked_deref(buffer)
    ccall((:iree_vm_abi_v_checked_deref, libiree), Ptr{iree_vm_abi_v_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_v_reset(value)
    ccall((:iree_vm_abi_v_reset, libiree), Cvoid, (Ptr{iree_vm_abi_v_t},), value)
end

struct iree_vm_abi_i_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_i_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_i_t, f::Symbol)
    r = Ref{iree_vm_abi_i_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_i_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_i_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_i_checked_deref(buffer)
    ccall((:iree_vm_abi_i_checked_deref, libiree), Ptr{iree_vm_abi_i_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_i_reset(value)
    ccall((:iree_vm_abi_i_reset, libiree), Cvoid, (Ptr{iree_vm_abi_i_t},), value)
end

struct iree_vm_abi_I_t
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_I_t}, f::Symbol)
    f === :i0 && return Ptr{Int64}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_I_t, f::Symbol)
    r = Ref{iree_vm_abi_I_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_I_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_I_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_I_checked_deref(buffer)
    ccall((:iree_vm_abi_I_checked_deref, libiree), Ptr{iree_vm_abi_I_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_I_reset(value)
    ccall((:iree_vm_abi_I_reset, libiree), Cvoid, (Ptr{iree_vm_abi_I_t},), value)
end

struct iree_vm_abi_f_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_f_t}, f::Symbol)
    f === :f0 && return Ptr{Cfloat}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_f_t, f::Symbol)
    r = Ref{iree_vm_abi_f_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_f_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_f_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_f_checked_deref(buffer)
    ccall((:iree_vm_abi_f_checked_deref, libiree), Ptr{iree_vm_abi_f_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_f_reset(value)
    ccall((:iree_vm_abi_f_reset, libiree), Cvoid, (Ptr{iree_vm_abi_f_t},), value)
end

struct iree_vm_abi_ii_t
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_ii_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_ii_t, f::Symbol)
    r = Ref{iree_vm_abi_ii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_ii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_ii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_ii_checked_deref(buffer)
    ccall((:iree_vm_abi_ii_checked_deref, libiree), Ptr{iree_vm_abi_ii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_ii_reset(value)
    ccall((:iree_vm_abi_ii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_ii_t},), value)
end

struct iree_vm_abi_iI_t
    data::NTuple{12, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_iI_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_iI_t, f::Symbol)
    r = Ref{iree_vm_abi_iI_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_iI_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_iI_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_iI_checked_deref(buffer)
    ccall((:iree_vm_abi_iI_checked_deref, libiree), Ptr{iree_vm_abi_iI_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_iI_reset(value)
    ccall((:iree_vm_abi_iI_reset, libiree), Cvoid, (Ptr{iree_vm_abi_iI_t},), value)
end

struct iree_vm_abi_II_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_II_t}, f::Symbol)
    f === :i0 && return Ptr{Int64}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_II_t, f::Symbol)
    r = Ref{iree_vm_abi_II_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_II_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_II_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_II_checked_deref(buffer)
    ccall((:iree_vm_abi_II_checked_deref, libiree), Ptr{iree_vm_abi_II_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_II_reset(value)
    ccall((:iree_vm_abi_II_reset, libiree), Cvoid, (Ptr{iree_vm_abi_II_t},), value)
end

struct iree_vm_abi_iii_t
    data::NTuple{12, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_iii_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 4)
    f === :i2 && return Ptr{Int32}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_iii_t, f::Symbol)
    r = Ref{iree_vm_abi_iii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_iii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_iii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_iii_checked_deref(buffer)
    ccall((:iree_vm_abi_iii_checked_deref, libiree), Ptr{iree_vm_abi_iii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_iii_reset(value)
    ccall((:iree_vm_abi_iii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_iii_t},), value)
end

struct iree_vm_abi_iiii_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_iiii_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 4)
    f === :i2 && return Ptr{Int32}(x + 8)
    f === :i3 && return Ptr{Int32}(x + 12)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_iiii_t, f::Symbol)
    r = Ref{iree_vm_abi_iiii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_iiii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_iiii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_iiii_checked_deref(buffer)
    ccall((:iree_vm_abi_iiii_checked_deref, libiree), Ptr{iree_vm_abi_iiii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_iiii_reset(value)
    ccall((:iree_vm_abi_iiii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_iiii_t},), value)
end

struct iree_vm_abi_irIi_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_irIi_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 4)
    f === :i2 && return Ptr{Int64}(x + 20)
    f === :i3 && return Ptr{Int32}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_irIi_t, f::Symbol)
    r = Ref{iree_vm_abi_irIi_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_irIi_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_irIi_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_irIi_checked_deref(buffer)
    ccall((:iree_vm_abi_irIi_checked_deref, libiree), Ptr{iree_vm_abi_irIi_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_irIi_reset(value)
    ccall((:iree_vm_abi_irIi_reset, libiree), Cvoid, (Ptr{iree_vm_abi_irIi_t},), value)
end

struct iree_vm_abi_irII_t
    data::NTuple{36, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_irII_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 4)
    f === :i2 && return Ptr{Int64}(x + 20)
    f === :i3 && return Ptr{Int64}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_irII_t, f::Symbol)
    r = Ref{iree_vm_abi_irII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_irII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_irII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_irII_checked_deref(buffer)
    ccall((:iree_vm_abi_irII_checked_deref, libiree), Ptr{iree_vm_abi_irII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_irII_reset(value)
    ccall((:iree_vm_abi_irII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_irII_t},), value)
end

struct iree_vm_abi_r_t
    data::NTuple{16, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_r_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_r_t, f::Symbol)
    r = Ref{iree_vm_abi_r_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_r_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_r_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_r_checked_deref(buffer)
    ccall((:iree_vm_abi_r_checked_deref, libiree), Ptr{iree_vm_abi_r_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_r_reset(value)
    ccall((:iree_vm_abi_r_reset, libiree), Cvoid, (Ptr{iree_vm_abi_r_t},), value)
end

struct iree_vm_abi_rr_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rr_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rr_t, f::Symbol)
    r = Ref{iree_vm_abi_rr_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rr_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rr_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rr_checked_deref(buffer)
    ccall((:iree_vm_abi_rr_checked_deref, libiree), Ptr{iree_vm_abi_rr_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rr_reset(value)
    ccall((:iree_vm_abi_rr_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rr_t},), value)
end

struct iree_vm_abi_rrr_t
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrr_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrr_t, f::Symbol)
    r = Ref{iree_vm_abi_rrr_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrr_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrr_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrr_checked_deref(buffer)
    ccall((:iree_vm_abi_rrr_checked_deref, libiree), Ptr{iree_vm_abi_rrr_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrr_reset(value)
    ccall((:iree_vm_abi_rrr_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrr_t},), value)
end

struct iree_vm_abi_ri_t
    data::NTuple{20, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_ri_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_ri_t, f::Symbol)
    r = Ref{iree_vm_abi_ri_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_ri_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_ri_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_ri_checked_deref(buffer)
    ccall((:iree_vm_abi_ri_checked_deref, libiree), Ptr{iree_vm_abi_ri_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_ri_reset(value)
    ccall((:iree_vm_abi_ri_reset, libiree), Cvoid, (Ptr{iree_vm_abi_ri_t},), value)
end

struct iree_vm_abi_rI_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rI_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rI_t, f::Symbol)
    r = Ref{iree_vm_abi_rI_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rI_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rI_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rI_checked_deref(buffer)
    ccall((:iree_vm_abi_rI_checked_deref, libiree), Ptr{iree_vm_abi_rI_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rI_reset(value)
    ccall((:iree_vm_abi_rI_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rI_t},), value)
end

struct iree_vm_abi_ririi_t
    data::NTuple{44, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_ririi_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 20)
    f === :i3 && return Ptr{Int32}(x + 36)
    f === :i4 && return Ptr{Int32}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_ririi_t, f::Symbol)
    r = Ref{iree_vm_abi_ririi_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_ririi_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_ririi_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_ririi_checked_deref(buffer)
    ccall((:iree_vm_abi_ririi_checked_deref, libiree), Ptr{iree_vm_abi_ririi_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_ririi_reset(value)
    ccall((:iree_vm_abi_ririi_reset, libiree), Cvoid, (Ptr{iree_vm_abi_ririi_t},), value)
end

struct iree_vm_abi_rii_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rii_t, f::Symbol)
    r = Ref{iree_vm_abi_rii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rii_checked_deref(buffer)
    ccall((:iree_vm_abi_rii_checked_deref, libiree), Ptr{iree_vm_abi_rii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rii_reset(value)
    ccall((:iree_vm_abi_rii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rii_t},), value)
end

struct iree_vm_abi_rIi_t
    data::NTuple{28, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIi_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIi_t, f::Symbol)
    r = Ref{iree_vm_abi_rIi_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIi_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIi_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIi_checked_deref(buffer)
    ccall((:iree_vm_abi_rIi_checked_deref, libiree), Ptr{iree_vm_abi_rIi_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rIi_reset(value)
    ccall((:iree_vm_abi_rIi_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rIi_t},), value)
end

struct iree_vm_abi_rIii_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 24)
    f === :i3 && return Ptr{Int32}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIii_t, f::Symbol)
    r = Ref{iree_vm_abi_rIii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIii_checked_deref(buffer)
    ccall((:iree_vm_abi_rIii_checked_deref, libiree), Ptr{iree_vm_abi_rIii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rIii_reset(value)
    ccall((:iree_vm_abi_rIii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rIii_t},), value)
end

struct iree_vm_abi_rII_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rII_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :i2 && return Ptr{Int64}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rII_t, f::Symbol)
    r = Ref{iree_vm_abi_rII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rII_checked_deref(buffer)
    ccall((:iree_vm_abi_rII_checked_deref, libiree), Ptr{iree_vm_abi_rII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rII_reset(value)
    ccall((:iree_vm_abi_rII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rII_t},), value)
end

struct iree_vm_abi_rif_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rif_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :f2 && return Ptr{Cfloat}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rif_t, f::Symbol)
    r = Ref{iree_vm_abi_rif_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rif_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rif_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rif_checked_deref(buffer)
    ccall((:iree_vm_abi_rif_checked_deref, libiree), Ptr{iree_vm_abi_rif_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rif_reset(value)
    ccall((:iree_vm_abi_rif_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rif_t},), value)
end

struct iree_vm_abi_riii_t
    data::NTuple{28, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :i3 && return Ptr{Int32}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riii_t, f::Symbol)
    r = Ref{iree_vm_abi_riii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riii_checked_deref(buffer)
    ccall((:iree_vm_abi_riii_checked_deref, libiree), Ptr{iree_vm_abi_riii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_riii_reset(value)
    ccall((:iree_vm_abi_riii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_riii_t},), value)
end

struct iree_vm_abi_riiii_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riiii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :i3 && return Ptr{Int32}(x + 24)
    f === :i4 && return Ptr{Int32}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riiii_t, f::Symbol)
    r = Ref{iree_vm_abi_riiii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riiii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riiii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riiii_checked_deref(buffer)
    ccall((:iree_vm_abi_riiii_checked_deref, libiree), Ptr{iree_vm_abi_riiii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_riiii_reset(value)
    ccall((:iree_vm_abi_riiii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_riiii_t},), value)
end

struct iree_vm_abi_riiI_t
    data::NTuple{32, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riiI_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :i3 && return Ptr{Int64}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riiI_t, f::Symbol)
    r = Ref{iree_vm_abi_riiI_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riiI_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riiI_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riiI_checked_deref(buffer)
    ccall((:iree_vm_abi_riiI_checked_deref, libiree), Ptr{iree_vm_abi_riiI_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_riiI_reset(value)
    ccall((:iree_vm_abi_riiI_reset, libiree), Cvoid, (Ptr{iree_vm_abi_riiI_t},), value)
end

struct iree_vm_abi_iirII_t
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_iirII_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 4)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 8)
    f === :i3 && return Ptr{Int64}(x + 24)
    f === :i4 && return Ptr{Int64}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_iirII_t, f::Symbol)
    r = Ref{iree_vm_abi_iirII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_iirII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_iirII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_iirII_checked_deref(buffer)
    ccall((:iree_vm_abi_iirII_checked_deref, libiree), Ptr{iree_vm_abi_iirII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_iirII_reset(value)
    ccall((:iree_vm_abi_iirII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_iirII_t},), value)
end

struct iree_vm_abi_riirII_t
    data::NTuple{56, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riirII_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 24)
    f === :i4 && return Ptr{Int64}(x + 40)
    f === :i5 && return Ptr{Int64}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riirII_t, f::Symbol)
    r = Ref{iree_vm_abi_riirII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riirII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riirII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riirII_checked_deref(buffer)
    ccall((:iree_vm_abi_riirII_checked_deref, libiree), Ptr{iree_vm_abi_riirII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_riirII_reset(value)
    ccall((:iree_vm_abi_riirII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_riirII_t},), value)
end

struct iree_vm_abi_riiirII_t
    data::NTuple{60, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riiirII_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :i3 && return Ptr{Int32}(x + 24)
    f === :r4 && return Ptr{iree_vm_ref_t}(x + 28)
    f === :i5 && return Ptr{Int64}(x + 44)
    f === :i6 && return Ptr{Int64}(x + 52)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riiirII_t, f::Symbol)
    r = Ref{iree_vm_abi_riiirII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riiirII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riiirII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riiirII_checked_deref(buffer)
    ccall((:iree_vm_abi_riiirII_checked_deref, libiree), Ptr{iree_vm_abi_riiirII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_riiirII_reset(value)
    ccall((:iree_vm_abi_riiirII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_riiirII_t},), value)
end

struct iree_vm_abi_rriirIIrIII_t
    data::NTuple{112, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rriirIIrIII_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :i3 && return Ptr{Int32}(x + 36)
    f === :r4 && return Ptr{iree_vm_ref_t}(x + 40)
    f === :i5 && return Ptr{Int64}(x + 56)
    f === :i6 && return Ptr{Int64}(x + 64)
    f === :r7 && return Ptr{iree_vm_ref_t}(x + 72)
    f === :i8 && return Ptr{Int64}(x + 88)
    f === :i9 && return Ptr{Int64}(x + 96)
    f === :i10 && return Ptr{Int64}(x + 104)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rriirIIrIII_t, f::Symbol)
    r = Ref{iree_vm_abi_rriirIIrIII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rriirIIrIII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rriirIIrIII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rriirIIrIII_checked_deref(buffer)
    ccall((:iree_vm_abi_rriirIIrIII_checked_deref, libiree), Ptr{iree_vm_abi_rriirIIrIII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rriirIIrIII_reset(value)
    ccall((:iree_vm_abi_rriirIIrIII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rriirIIrIII_t},), value)
end

struct iree_vm_abi_rriiii_t
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rriiii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :i3 && return Ptr{Int32}(x + 36)
    f === :i4 && return Ptr{Int32}(x + 40)
    f === :i5 && return Ptr{Int32}(x + 44)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rriiii_t, f::Symbol)
    r = Ref{iree_vm_abi_rriiii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rriiii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rriiii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rriiii_checked_deref(buffer)
    ccall((:iree_vm_abi_rriiii_checked_deref, libiree), Ptr{iree_vm_abi_rriiii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rriiii_reset(value)
    ccall((:iree_vm_abi_rriiii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rriiii_t},), value)
end

struct iree_vm_abi_rrIIii_t
    data::NTuple{56, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrIIii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int64}(x + 32)
    f === :i3 && return Ptr{Int64}(x + 40)
    f === :i4 && return Ptr{Int32}(x + 48)
    f === :i5 && return Ptr{Int32}(x + 52)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrIIii_t, f::Symbol)
    r = Ref{iree_vm_abi_rrIIii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrIIii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrIIii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrIIii_checked_deref(buffer)
    ccall((:iree_vm_abi_rrIIii_checked_deref, libiree), Ptr{iree_vm_abi_rrIIii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrIIii_reset(value)
    ccall((:iree_vm_abi_rrIIii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrIIii_t},), value)
end

struct iree_vm_abi_rrirI_t
    data::NTuple{60, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrirI_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 36)
    f === :i4 && return Ptr{Int64}(x + 52)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrirI_t, f::Symbol)
    r = Ref{iree_vm_abi_rrirI_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrirI_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrirI_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrirI_checked_deref(buffer)
    ccall((:iree_vm_abi_rrirI_checked_deref, libiree), Ptr{iree_vm_abi_rrirI_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrirI_reset(value)
    ccall((:iree_vm_abi_rrirI_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrirI_t},), value)
end

struct iree_vm_abi_rrIrII_t
    data::NTuple{72, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrIrII_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int64}(x + 32)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 40)
    f === :i4 && return Ptr{Int64}(x + 56)
    f === :i5 && return Ptr{Int64}(x + 64)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrIrII_t, f::Symbol)
    r = Ref{iree_vm_abi_rrIrII_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrIrII_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrIrII_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrIrII_checked_deref(buffer)
    ccall((:iree_vm_abi_rrIrII_checked_deref, libiree), Ptr{iree_vm_abi_rrIrII_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrIrII_reset(value)
    ccall((:iree_vm_abi_rrIrII_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrIrII_t},), value)
end

struct iree_vm_abi_rrIii_t
    data::NTuple{48, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrIii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int64}(x + 32)
    f === :i3 && return Ptr{Int32}(x + 40)
    f === :i4 && return Ptr{Int32}(x + 44)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrIii_t, f::Symbol)
    r = Ref{iree_vm_abi_rrIii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrIii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrIii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrIii_checked_deref(buffer)
    ccall((:iree_vm_abi_rrIii_checked_deref, libiree), Ptr{iree_vm_abi_rrIii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrIii_reset(value)
    ccall((:iree_vm_abi_rrIii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrIii_t},), value)
end

struct iree_vm_abi_rrrIii_t
    data::NTuple{64, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrrIii_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 32)
    f === :i3 && return Ptr{Int64}(x + 48)
    f === :i4 && return Ptr{Int32}(x + 56)
    f === :i5 && return Ptr{Int32}(x + 60)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrrIii_t, f::Symbol)
    r = Ref{iree_vm_abi_rrrIii_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrrIii_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrrIii_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrrIii_checked_deref(buffer)
    ccall((:iree_vm_abi_rrrIii_checked_deref, libiree), Ptr{iree_vm_abi_rrrIii_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rrrIii_reset(value)
    ccall((:iree_vm_abi_rrrIii_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rrrIii_t},), value)
end

struct iree_vm_abi_rIrriiiI_t
    data::NTuple{76, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIrriiiI_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 24)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 40)
    f === :i4 && return Ptr{Int32}(x + 56)
    f === :i5 && return Ptr{Int32}(x + 60)
    f === :i6 && return Ptr{Int32}(x + 64)
    f === :i7 && return Ptr{Int64}(x + 68)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIrriiiI_t, f::Symbol)
    r = Ref{iree_vm_abi_rIrriiiI_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIrriiiI_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIrriiiI_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIrriiiI_checked_deref(buffer)
    ccall((:iree_vm_abi_rIrriiiI_checked_deref, libiree), Ptr{iree_vm_abi_rIrriiiI_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rIrriiiI_reset(value)
    ccall((:iree_vm_abi_rIrriiiI_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rIrriiiI_t},), value)
end

struct iree_vm_abi_rIrrr_t
    data::NTuple{72, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIrrr_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 24)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 40)
    f === :r4 && return Ptr{iree_vm_ref_t}(x + 56)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIrrr_t, f::Symbol)
    r = Ref{iree_vm_abi_rIrrr_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIrrr_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIrrr_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIrrr_checked_deref(buffer)
    ccall((:iree_vm_abi_rIrrr_checked_deref, libiree), Ptr{iree_vm_abi_rIrrr_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_abi_rIrrr_reset(value)
    ccall((:iree_vm_abi_rIrrr_reset, libiree), Cvoid, (Ptr{iree_vm_abi_rIrrr_t},), value)
end

struct iree_vm_abi_rIrrCrD_t
    data::NTuple{60, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIrrCrD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 24)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 40)
    f === :a4_count && return Ptr{iree_vm_size_t}(x + 56)
    f === :a4 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 60)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIrrCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_rIrrCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIrrCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIrrCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIrrCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_rIrrCrD_checked_deref, libiree), Ptr{iree_vm_abi_rIrrCrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rCiD_t
    data::NTuple{20, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rCiD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :a1_count && return Ptr{iree_vm_size_t}(x + 16)
    f === :a1 && return Ptr{NTuple{0, iree_vm_abi_i_t}}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rCiD_t, f::Symbol)
    r = Ref{iree_vm_abi_rCiD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rCiD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rCiD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rCiD_checked_deref(buffer)
    ccall((:iree_vm_abi_rCiD_checked_deref, libiree), Ptr{iree_vm_abi_rCiD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rCrD_t
    data::NTuple{20, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rCrD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :a1_count && return Ptr{iree_vm_size_t}(x + 16)
    f === :a1 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_rCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_rCrD_checked_deref, libiree), Ptr{iree_vm_abi_rCrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_riCiD_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riCiD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :a2_count && return Ptr{iree_vm_size_t}(x + 20)
    f === :a2 && return Ptr{NTuple{0, iree_vm_abi_i_t}}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riCiD_t, f::Symbol)
    r = Ref{iree_vm_abi_riCiD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riCiD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riCiD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riCiD_checked_deref(buffer)
    ccall((:iree_vm_abi_riCiD_checked_deref, libiree), Ptr{iree_vm_abi_riCiD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rIIiiCID_t
    data::NTuple{44, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rIIiiCID_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int64}(x + 16)
    f === :i2 && return Ptr{Int64}(x + 24)
    f === :i3 && return Ptr{Int32}(x + 32)
    f === :i4 && return Ptr{Int32}(x + 36)
    f === :a5_count && return Ptr{iree_vm_size_t}(x + 40)
    f === :a5 && return Ptr{NTuple{0, iree_vm_abi_I_t}}(x + 44)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rIIiiCID_t, f::Symbol)
    r = Ref{iree_vm_abi_rIIiiCID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rIIiiCID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rIIiiCID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rIIiiCID_checked_deref(buffer)
    ccall((:iree_vm_abi_rIIiiCID_checked_deref, libiree), Ptr{iree_vm_abi_rIIiiCID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rriiCID_t
    data::NTuple{44, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rriiCID_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :i3 && return Ptr{Int32}(x + 36)
    f === :a4_count && return Ptr{iree_vm_size_t}(x + 40)
    f === :a4 && return Ptr{NTuple{0, iree_vm_abi_I_t}}(x + 44)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rriiCID_t, f::Symbol)
    r = Ref{iree_vm_abi_rriiCID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rriiCID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rriiCID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rriiCID_checked_deref(buffer)
    ccall((:iree_vm_abi_rriiCID_checked_deref, libiree), Ptr{iree_vm_abi_rriiCID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_riCrD_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riCrD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :a2_count && return Ptr{iree_vm_size_t}(x + 20)
    f === :a2 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_riCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_riCrD_checked_deref, libiree), Ptr{iree_vm_abi_riCrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_riiCriD_t
    data::NTuple{28, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riiCriD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 20)
    f === :a3_count && return Ptr{iree_vm_size_t}(x + 24)
    f === :a3 && return Ptr{NTuple{0, iree_vm_abi_ri_t}}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riiCriD_t, f::Symbol)
    r = Ref{iree_vm_abi_riiCriD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riiCriD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riiCriD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riiCriD_checked_deref(buffer)
    ccall((:iree_vm_abi_riiCriD_checked_deref, libiree), Ptr{iree_vm_abi_riiCriD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rirCrD_t
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rirCrD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 20)
    f === :a3_count && return Ptr{iree_vm_size_t}(x + 36)
    f === :a3 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rirCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_rirCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rirCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rirCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rirCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_rirCrD_checked_deref, libiree), Ptr{iree_vm_abi_rirCrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rrrrCrD_t
    data::NTuple{68, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrrrCrD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :r2 && return Ptr{iree_vm_ref_t}(x + 32)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 48)
    f === :a4_count && return Ptr{iree_vm_size_t}(x + 64)
    f === :a4 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 68)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrrrCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_rrrrCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrrrCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrrrCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrrrCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_rrrrCrD_checked_deref, libiree), Ptr{iree_vm_abi_rrrrCrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rriCiD_t
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rriCiD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :a3_count && return Ptr{iree_vm_size_t}(x + 36)
    f === :a3 && return Ptr{NTuple{0, iree_vm_abi_i_t}}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rriCiD_t, f::Symbol)
    r = Ref{iree_vm_abi_rriCiD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rriCiD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rriCiD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rriCiD_checked_deref(buffer)
    ccall((:iree_vm_abi_rriCiD_checked_deref, libiree), Ptr{iree_vm_abi_rriCiD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rrirCID_t
    data::NTuple{56, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrirCID_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :r3 && return Ptr{iree_vm_ref_t}(x + 36)
    f === :a4_count && return Ptr{iree_vm_size_t}(x + 52)
    f === :a4 && return Ptr{NTuple{0, iree_vm_abi_I_t}}(x + 56)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrirCID_t, f::Symbol)
    r = Ref{iree_vm_abi_rrirCID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrirCID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrirCID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrirCID_checked_deref(buffer)
    ccall((:iree_vm_abi_rrirCID_checked_deref, libiree), Ptr{iree_vm_abi_rrirCID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_riCiiiD_t
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_riCiiiD_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :i1 && return Ptr{Int32}(x + 16)
    f === :a2_count && return Ptr{iree_vm_size_t}(x + 20)
    f === :a2 && return Ptr{NTuple{0, iree_vm_abi_iii_t}}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_riCiiiD_t, f::Symbol)
    r = Ref{iree_vm_abi_riCiiiD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_riCiiiD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_riCiiiD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_riCiiiD_checked_deref(buffer)
    ccall((:iree_vm_abi_riCiiiD_checked_deref, libiree), Ptr{iree_vm_abi_riCiiiD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rrCrIID_t
    data::NTuple{36, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rrCrIID_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :a2_count && return Ptr{iree_vm_size_t}(x + 32)
    f === :a2 && return Ptr{NTuple{0, iree_vm_abi_rII_t}}(x + 36)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rrCrIID_t, f::Symbol)
    r = Ref{iree_vm_abi_rrCrIID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rrCrIID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rrCrIID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rrCrIID_checked_deref(buffer)
    ccall((:iree_vm_abi_rrCrIID_checked_deref, libiree), Ptr{iree_vm_abi_rrCrIID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_rriCiirIID_t
    data::NTuple{40, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_rriCiirIID_t}, f::Symbol)
    f === :r0 && return Ptr{iree_vm_ref_t}(x + 0)
    f === :r1 && return Ptr{iree_vm_ref_t}(x + 16)
    f === :i2 && return Ptr{Int32}(x + 32)
    f === :a3_count && return Ptr{iree_vm_size_t}(x + 36)
    f === :a3 && return Ptr{NTuple{0, iree_vm_abi_iirII_t}}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_rriCiirIID_t, f::Symbol)
    r = Ref{iree_vm_abi_rriCiirIID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_rriCiirIID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_rriCiirIID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_rriCiirIID_checked_deref(buffer)
    ccall((:iree_vm_abi_rriCiirIID_checked_deref, libiree), Ptr{iree_vm_abi_rriCiirIID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_CrD_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_CrD_t}, f::Symbol)
    f === :a0_count && return Ptr{iree_vm_size_t}(x + 0)
    f === :a0 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_CrD_t, f::Symbol)
    r = Ref{iree_vm_abi_CrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_CrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_CrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_CrD_checked_deref(buffer)
    ccall((:iree_vm_abi_CrD_checked_deref, libiree), Ptr{iree_vm_abi_CrD_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_CrID_t
    data::NTuple{4, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_CrID_t}, f::Symbol)
    f === :a0_count && return Ptr{iree_vm_size_t}(x + 0)
    f === :a0 && return Ptr{NTuple{0, iree_vm_abi_rI_t}}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_CrID_t, f::Symbol)
    r = Ref{iree_vm_abi_CrID_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_CrID_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_CrID_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_CrID_checked_deref(buffer)
    ccall((:iree_vm_abi_CrID_checked_deref, libiree), Ptr{iree_vm_abi_CrID_t}, (iree_byte_span_t,), buffer)
end

struct iree_vm_abi_iCrD_t
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{iree_vm_abi_iCrD_t}, f::Symbol)
    f === :i0 && return Ptr{Int32}(x + 0)
    f === :a1_count && return Ptr{iree_vm_size_t}(x + 4)
    f === :a1 && return Ptr{NTuple{0, iree_vm_abi_r_t}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::iree_vm_abi_iCrD_t, f::Symbol)
    r = Ref{iree_vm_abi_iCrD_t}(x)
    ptr = Base.unsafe_convert(Ptr{iree_vm_abi_iCrD_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{iree_vm_abi_iCrD_t}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function iree_vm_abi_iCrD_checked_deref(buffer)
    ccall((:iree_vm_abi_iCrD_checked_deref, libiree), Ptr{iree_vm_abi_iCrD_t}, (iree_byte_span_t,), buffer)
end

function iree_vm_shim_irIi_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_irIi_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_I(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_I, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_ii(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_ii, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_iI(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_iI, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_iii(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_iii, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_iiii(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_iiii, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_rI(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_rI, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_r_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_r_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rCiD_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rCiD_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rCrD_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rCrD_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_ii(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_ii, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_I(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_I, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_f(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_f, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ri_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ri_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rI_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rI_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rI_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rI_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rI_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rI_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riCiD_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riCiD_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIIiiCID_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIIiiCID_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riCiiiD_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riCiiiD_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riCrD_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riCrD_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIi_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIi_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIii_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIii_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rii_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rii_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rII_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rII_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rif_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rif_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riii_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riii_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riiI_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riiI_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riirII_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riirII_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_riiirII_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_riiirII_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rriirIIrIII_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rriirIIrIII_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrrrCrD_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrrrCrD_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_ririi_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_ririi_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rr_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rr_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rr_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rr_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rr_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rr_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rr_ii(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rr_ii, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rr_iI(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rr_iI, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrr_iI(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrr_iI, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrr_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrr_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrCrIID_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrCrIID_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rriCiD_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rriCiD_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rriiCID_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rriiCID_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rriCiirIID_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rriCiirIID_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rriiii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rriiii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrIIii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrIIii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrirCID_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrirCID_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrirI_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrirI_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrIrII_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrIrII_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrIii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrIii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rrrIii_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rrrIii_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIrriiiI_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIrriiiI_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIrrr_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIrrr_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_rIrrCrD_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_rIrrCrD_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_CrID_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_CrID_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_CrD_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_CrD_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_iCrD_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_iCrD_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_iI_rr(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_iI_rr, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_irII_rr(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_irII_rr, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_v_i(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_v_i, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_v_r(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_v_r, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

function iree_vm_shim_v_v(stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
    ccall((:iree_vm_shim_v_v, libiree), iree_status_t, (Ptr{iree_vm_stack_t}, iree_vm_native_function_flags_t, iree_byte_span_t, iree_byte_span_t, iree_vm_native_function_target2_t, Ptr{Cvoid}, Ptr{Cvoid}), stack, flags, args_storage, rets_storage, target_fn, _module, module_state)
end

mutable struct iree_runtime_session_t end

@cenum iree_runtime_call_flag_bits_t::UInt32 begin
    IREE_RUNTIME_CALL_FLAG_RESERVED = 0
end

const iree_runtime_call_flags_t = UInt32

mutable struct iree_runtime_instance_t end

mutable struct iree_runtime_instance_options_t
    driver_registry::Ptr{iree_hal_driver_registry_t}
end

function iree_runtime_instance_options_initialize(out_options)
    ccall((:iree_runtime_instance_options_initialize, libiree), Cvoid, (Ptr{iree_runtime_instance_options_t},), out_options)
end

function iree_runtime_instance_options_use_all_available_drivers(options)
    ccall((:iree_runtime_instance_options_use_all_available_drivers, libiree), Cvoid, (Ptr{iree_runtime_instance_options_t},), options)
end

function iree_runtime_instance_create(options, host_allocator, out_instance)
    ccall((:iree_runtime_instance_create, libiree), iree_status_t, (Ptr{iree_runtime_instance_options_t}, iree_allocator_t, Ptr{Ptr{iree_runtime_instance_t}}), options, host_allocator, out_instance)
end

function iree_runtime_instance_retain(instance)
    ccall((:iree_runtime_instance_retain, libiree), Cvoid, (Ptr{iree_runtime_instance_t},), instance)
end

function iree_runtime_instance_release(instance)
    ccall((:iree_runtime_instance_release, libiree), Cvoid, (Ptr{iree_runtime_instance_t},), instance)
end

function iree_runtime_instance_host_allocator(instance)
    ccall((:iree_runtime_instance_host_allocator, libiree), iree_allocator_t, (Ptr{iree_runtime_instance_t},), instance)
end

function iree_runtime_instance_vm_instance(instance)
    ccall((:iree_runtime_instance_vm_instance, libiree), Ptr{iree_vm_instance_t}, (Ptr{iree_runtime_instance_t},), instance)
end

function iree_runtime_instance_driver_registry(instance)
    ccall((:iree_runtime_instance_driver_registry, libiree), Ptr{iree_hal_driver_registry_t}, (Ptr{iree_runtime_instance_t},), instance)
end

function iree_runtime_instance_try_create_default_device(instance, driver_name, out_device)
    ccall((:iree_runtime_instance_try_create_default_device, libiree), iree_status_t, (Ptr{iree_runtime_instance_t}, iree_string_view_t, Ptr{Ptr{iree_hal_device_t}}), instance, driver_name, out_device)
end

@cenum iree_runtime_session_builtins_bits_t::UInt64 begin
    IREE_RUNTIME_SESSION_BUILTIN_ALL = 0xffffffffffffffff
end

const iree_runtime_session_builtins_t = UInt64

mutable struct iree_runtime_session_options_t
    context_flags::iree_vm_context_flags_t
    builtin_modules::iree_runtime_session_builtins_t
end

function iree_runtime_session_options_initialize(out_options)
    ccall((:iree_runtime_session_options_initialize, libiree), Cvoid, (Ptr{iree_runtime_session_options_t},), out_options)
end

function iree_runtime_session_create_with_device(instance, options, device, host_allocator, out_session)
    ccall((:iree_runtime_session_create_with_device, libiree), iree_status_t, (Ptr{iree_runtime_instance_t}, Ptr{iree_runtime_session_options_t}, Ptr{iree_hal_device_t}, iree_allocator_t, Ptr{Ptr{iree_runtime_session_t}}), instance, options, device, host_allocator, out_session)
end

function iree_runtime_session_retain(session)
    ccall((:iree_runtime_session_retain, libiree), Cvoid, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_release(session)
    ccall((:iree_runtime_session_release, libiree), Cvoid, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_host_allocator(session)
    ccall((:iree_runtime_session_host_allocator, libiree), iree_allocator_t, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_instance(session)
    ccall((:iree_runtime_session_instance, libiree), Ptr{iree_runtime_instance_t}, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_context(session)
    ccall((:iree_runtime_session_context, libiree), Ptr{iree_vm_context_t}, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_device(session)
    ccall((:iree_runtime_session_device, libiree), Ptr{iree_hal_device_t}, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_device_allocator(session)
    ccall((:iree_runtime_session_device_allocator, libiree), Ptr{iree_hal_allocator_t}, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_trim(session)
    ccall((:iree_runtime_session_trim, libiree), iree_status_t, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_append_bytecode_module_from_memory(session, flatbuffer_data, flatbuffer_allocator)
    ccall((:iree_runtime_session_append_bytecode_module_from_memory, libiree), iree_status_t, (Ptr{iree_runtime_session_t}, iree_const_byte_span_t, iree_allocator_t), session, flatbuffer_data, flatbuffer_allocator)
end

function iree_runtime_session_append_bytecode_module_from_file(session, file_path)
    ccall((:iree_runtime_session_append_bytecode_module_from_file, libiree), iree_status_t, (Ptr{iree_runtime_session_t}, Cstring), session, file_path)
end

function iree_runtime_session_append_bytecode_module_from_stdin(session)
    ccall((:iree_runtime_session_append_bytecode_module_from_stdin, libiree), iree_status_t, (Ptr{iree_runtime_session_t},), session)
end

function iree_runtime_session_call_by_name(session, full_name, input_list, output_list)
    ccall((:iree_runtime_session_call_by_name, libiree), iree_status_t, (Ptr{iree_runtime_session_t}, iree_string_view_t, Ptr{iree_vm_list_t}, Ptr{iree_vm_list_t}), session, full_name, input_list, output_list)
end

mutable struct __JL_Ctag_50
    fd::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_50}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_50, f::Symbol)
    r = Ref{__JL_Ctag_50}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_50}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_50}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_51
    data::NTuple{8, UInt8}
end

function Base.getproperty(x::Ptr{__JL_Ctag_51}, f::Symbol)
    f === :read_fd && return Ptr{Cint}(x + 0)
    f === :write_fd && return Ptr{Cint}(x + 4)
    f === :fds && return Ptr{NTuple{2, Cint}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_51, f::Symbol)
    r = Ref{__JL_Ctag_51}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_51}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_51}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

mutable struct __JL_Ctag_55
    ptr::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{__JL_Ctag_55}, f::Symbol)
    f === :ptr && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_55, f::Symbol)
    r = Ref{__JL_Ctag_55}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_55}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_55}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


mutable struct __JL_Ctag_56
    fd::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_56}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_56, f::Symbol)
    r = Ref{__JL_Ctag_56}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_56}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_56}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


mutable struct __JL_Ctag_57
    handle::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{__JL_Ctag_57}, f::Symbol)
    f === :handle && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_57, f::Symbol)
    r = Ref{__JL_Ctag_57}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_57}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_57}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


const IREE_ARCH = "x86_64"

const IREE_ARCH_X86_64 = 1

const IREE_ENDIANNESS_LITTLE = 1

const IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED = 0

const IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_8 = IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED

const IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_16 = IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED

const IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_32 = IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED

const IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED_64 = IREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED

const IREE_COMPILER_CLANG = 1

const IREE_COMPILER_GCC_COMPAT = 1

const IREE_COMPILER_HAS_BUILTIN_DEBUG_TRAP = 1

const IREE_PLATFORM_LINUX = 1

const IREE_HOST_SIZE_T = Csize_t

const PRIhsz = "zu"

# Skipping MacroDefinition: IREE_HOST_SIZE_MAX ( sizeof ( iree_host_size_t ) == 4 ? UINT32_MAX : UINT64_MAX )

const IREE_DEVICE_SIZE_T = Csize_t

const PRIdsz = "zu"

# Skipping MacroDefinition: IREE_DEVICE_SIZE_MAX ( sizeof ( iree_device_size_t ) == 4 ? UINT32_MAX : UINT64_MAX )

const IREE_STATUS_MODE = 3

const IREE_SYNCHRONIZATION_DISABLE_UNSAFE = 0

const IREE_FILE_IO_ENABLE = 1

const IREE_STATISTICS_ENABLE = 1

const IREE_HAL_HEAP_BUFFER_ALIGNMENT = 64

const IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE = 1

const IREE_HAL_MODULE_STRING_UTIL_ENABLE = 1

const IREE_VM_BACKTRACE_ENABLE = 1

const IREE_VM_EXECUTION_TRACING_ENABLE = 1

const IREE_VM_EXECUTION_TRACING_FORCE_ENABLE = 0

const IREE_VM_EXECUTION_TRACING_SRC_LOC_ENABLE = 0

const IREE_VM_BYTECODE_DISPATCH_COMPUTED_GOTO_ENABLE = 0

const IREE_VM_EXT_F32_ENABLE = 1

const IREE_VM_EXT_F64_ENABLE = 0

const IREE_VM_UBSAN_CHECKABLE_ENABLE = 0

const IREE_PTR_SIZE = 8

# Skipping MacroDefinition: iree_max_align_t sizeof ( long double )

# Skipping MacroDefinition: IREE_ATTRIBUTE_NORETURN __attribute__ ( ( noreturn ) )

# Skipping MacroDefinition: IREE_MUST_USE_RESULT __attribute__ ( ( warn_unused_result ) )

# const IREE_RESTRICT = restrict

# Skipping MacroDefinition: IREE_ATTRIBUTE_ALWAYS_INLINE __attribute__ ( ( always_inline ) )

# Skipping MacroDefinition: IREE_ATTRIBUTE_NOINLINE __attribute__ ( ( noinline ) )

# Skipping MacroDefinition: IREE_ATTRIBUTE_HOT __attribute__ ( ( hot ) )

# Skipping MacroDefinition: IREE_ATTRIBUTE_COLD __attribute__ ( ( cold ) )

# Skipping MacroDefinition: IREE_ATTRIBUTE_PACKED __attribute__ ( ( __packed__ ) )

# Skipping MacroDefinition: IREE_ATTRIBUTE_UNUSED __attribute__ ( ( unused ) )

# const IREE_STRING_VIEW_NPOS = SIZE_MAX

const IREE_STATUS_FEATURE_SOURCE_LOCATION = 1 << 0

const IREE_STATUS_FEATURE_ANNOTATIONS = 1 << 1

const IREE_STATUS_FEATURE_STACK_TRACE = 1 << 2

const IREE_STATUS_FEATURES = (IREE_STATUS_FEATURE_SOURCE_LOCATION | IREE_STATUS_FEATURE_ANNOTATIONS) | IREE_STATUS_FEATURE_STACK_TRACE

# const iree_make_status = IREE_STATUS_IMPL_MAKE_

# const iree_make_status_with_location = IREE_STATUS_IMPL_MAKE_LOC_

const IREE_TIME_INFINITE_PAST = typemin(Int64)

const IREE_TIME_INFINITE_FUTURE = typemax(Int64)

const IREE_DURATION_ZERO = 0

const IREE_DURATION_INFINITE = typemax(Int64)

const IREE_HAVE_WAIT_TYPE_EVENTFD = 1

const IREE_HAVE_WAIT_TYPE_PIPE = 1

const IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX = 1

const IREE_LOOP_INLINE_STORAGE_SIZE = 512

const iree_hardware_destructive_interference_size = 64

const iree_hardware_constructive_interference_size = 64

# const iree_atomic_load_int32 = iree_atomic_load_auto

# const iree_atomic_store_int32 = iree_atomic_store_auto

# const iree_atomic_fetch_add_int32 = iree_atomic_fetch_add_auto

# const iree_atomic_fetch_sub_int32 = iree_atomic_fetch_sub_auto
# 
# const iree_atomic_fetch_and_int32 = iree_atomic_fetch_and_auto
# 
# const iree_atomic_fetch_or_int32 = iree_atomic_fetch_or_auto
# 
# const iree_atomic_fetch_xor_int32 = iree_atomic_fetch_xor_auto
# 
# const iree_atomic_exchange_int32 = iree_atomic_exchange_auto
# 
# const iree_atomic_compare_exchange_strong_int32 = iree_atomic_compare_exchange_strong_auto
# 
# const iree_atomic_compare_exchange_weak_int32 = iree_atomic_compare_exchange_weak_auto
# 
# const iree_atomic_load_int64 = iree_atomic_load_auto
# 
# const iree_atomic_store_int64 = iree_atomic_store_auto
# 
# const iree_atomic_fetch_add_int64 = iree_atomic_fetch_add_auto
# 
# const iree_atomic_fetch_sub_int64 = iree_atomic_fetch_sub_auto
# 
# const iree_atomic_fetch_and_int64 = iree_atomic_fetch_and_auto
# 
# const iree_atomic_fetch_or_int64 = iree_atomic_fetch_or_auto
# 
# const iree_atomic_fetch_xor_int64 = iree_atomic_fetch_xor_auto
# 
# const iree_atomic_exchange_int64 = iree_atomic_exchange_auto
# 
# const iree_atomic_compare_exchange_strong_int64 = iree_atomic_compare_exchange_strong_auto
# 
# const iree_atomic_compare_exchange_weak_int64 = iree_atomic_compare_exchange_weak_auto
# 
# const iree_atomic_load_intptr = iree_atomic_load_auto
# 
# const iree_atomic_store_intptr = iree_atomic_store_auto
# 
# const iree_atomic_fetch_add_intptr = iree_atomic_fetch_add_auto
# 
# const iree_atomic_fetch_sub_intptr = iree_atomic_fetch_sub_auto
# 
# const iree_atomic_exchange_intptr = iree_atomic_exchange_auto
# 
# const iree_atomic_compare_exchange_strong_intptr = iree_atomic_compare_exchange_strong_auto
# 
# const iree_atomic_compare_exchange_weak_intptr = iree_atomic_compare_exchange_weak_auto

const IREE_WHOLE_BUFFER = iree_device_size_t(-1)

const IREE_HAL_QUEUE_AFFINITY_ANY = iree_hal_queue_affinity_t(-1)

const PRIdim = PRIdsz

const IREE_HAL_CHANNEL_RANK_DEFAULT = iree_hal_queue_affinity_t(-1)

const IREE_HAL_CHANNEL_COUNT_DEFAULT = iree_hal_queue_affinity_t(-1)

const IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE = iree_device_size_t(64 * 1024)

const IREE_HAL_DEVICE_ID_DEFAULT = Culonglong(0)

const IREE_VM_CCONV_TYPE_VOID = Cchar('v')

const IREE_VM_CCONV_TYPE_I32 = Cchar('i')

const IREE_VM_CCONV_TYPE_I64 = Cchar('I')

const IREE_VM_CCONV_TYPE_F32 = Cchar('f')

const IREE_VM_CCONV_TYPE_F64 = Cchar('F')

const IREE_VM_CCONV_TYPE_REF = Cchar('r')

const IREE_VM_CCONV_TYPE_SPAN_START = Cchar('C')

const IREE_VM_CCONV_TYPE_SPAN_END = Cchar('D')

const IREE_TRACING_FEATURE_INSTRUMENTATION = 1 << 0

const IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS = 1 << 1

const IREE_TRACING_FEATURE_ALLOCATION_TRACKING = 1 << 2

const IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS = 1 << 3

const IREE_TRACING_FEATURE_FAST_LOCKS = 1 << 4

const IREE_TRACING_FEATURE_SLOW_LOCKS = 1 << 5

const IREE_TRACING_FEATURE_LOG_MESSAGES = 1 << 6

const IREE_TRACING_FEATURE_FIBERS = 1 << 7

const IREE_TRACING_MAX_CALLSTACK_DEPTH = 16

const IREE_TRACING_FEATURES = 0

# const TRACY_DBGHELP_LOCK = IREEDbgHelp

const TRACY_NO_FRAME_IMAGE = 1

const TRACY_NO_VSYNC_CAPTURE = 1

const IREE_VM_STACK_DEFAULT_SIZE = 8 * 1024

const IREE_VM_STACK_MIN_SIZE = 1 * 1024

const IREE_VM_STACK_MAX_SIZE = 1 * 1024 * 1024

const IREE_VM_VALUE_STORAGE_SIZE = 8

end # module
