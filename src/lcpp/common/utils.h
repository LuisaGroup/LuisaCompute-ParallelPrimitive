/*
 * @Author: Ligo 
 * @Date: 2025-10-22 17:17:43 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 14:53:46
 */


#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/shader_cache.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{

static inline float to_radius(float degree)
{
    return degree * 0.0174532925f;
}
static inline int imax(int a, int b)
{
    return a > b ? a : b;
}
static constexpr inline bool is_power_of_two(int x)
{
    return (x & (x - 1)) == 0;
}
static inline float radians(float degree)
{
    return degree * 0.017453292519943295769236907684886f;
}
static inline int floor_pow_2(int n)
{
#ifdef WIN32
    return 1 << (int)logb((float)n);
#else
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

template <typename T>
constexpr inline T ceil_div(T a, T b) noexcept
{
    return a / b + static_cast<T>(a % b != 0);
}

namespace details
{
    // Batching must not change a callable's thread identity or collective context.
    inline bool can_batch_callable(luisa::compute::Function function) noexcept
    {
        using luisa::compute::CallOp;
        if(!function.builtin_variables().empty() || !function.shared_variables().empty()
           || !function.external_callables().empty() || function.requires_atomic()
           || function.use_cooperative_operations())
        {
            return false;
        }
        for(auto operation : function.propagated_builtin_callables())
        {
            if(operation == CallOp::SYNCHRONIZE_BLOCK
               || (operation >= CallOp::WARP_IS_FIRST_ACTIVE_LANE && operation <= CallOp::WARP_READ_FIRST_ACTIVE_LANE))
            {
                return false;
            }
        }
        return true;
    }
}  // namespace details

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
    /// Static logarithm value
    static constexpr int VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    static constexpr int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};


// Operators must be identified by their generated code: the same Callable or
// functor type can produce different functions for different captured values.
template <typename Type, typename BinaryOp>
uint64_t get_binary_op_hash(BinaryOp op)
{
    luisa::compute::Callable callable = [op](luisa::compute::Var<Type> a, luisa::compute::Var<Type> b) noexcept
    { return op(a, b); };
    return callable.function().hash();
}

template <typename Type, typename UnaryOp>
uint64_t get_unary_op_hash(UnaryOp op)
{
    luisa::compute::Callable callable = [op](luisa::compute::Var<Type> value) noexcept
    { return op(value); };
    return callable.function().hash();
}

template <NumericT Type4Byte, typename ReduceOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op)
{
    return detail::make_shader_cache_key(detail::type_ptr<Type4Byte>(), get_binary_op_hash<Type4Byte>(op));
}

template <KeyValuePairType KeyValueType, typename ReduceOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op)
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyValueType>(), get_binary_op_hash<KeyValueType>(op));
}

template <KeyValuePairType KeyValueType, typename ReduceOp, typename TransformOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyValueType>(),
                                         get_binary_op_hash<KeyValueType>(op),
                                         get_unary_op_hash<KeyValueType>(transform_op));
}

template <typename KeyType, typename ValueType>
detail::ShaderCacheKey get_type_and_op_desc()
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyType>(), detail::type_ptr<ValueType>());
}

template <typename KeyType, typename ValueType, typename ReduceOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op)
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyType>(),
                                         detail::type_ptr<ValueType>(),
                                         get_binary_op_hash<ValueType>(op));
}

template <NumericT Type4Byte, typename ReduceOp, typename TransformOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return detail::make_shader_cache_key(detail::type_ptr<Type4Byte>(),
                                         get_binary_op_hash<Type4Byte>(op),
                                         get_unary_op_hash<Type4Byte>(transform_op));
}


template <typename KeyType, typename ValueType, typename ReduceOp, typename TransformOp>
detail::ShaderCacheKey get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyType>(),
                                         detail::type_ptr<ValueType>(),
                                         get_binary_op_hash<ValueType>(op),
                                         get_unary_op_hash<ValueType>(transform_op));
}

// Radix-sort cache key: the four per-call maps in DeviceRadixSort are all keyed
// by (key type, value type, descending) in the original code.  KEY_ONLY is an
// additional template parameter that only changes the one-sweep shader; it was
// not encoded before (a latent collision when SortKeys<T> and SortPairs<T,T>
// share an instance).  Encoding it here is strictly safer and costs at most one
// extra identical compile for the key-independent reset/histogram/exclusive-sum
// shaders.
template <typename KeyType, typename ValueType, bool KEY_ONLY, bool IS_DESCENDING>
detail::ShaderCacheKey get_radix_sort_key()
{
    return detail::make_shader_cache_key(detail::type_ptr<KeyType>(),
                                         detail::type_ptr<ValueType>(),
                                         detail::bool_ptr(KEY_ONLY),
                                         detail::bool_ptr(IS_DESCENDING));
}

// Helper: round up byte_offset to the given alignment
inline size_t align_up_uint(size_t byte_offset, size_t alignment)
{
    return (byte_offset + alignment - 1) / alignment * alignment;
}

// Byte alignment for kernel-argument buffer subviews carved out of a temporary
// storage buffer. The Vulkan legacy-HLSL (DXC) descriptor path validates every
// dispatched buffer view with the device minStorageBufferOffsetAlignment (16 on
// NVIDIA, larger on some hardware) and additionally requires whole elements;
// 256 is a multiple of every element stride lcpp uses, so aligning every carved
// region to 256 bytes satisfies both constraints on every backend and is a
// no-op for cuda/dx. Regions are a few KB each, so the padding is negligible.
static constexpr size_t descriptor_view_alignment = 256;
static constexpr size_t descriptor_view_alignment_uints = descriptor_view_alignment / sizeof(uint);
inline size_t align_up_view(size_t byte_offset) noexcept
{
    return align_up_uint(byte_offset, descriptor_view_alignment);
}

// Temp-storage size for the recursive device reduce, walking byte offsets so
// that every recursion-level region is aligned to descriptor_view_alignment
// (the subviews are passed to kernels as buffer arguments). `elem_size` is the
// reduced element type size; the result is a count of those elements.
static void get_temp_size_scan(size_t& temp_storage_size, size_t m_block_size, size_t items_per_thread, size_t num_items, size_t elem_size = sizeof(uint))
{
    auto block_size = m_block_size;
    temp_storage_size = 0;
    size_t offset_bytes = 0;
    size_t num_elements = num_items;  // input segment size
    while(true)
    {
        // output segment size
        size_t num_blocks = std::max<size_t>(1u, ceil_div(num_elements, items_per_thread * block_size));
        if(num_blocks > 1)
        {
            offset_bytes = align_up_view(offset_bytes);
            offset_bytes += (size_t)num_blocks * elem_size;
        }
        num_elements = num_blocks;
        if(num_elements <= 1) { break; }
    }
    offset_bytes = align_up_view(offset_bytes);
    temp_storage_size = offset_bytes / elem_size + 1;
}


// Vulkan guarantees minStorageBufferOffsetAlignment is at most 256 bytes.
// Keeping internal scratch-buffer views on this boundary also remains valid
// for backends with less restrictive storage-buffer alignment requirements.
inline constexpr size_t TEMP_STORAGE_VIEW_ALIGNMENT = 256u;

inline size_t align_up_temp_storage(size_t byte_offset)
{
    return align_up_uint(byte_offset, TEMP_STORAGE_VIEW_ALIGNMENT);
}

template <typename T>
inline size_t aligned_temp_storage_count(size_t element_count)
{
    static_assert(TEMP_STORAGE_VIEW_ALIGNMENT % sizeof(T) == 0u,
                  "Temporary-storage element size must divide the portable view alignment.");
    return align_up_temp_storage(element_count * sizeof(T)) / sizeof(T);
}

// Helper: convert byte size to uint element count (round up)
inline size_t bytes_to_uint_count(size_t byte_size)
{
    return (byte_size + sizeof(uint) - 1) / sizeof(uint);
}


static inline luisa::compute::Callable bit_log2 = [](luisa::compute::UInt x)
{ return 31 - luisa::compute::clz(x); };

template <typename T>
    requires(!KeyValuePairType<T>)
luisa::compute::Var<T> WarpReadLane(const luisa::compute::Var<T>& input, luisa::compute::UInt source_lane)
{
    if constexpr(std::is_same_v<T, double>)
    {
        auto bits = input.template as<luisa::ulong>();
        return compute::warp_read_lane(bits, source_lane).template as<double>();
    }
    else
    {
        return compute::warp_read_lane(input, source_lane);
    }
}

template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> WarpReadLane(
    const luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input, luisa::compute::UInt source_lane)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    result.key   = WarpReadLane(input.key, source_lane);
    result.value = WarpReadLane(input.value, source_lane);
    return result;
}

template <typename T>
    requires(!KeyValuePairType<T>)
luisa::compute::Var<T> Select(const luisa::compute::Var<T>& false_value,
                              const luisa::compute::Var<T>& true_value,
                              luisa::compute::Bool          predicate)
{
    return compute::select(false_value, true_value, predicate);
}

template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> Select(
    const luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& false_value,
    const luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& true_value,
    luisa::compute::Bool                                         predicate)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    result.key   = compute::select(false_value.key, true_value.key, predicate);
    result.value = compute::select(false_value.value, true_value.value, predicate);
    return result;
}

template <NumericT Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleUp(luisa::compute::Var<Type4Byte>& input,
                                         luisa::compute::UInt            curr_lane_id,
                                         luisa::compute::UInt            offset,
                                         luisa::compute::UInt            first_lane = 0u)
{
    luisa::compute::Var<Type4Byte> result = WarpReadLane(input, curr_lane_id - offset);

    $if(compute::Int(curr_lane_id - offset) < compute::Int(first_lane))
    {
        result = input;
    };
    return result;
};


template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> ShuffleUp(luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input,
                                                                luisa::compute::UInt curr_lane_id,
                                                                luisa::compute::UInt offset,
                                                                luisa::compute::UInt first_lane = 0u)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    luisa::compute::UInt                                  src_lane = curr_lane_id - offset;
    $if(src_lane >= first_lane)
    {
        result.key   = WarpReadLane(input.key, src_lane);
        result.value = WarpReadLane(input.value, src_lane);
    }
    $else
    {
        result = input;
    };
    return result;
};

template <NumericT Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleDown(luisa::compute::Var<Type4Byte>& input,
                                           luisa::compute::UInt            curr_lane_id,
                                           luisa::compute::UInt            offset,
                                           luisa::compute::UInt            last_lane = 32u)
{
    luisa::compute::UInt           src_lane = curr_lane_id + offset;
    luisa::compute::Var<Type4Byte> result   = WarpReadLane(input, src_lane);
    $if(src_lane > last_lane)
    {
        result = input;
    };
    return result;
};


template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> ShuffleDown(luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input,
                                                                  luisa::compute::UInt curr_lane_id,
                                                                  luisa::compute::UInt offset,
                                                                  luisa::compute::UInt last_lane = 32u)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    luisa::compute::UInt                                  src_lane = curr_lane_id + offset;
    result.key   = WarpReadLane(input.key, src_lane);
    result.value = WarpReadLane(input.value, src_lane);
    $if(src_lane > last_lane)
    {
        result = input;
    };
    return result;
};

template <size_t log_mem_banks = 5>
inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
{
    return i >> log_mem_banks;
}

static luisa::compute::Callable get_lane_mask_ge = []()
{ return 0xFFFFFFFFu << compute::warp_lane_id(); };

static luisa::compute::Callable get_lane_mask_le = []()
{ return 0xFFFFFFFFu >> (31u - compute::warp_lane_id()); };

template <size_t LOGIC_WARP_SIZE>
inline luisa::compute::UInt warp_mask(luisa::compute::UInt warp_id)
{
    constexpr bool is_pow_of_two = is_power_of_two(LOGIC_WARP_SIZE);
    constexpr bool is_arch_warp  = (LOGIC_WARP_SIZE == details::WARP_SIZE);

    luisa::compute::UInt member_mask = 0xFFFFFFFFu >> (details::WARP_SIZE - LOGIC_WARP_SIZE);

    if constexpr(is_pow_of_two && !is_arch_warp)
    {
        member_mask <<= warp_id * luisa::compute::UInt(LOGIC_WARP_SIZE);
    };

    return member_mask;
}


namespace details
{
    template <int LABEL_BITS, int WARP_ACTIVE_THREADS>
    struct warp_matcher_t
    {
        static compute::UInt match_any(compute::UInt label)
        {
            return warp_matcher_t<LABEL_BITS, 32>::match_any(label) & ~(~0 << WARP_ACTIVE_THREADS);
        }
    };

    template <int LABEL_BITS>
    struct warp_matcher_t<LABEL_BITS, details::WARP_SIZE>
    {
        // match.any.sync.b32 is slower when matching a few bits
        // using a ballot loop instead
        static compute::UInt match_any(compute::UInt label)
        {
            compute::UInt retval = 0xFFFFFFFFu;

            // Extract masks of common threads for each bit
            for(auto BIT = 0u; BIT < LABEL_BITS; ++BIT)
            {
                compute::UInt current_bit = 1 << BIT;
                compute::Bool bit_is_set  = (label & current_bit) != 0;

                // get warp active mask for this bit
                compute::UInt mask = compute::warp_active_bit_mask(bit_is_set).x;

                // if the bit is not set, invert the mask
                $if(!bit_is_set)
                {
                    mask = ~mask;
                };

                // Remove peers who differ
                retval = retval & mask;
            }

            return retval;
        }
    };
}  // namespace details
/**
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
template <int LABEL_BITS, int WARP_ACTIVE_THREADS = details::WARP_SIZE>
inline compute::UInt MatchAny(compute::UInt label)
{
    return details::warp_matcher_t<LABEL_BITS, WARP_ACTIVE_THREADS>::match_any(label);
}


};  // namespace luisa::parallel_primitive
