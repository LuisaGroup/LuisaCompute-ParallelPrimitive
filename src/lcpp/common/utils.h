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
luisa::string get_type_and_op_desc(ReduceOp op)
{
    return luisa::format("{}+{}",
                         luisa::compute::Type::of<Type4Byte>()->description(),
                         get_binary_op_hash<Type4Byte>(op));
}

template <KeyValuePairType KeyValueType, typename ReduceOp>
luisa::string get_type_and_op_desc(ReduceOp op)
{
    return luisa::format("{}+{}",
                         luisa::compute::Type::of<KeyValueType>()->description(),
                         get_binary_op_hash<KeyValueType>(op));
}

template <KeyValuePairType KeyValueType, typename ReduceOp, typename TransformOp>
luisa::string get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return luisa::format("{}+{}+{}",
                         luisa::compute::Type::of<KeyValueType>()->description(),
                         get_binary_op_hash<KeyValueType>(op),
                         get_unary_op_hash<KeyValueType>(transform_op));
}

template <typename KeyType, typename ValueType>
luisa::string get_type_and_op_desc()
{
    luisa::string_view key_desc   = luisa::compute::Type::of<KeyType>()->description();
    luisa::string_view value_desc = luisa::compute::Type::of<ValueType>()->description();

    return luisa::string(key_desc) + "+" + luisa::string(value_desc);
}

template <typename KeyType, typename ValueType, typename ReduceOp>
luisa::string get_type_and_op_desc(ReduceOp op)
{
    return luisa::format("{}+{}+{}",
                         luisa::compute::Type::of<KeyType>()->description(),
                         luisa::compute::Type::of<ValueType>()->description(),
                         get_binary_op_hash<ValueType>(op));
}

template <NumericT Type4Byte, typename ReduceOp, typename TransformOp>
luisa::string get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return luisa::format("{}+{}+{}",
                         luisa::compute::Type::of<Type4Byte>()->description(),
                         get_binary_op_hash<Type4Byte>(op),
                         get_unary_op_hash<Type4Byte>(transform_op));
}


template <typename KeyType, typename ValueType, typename ReduceOp, typename TransformOp>
luisa::string get_type_and_op_desc(ReduceOp op, TransformOp transform_op)
{
    return luisa::format("{}+{}+{}+{}",
                         luisa::compute::Type::of<KeyType>()->description(),
                         luisa::compute::Type::of<ValueType>()->description(),
                         get_binary_op_hash<ValueType>(op),
                         get_unary_op_hash<ValueType>(transform_op));
}

static void get_temp_size_scan(size_t& temp_storage_size, size_t m_block_size, size_t items_per_thread, size_t num_items)
{
    auto   block_size   = m_block_size;
    auto   tile_items   = items_per_thread * block_size;
    size_t num_elements = num_items;
    temp_storage_size   = 0;
    do
    {
        size_t num_blocks = std::max<size_t>(1u, ceil_div(num_elements, tile_items));
        if(num_blocks > 1)
        {
            temp_storage_size += num_blocks;
        }
        num_elements = num_blocks;
    } while(num_elements > 1);
    temp_storage_size += 1;
}

// Helper: round up byte_offset to the given alignment
inline size_t align_up_uint(size_t byte_offset, size_t alignment)
{
    return (byte_offset + alignment - 1) / alignment * alignment;
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
    return compute::warp_read_lane(input, source_lane);
}

template <NumericT KeyType, NumericT ValueType>
luisa::compute::Var<KeyValuePair<KeyType, ValueType>> WarpReadLane(
    const luisa::compute::Var<KeyValuePair<KeyType, ValueType>>& input, luisa::compute::UInt source_lane)
{
    luisa::compute::Var<KeyValuePair<KeyType, ValueType>> result;
    result.key   = compute::warp_read_lane(input.key, source_lane);
    result.value = compute::warp_read_lane(input.value, source_lane);
    return result;
}

template <NumericT Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleUp(luisa::compute::Var<Type4Byte>& input,
                                         luisa::compute::UInt            curr_lane_id,
                                         luisa::compute::UInt            offset,
                                         luisa::compute::UInt            first_lane = 0u)
{
    luisa::compute::Var<Type4Byte> result = compute::warp_read_lane(input, curr_lane_id - offset);

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
        result.key   = compute::warp_read_lane(input.key, src_lane);
        result.value = compute::warp_read_lane(input.value, src_lane);
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
    luisa::compute::Var<Type4Byte> result   = compute::warp_read_lane(input, src_lane);
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
    result.key   = compute::warp_read_lane(input.key, src_lane);
    result.value = compute::warp_read_lane(input.value, src_lane);
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
