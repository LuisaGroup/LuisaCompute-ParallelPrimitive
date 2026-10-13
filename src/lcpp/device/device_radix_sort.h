/*
 * @Author: Ligo 
 * @Date: 2025-11-12 11:08:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 15:50:13
 */


#pragma once

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/utils.h>
#include <lcpp/agent/policy.h>
#include <lcpp/device/details/radix_sort.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE,
          size_t WARP_NUMS = details::WARP_SIZE,
          size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD,
          // Keep CUB's tuned 32-bit key-only onesweep policy by default while
          // allowing callers to benchmark a single policy end to end.
          bool CUB_SMALL_KEY_POLICY = true>
class DeviceRadixSort : public LuisaModule
{
    enum class RadixSortAlgorithm
    {
        ONE_SWEEP = 0,
    };

  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_decomposed_key_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_decomposed_pair_shaders;

    template <typename KeyType, typename Decomposer>
    using DecomposedTuple =
        std::remove_cvref_t<decltype(std::declval<Decomposer&>()(std::declval<Var<KeyType>&>()))>;

    template <typename Tuple, size_t... I>
    [[nodiscard]] static consteval uint decomposed_bit_count_impl(std::index_sequence<I...>)
    {
        static_assert((NumericT<expr_value_t<std::tuple_element_t<I, Tuple>>> && ...),
                      "DeviceRadixSort decomposer components must be arithmetic DSL values.");
        return (0u + ... + static_cast<uint>(sizeof(expr_value_t<std::tuple_element_t<I, Tuple>>) * 8u));
    }

    template <typename KeyType, typename Decomposer>
    [[nodiscard]] static consteval uint decomposed_bit_count()
    {
        using Tuple = DecomposedTuple<KeyType, Decomposer>;
        static_assert(std::tuple_size_v<Tuple> > 0u, "DeviceRadixSort decomposer result must not be empty.");
        return decomposed_bit_count_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }

    template <size_t INDEX, typename Tuple>
    [[nodiscard]] static consteval uint decomposed_suffix_bits()
    {
        if constexpr(INDEX + 1u == std::tuple_size_v<Tuple>)
        {
            return 0u;
        }
        else
        {
            using NextT = expr_value_t<std::tuple_element_t<INDEX + 1u, Tuple>>;
            return static_cast<uint>(sizeof(NextT) * 8u) + decomposed_suffix_bits<INDEX + 1u, Tuple>();
        }
    }

    template <typename Value>
    [[nodiscard]] static auto decomposed_ordered_bits(const Value& value)
    {
        using ValueT           = expr_value_t<Value>;
        using UnsignedBits     = typename Traits<ValueT>::UnsignedBits;
        Var<UnsignedBits> bits = as<UnsignedBits>(value);
        bits                   = Traits<ValueT>::TwiddleIn(bits);
        return BaseDigitExtractor<ValueT>::ProcessFloatMinusZero(bits);
    }

    template <size_t INDEX, typename Tuple>
    static void compare_decomposed_component(
        const Tuple& lhs, const Tuple& rhs, uint begin_bit, uint end_bit, Bool& less, Bool& equal)
    {
        using ValueT                   = expr_value_t<std::tuple_element_t<INDEX, Tuple>>;
        using UnsignedBits             = typename Traits<ValueT>::UnsignedBits;
        constexpr uint component_bits  = static_cast<uint>(sizeof(ValueT) * 8u);
        constexpr uint component_begin = decomposed_suffix_bits<INDEX, Tuple>();
        constexpr uint component_end   = component_begin + component_bits;
        auto           overlap_begin   = std::max(begin_bit, component_begin);
        auto           overlap_end     = std::min(end_bit, component_end);
        if(overlap_begin < overlap_end)
        {
            auto local_begin = overlap_begin - component_begin;
            auto width       = overlap_end - overlap_begin;
            auto mask        = width == component_bits ?
                                   std::numeric_limits<UnsignedBits>::max() :
                                   static_cast<UnsignedBits>((UnsignedBits{1} << width) - 1u);
            auto lhs_bits = (decomposed_ordered_bits(std::get<INDEX>(lhs)) >> local_begin) & mask;
            auto rhs_bits = (decomposed_ordered_bits(std::get<INDEX>(rhs)) >> local_begin) & mask;
            less          = less | (equal & (lhs_bits < rhs_bits));
            equal         = equal & (lhs_bits == rhs_bits);
        }
        if constexpr(INDEX + 1u < std::tuple_size_v<Tuple>)
        {
            compare_decomposed_component<INDEX + 1u>(lhs, rhs, begin_bit, end_bit, less, equal);
        }
    }

    template <typename Tuple>
    [[nodiscard]] static Bool decomposed_less(const Tuple& lhs, const Tuple& rhs, uint begin_bit, uint end_bit)
    {
        Bool less  = false;
        Bool equal = true;
        compare_decomposed_component<0u>(lhs, rhs, begin_bit, end_bit, less, equal);
        return less;
    }

  public:
    DeviceRadixSort()  = default;
    ~DeviceRadixSort() = default;

#ifndef NDEBUG
    Stream* m_debug_stream;  // bind debug stream for sync
#endif
    inline Stream* debug_stream() noexcept
    {
#ifndef NDEBUG
        return m_debug_stream;
#else
        return nullptr;
#endif
    }
    void create(Device& device, Stream* debug_stream = nullptr)
    {
        m_device = device;
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for SortPairs / SortPairsDescending
    template <typename KeyType, typename ValueType>
    static size_t GetSortPairsTempStorageBytes(uint num_items)
    {
        const uint RADIX_BITS          = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint RADIX_DIGITS        = 1 << RADIX_BITS;
        const uint ONESWEEP_TILE_ITEMS = ITEMS_PER_THREAD * BLOCK_SIZE;
        const auto PORTION_SIZE = ((1u << 28u) - 1u) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div((uint)(sizeof(KeyType) * 8), RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t bytes = 0;
        // d_bins: num_portions * num_passes * RADIX_DIGITS * uint
        size_t bins_count = (size_t)num_portions * num_passes * RADIX_DIGITS;
        bytes += bins_count * sizeof(uint);
        // d_lookback: max_num_blocks * RADIX_DIGITS * uint
        size_t lookback_count = (size_t)max_num_blocks * RADIX_DIGITS;
        bytes += lookback_count * sizeof(uint);
        // extra key buffer (for multi-pass): num_items * sizeof(KeyType)
        if(num_passes > 1)
        {
            bytes = align_up_uint(bytes, alignof(KeyType));
            bytes += (size_t)num_items * sizeof(KeyType);
            // extra value buffer
            bytes = align_up_uint(bytes, alignof(ValueType));
            bytes += (size_t)num_items * sizeof(ValueType);
        }
        // d_ctrs: num_portions * num_passes * uint
        size_t ctrs_count = (size_t)num_portions * num_passes;
        bytes += ctrs_count * sizeof(uint);
        return bytes;
    }

    /// Temp storage bytes for SortKeys / SortKeysDescending
    template <typename KeyType>
    static size_t GetSortKeysTempStorageBytes(uint num_items)
    {
        constexpr uint onesweep_block_threads =
            CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 384u : BLOCK_SIZE;
        constexpr uint onesweep_items_per_thread =
            CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 20u : ITEMS_PER_THREAD;
        const uint radix_bits = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint radix_digits = 1u << radix_bits;
        const uint onesweep_tile_items = onesweep_items_per_thread * onesweep_block_threads;
        const auto portion_size = ((1u << 28u) - 1u) / onesweep_tile_items * onesweep_tile_items;
        const auto num_passes = ceil_div(static_cast<uint>(sizeof(KeyType) * 8u), radix_bits);
        const auto num_portions = ceil_div(num_items, portion_size);
        const auto max_num_blocks = ceil_div(std::min(num_items, portion_size), onesweep_tile_items);

        size_t bytes = 0u;
        bytes += static_cast<size_t>(num_portions) * num_passes * radix_digits * sizeof(uint);
        bytes += static_cast<size_t>(max_num_blocks) * radix_digits * sizeof(uint);
        if(num_passes > 1u)
        {
            bytes = align_up_uint(bytes, alignof(KeyType));
            bytes += static_cast<size_t>(num_items) * sizeof(KeyType);
            bytes = align_up_uint(bytes, alignof(KeyType));
            bytes += static_cast<size_t>(num_items) * sizeof(KeyType);
        }
        bytes += static_cast<size_t>(num_portions) * num_passes * sizeof(uint);
        return bytes;
    }


    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT KeyType, NumericT ValueType>
    void SortPairs(CommandList&          cmdlist,
                   BufferView<uint>      temp_storage,
                   BufferView<KeyType>   d_keys_in,
                   BufferView<KeyType>   d_keys_out,
                   BufferView<ValueType> d_values_in,
                   BufferView<ValueType> d_values_out,
                   uint                  num_items,
                   uint                  begin_bit = 0u,
                   uint                  end_bit   = sizeof(KeyType) * 8u)
    {
        validate_pair_views(temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            cmdlist << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items))
                    << d_values_out.subview(0u, num_items).copy_from(d_values_in.subview(0u, num_items));
            return;
        }
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        lcpp_check(onesweep_radix_sort<KeyType, ValueType, false, false>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, false),
                   cmdlist,
                   debug_stream());
    };

    template <NumericT KeyType>
    void SortKeys(CommandList&        cmdlist,
                  BufferView<uint>    temp_storage,
                  BufferView<KeyType> d_keys_in,
                  BufferView<KeyType> d_keys_out,
                  uint                num_items,
                  uint                begin_bit = 0u,
                  uint                end_bit   = sizeof(KeyType) * 8u)
    {
        validate_key_views(temp_storage, d_keys_in, d_keys_out, num_items, begin_bit, end_bit);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            cmdlist << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items));
            return;
        }
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, false>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, false),
                   cmdlist,
                   debug_stream());
    };


    template <NumericT KeyType, NumericT ValueType>
    void SortPairsDescending(CommandList&          cmdlist,
                             BufferView<uint>      temp_storage,
                             BufferView<KeyType>   d_keys_in,
                             BufferView<KeyType>   d_keys_out,
                             BufferView<ValueType> d_values_in,
                             BufferView<ValueType> d_values_out,
                             uint                  num_items,
                             uint                  begin_bit = 0u,
                             uint                  end_bit   = sizeof(KeyType) * 8u)
    {
        validate_pair_views(temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            cmdlist << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items))
                    << d_values_out.subview(0u, num_items).copy_from(d_values_in.subview(0u, num_items));
            return;
        }
        DoubleBuffer<KeyType>   d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<ValueType> d_values(d_values_in, d_values_out);
        lcpp_check(onesweep_radix_sort<KeyType, ValueType, false, true>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, false),
                   cmdlist,
                   debug_stream());
    };

    template <NumericT KeyType>
    void SortKeysDescending(CommandList&        cmdlist,
                            BufferView<uint>    temp_storage,
                            BufferView<KeyType> d_keys_in,
                            BufferView<KeyType> d_keys_out,
                            uint                num_items,
                            uint                begin_bit = 0u,
                            uint                end_bit   = sizeof(KeyType) * 8u)
    {
        validate_key_views(temp_storage, d_keys_in, d_keys_out, num_items, begin_bit, end_bit);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            cmdlist << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items));
            return;
        }
        DoubleBuffer<KeyType> d_keys(d_keys_in, d_keys_out);
        DoubleBuffer<KeyType> d_values(d_keys_in, d_keys_out);  // dummy
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, true>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, false),
                   cmdlist,
                   debug_stream());
    };

    template <NumericT KeyType, NumericT ValueType>
    void SortPairs(CommandList&             cmdlist,
                   BufferView<uint>         temp_storage,
                   DoubleBuffer<KeyType>&   d_keys,
                   DoubleBuffer<ValueType>& d_values,
                   uint                     num_items,
                   uint                     begin_bit = 0u,
                   uint                     end_bit   = sizeof(KeyType) * 8u)
    {
        dispatch_double_buffer<KeyType, ValueType, false, false>(
            cmdlist, temp_storage, d_keys, d_values, num_items, begin_bit, end_bit);
    }

    template <NumericT KeyType>
    void SortKeys(CommandList&           cmdlist,
                  BufferView<uint>       temp_storage,
                  DoubleBuffer<KeyType>& d_keys,
                  uint                   num_items,
                  uint                   begin_bit = 0u,
                  uint                   end_bit   = sizeof(KeyType) * 8u)
    {
        validate_double_buffer(d_keys, num_items, "key");
        auto current   = d_keys.current();
        auto alternate = d_keys.alternate();
        validate_key_views(temp_storage, current, alternate, num_items, begin_bit, end_bit);
        if(num_items == 0u || begin_bit == end_bit)
        {
            return;
        }
        DoubleBuffer<KeyType> d_values = d_keys;
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, false>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, true),
                   cmdlist,
                   debug_stream());
    }

    template <NumericT KeyType, NumericT ValueType>
    void SortPairsDescending(CommandList&             cmdlist,
                             BufferView<uint>         temp_storage,
                             DoubleBuffer<KeyType>&   d_keys,
                             DoubleBuffer<ValueType>& d_values,
                             uint                     num_items,
                             uint                     begin_bit = 0u,
                             uint                     end_bit   = sizeof(KeyType) * 8u)
    {
        dispatch_double_buffer<KeyType, ValueType, false, true>(
            cmdlist, temp_storage, d_keys, d_values, num_items, begin_bit, end_bit);
    }

    template <NumericT KeyType>
    void SortKeysDescending(CommandList&           cmdlist,
                            BufferView<uint>       temp_storage,
                            DoubleBuffer<KeyType>& d_keys,
                            uint                   num_items,
                            uint                   begin_bit = 0u,
                            uint                   end_bit   = sizeof(KeyType) * 8u)
    {
        validate_double_buffer(d_keys, num_items, "key");
        auto current   = d_keys.current();
        auto alternate = d_keys.alternate();
        validate_key_views(temp_storage, current, alternate, num_items, begin_bit, end_bit);
        if(num_items == 0u || begin_bit == end_bit)
        {
            return;
        }
        DoubleBuffer<KeyType> d_values = d_keys;
        lcpp_check(onesweep_radix_sort<KeyType, KeyType, true, true>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, true),
                   cmdlist,
                   debug_stream());
    }

    template <typename KeyType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortKeys(CommandList&        cmdlist,
                  BufferView<uint>    temp_storage,
                  BufferView<KeyType> d_keys_in,
                  BufferView<KeyType> d_keys_out,
                  uint                num_items,
                  Decomposer          decomposer,
                  uint                begin_bit = 0u,
                  uint                end_bit   = decomposed_bit_count<KeyType, Decomposer>())
    {
        dispatch_decomposed_keys<false, false>(
            cmdlist, temp_storage, d_keys_in, d_keys_out, num_items, decomposer, begin_bit, end_bit, nullptr);
    }

    template <typename KeyType, typename ValueType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortPairs(CommandList&          cmdlist,
                   BufferView<uint>      temp_storage,
                   BufferView<KeyType>   d_keys_in,
                   BufferView<KeyType>   d_keys_out,
                   BufferView<ValueType> d_values_in,
                   BufferView<ValueType> d_values_out,
                   uint                  num_items,
                   Decomposer            decomposer,
                   uint                  begin_bit = 0u,
                   uint                  end_bit   = decomposed_bit_count<KeyType, Decomposer>())
    {
        dispatch_decomposed_pairs<false, false>(
            cmdlist, temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, decomposer, begin_bit, end_bit, nullptr, nullptr);
    }

    template <typename KeyType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortKeysDescending(CommandList&        cmdlist,
                            BufferView<uint>    temp_storage,
                            BufferView<KeyType> d_keys_in,
                            BufferView<KeyType> d_keys_out,
                            uint                num_items,
                            Decomposer          decomposer,
                            uint                begin_bit = 0u,
                            uint end_bit = decomposed_bit_count<KeyType, Decomposer>())
    {
        dispatch_decomposed_keys<true, false>(
            cmdlist, temp_storage, d_keys_in, d_keys_out, num_items, decomposer, begin_bit, end_bit, nullptr);
    }

    template <typename KeyType, typename ValueType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortPairsDescending(CommandList&          cmdlist,
                             BufferView<uint>      temp_storage,
                             BufferView<KeyType>   d_keys_in,
                             BufferView<KeyType>   d_keys_out,
                             BufferView<ValueType> d_values_in,
                             BufferView<ValueType> d_values_out,
                             uint                  num_items,
                             Decomposer            decomposer,
                             uint                  begin_bit = 0u,
                             uint end_bit = decomposed_bit_count<KeyType, Decomposer>())
    {
        dispatch_decomposed_pairs<true, false>(
            cmdlist, temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, decomposer, begin_bit, end_bit, nullptr, nullptr);
    }

    template <typename KeyType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortKeys(CommandList&           cmdlist,
                  BufferView<uint>       temp_storage,
                  DoubleBuffer<KeyType>& d_keys,
                  uint                   num_items,
                  Decomposer             decomposer,
                  uint                   begin_bit = 0u,
                  uint                   end_bit   = decomposed_bit_count<KeyType, Decomposer>())
    {
        validate_double_buffer(d_keys, num_items, "key");
        dispatch_decomposed_keys<false, true>(cmdlist,
                                              temp_storage,
                                              d_keys.current(),
                                              d_keys.alternate(),
                                              num_items,
                                              decomposer,
                                              begin_bit,
                                              end_bit,
                                              &d_keys.selector);
    }

    template <typename KeyType, typename ValueType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortPairs(CommandList&             cmdlist,
                   BufferView<uint>         temp_storage,
                   DoubleBuffer<KeyType>&   d_keys,
                   DoubleBuffer<ValueType>& d_values,
                   uint                     num_items,
                   Decomposer               decomposer,
                   uint                     begin_bit = 0u,
                   uint                     end_bit   = decomposed_bit_count<KeyType, Decomposer>())
    {
        validate_decomposed_pair_buffers(d_keys, d_values, num_items);
        dispatch_decomposed_pairs<false, true>(cmdlist,
                                               temp_storage,
                                               d_keys.current(),
                                               d_keys.alternate(),
                                               d_values.current(),
                                               d_values.alternate(),
                                               num_items,
                                               decomposer,
                                               begin_bit,
                                               end_bit,
                                               &d_keys.selector,
                                               &d_values.selector);
    }

    template <typename KeyType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortKeysDescending(CommandList&           cmdlist,
                            BufferView<uint>       temp_storage,
                            DoubleBuffer<KeyType>& d_keys,
                            uint                   num_items,
                            Decomposer             decomposer,
                            uint                   begin_bit = 0u,
                            uint end_bit = decomposed_bit_count<KeyType, Decomposer>())
    {
        validate_double_buffer(d_keys, num_items, "key");
        dispatch_decomposed_keys<true, true>(cmdlist,
                                             temp_storage,
                                             d_keys.current(),
                                             d_keys.alternate(),
                                             num_items,
                                             decomposer,
                                             begin_bit,
                                             end_bit,
                                             &d_keys.selector);
    }

    template <typename KeyType, typename ValueType, typename Decomposer>
        requires requires(Decomposer& decomposer, Var<KeyType>& key) { decomposer(key); }
    void SortPairsDescending(CommandList&             cmdlist,
                             BufferView<uint>         temp_storage,
                             DoubleBuffer<KeyType>&   d_keys,
                             DoubleBuffer<ValueType>& d_values,
                             uint                     num_items,
                             Decomposer               decomposer,
                             uint                     begin_bit = 0u,
                             uint end_bit = decomposed_bit_count<KeyType, Decomposer>())
    {
        validate_decomposed_pair_buffers(d_keys, d_values, num_items);
        dispatch_decomposed_pairs<true, true>(cmdlist,
                                              temp_storage,
                                              d_keys.current(),
                                              d_keys.alternate(),
                                              d_values.current(),
                                              d_values.alternate(),
                                              num_items,
                                              decomposer,
                                              begin_bit,
                                              end_bit,
                                              &d_keys.selector,
                                              &d_values.selector);
    }

  private:
    template <typename FirstT, typename SecondT>
    static void validate_decomposed_non_overlap(BufferView<FirstT> first, BufferView<SecondT> second, uint num_items)
    {
        if(num_items == 0u || first.handle() != second.handle())
        {
            return;
        }
        auto first_begin  = first.offset_bytes();
        auto first_end    = first_begin + static_cast<size_t>(num_items) * first.stride();
        auto second_begin = second.offset_bytes();
        auto second_end   = second_begin + static_cast<size_t>(num_items) * second.stride();
        LUISA_ASSERT(first_end <= second_begin || second_end <= first_begin,
                     "DeviceRadixSort decomposer input and output ranges must not overlap.");
    }

    template <typename KeyType, typename Decomposer>
    void validate_decomposed_key_views(BufferView<KeyType> d_keys_in,
                                       BufferView<KeyType> d_keys_out,
                                       uint                num_items,
                                       uint                begin_bit,
                                       uint                end_bit) const
    {
        constexpr auto total_bits = decomposed_bit_count<KeyType, Decomposer>();
        LUISA_ASSERT(m_created, "DeviceRadixSort::create must be called before dispatch.");
        LUISA_ASSERT(begin_bit <= end_bit, "DeviceRadixSort begin_bit must not exceed end_bit.");
        LUISA_ASSERT(end_bit <= total_bits,
                     "DeviceRadixSort decomposed end_bit {} exceeds the {}-bit key width.",
                     end_bit,
                     total_bits);
        LUISA_ASSERT(d_keys_in.size() >= num_items, "DeviceRadixSort decomposed key input view is too small.");
        LUISA_ASSERT(d_keys_out.size() >= num_items, "DeviceRadixSort decomposed key output view is too small.");
        validate_decomposed_non_overlap(d_keys_in, d_keys_out, num_items);
    }

    template <typename KeyType, typename ValueType>
    void validate_decomposed_pair_buffers(const DoubleBuffer<KeyType>&   d_keys,
                                          const DoubleBuffer<ValueType>& d_values,
                                          uint                           num_items) const
    {
        validate_double_buffer(d_keys, num_items, "key");
        validate_double_buffer(d_values, num_items, "value");
        LUISA_ASSERT(d_keys.selector == d_values.selector,
                     "DeviceRadixSort key and value DoubleBuffer selectors must match.");
    }

    template <bool IS_DESCENDING, bool DOUBLE_BUFFER, typename KeyType, typename Decomposer>
    void dispatch_decomposed_keys(CommandList&        cmdlist,
                                  BufferView<uint>    temp_storage,
                                  BufferView<KeyType> d_keys_in,
                                  BufferView<KeyType> d_keys_out,
                                  uint                num_items,
                                  Decomposer          decomposer,
                                  uint                begin_bit,
                                  uint                end_bit,
                                  int*                selector)
    {
        static_cast<void>(temp_storage);
        validate_decomposed_key_views<KeyType, Decomposer>(d_keys_in, d_keys_out, num_items, begin_bit, end_bit);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            if constexpr(!DOUBLE_BUFFER)
            {
                cmdlist << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items));
            }
            return;
        }

        auto better = [decomposer, begin_bit, end_bit](Var<KeyType> lhs, Var<KeyType> rhs) noexcept
        {
            auto lhs_tuple = decomposer(lhs);
            auto rhs_tuple = decomposer(rhs);
            if constexpr(IS_DESCENDING)
            {
                return decomposed_less(rhs_tuple, lhs_tuple, begin_bit, end_bit);
            }
            else
            {
                return decomposed_less(lhs_tuple, rhs_tuple, begin_bit, end_bit);
            }
        };
        using SortShader = Shader<1, Buffer<KeyType>, Buffer<KeyType>, uint>;
        auto key         = luisa::format("device_radix_sort_decomposed_keys+{}+{}+{}",
                                 IS_DESCENDING ? "desc" : "asc",
                                 DOUBLE_BUFFER ? "double" : "copy",
                                 Type::of<KeyType>()->description());
        key += luisa::format("+{}", get_binary_op_hash<KeyType>(better));
        auto shader_it = m_decomposed_key_shaders.find(key);
        if(shader_it == m_decomposed_key_shaders.end())
        {
            U<SortShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [better](BufferVar<KeyType> input, BufferVar<KeyType> output, UInt num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt         candidate     = dispatch_id().x;
                             Var<KeyType> candidate_key = input.read(candidate);
                             UInt         rank          = 0u;
                             $for(other, 0u, num_items_var, 1u)
                             {
                                 Var<KeyType> other_key        = input.read(other);
                                 Bool         other_better     = better(other_key, candidate_key);
                                 Bool         candidate_better = better(candidate_key, other_key);
                                 Bool         equivalent       = !other_better & !candidate_better;
                                 rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                             };
                             output.write(rank, candidate_key);
                         });
            m_decomposed_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_decomposed_key_shaders.find(key);
        }
        auto shader = reinterpret_cast<SortShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile decomposed DeviceRadixSort key shader.");
        cmdlist << (*shader)(d_keys_in, d_keys_out, num_items).dispatch(num_items);
        if constexpr(DOUBLE_BUFFER)
        {
            LUISA_ASSERT(selector != nullptr, "Missing DeviceRadixSort DoubleBuffer selector.");
            *selector ^= 1;
        }
    }

    template <bool IS_DESCENDING, bool DOUBLE_BUFFER, typename KeyType, typename ValueType, typename Decomposer>
    void dispatch_decomposed_pairs(CommandList&          cmdlist,
                                   BufferView<uint>      temp_storage,
                                   BufferView<KeyType>   d_keys_in,
                                   BufferView<KeyType>   d_keys_out,
                                   BufferView<ValueType> d_values_in,
                                   BufferView<ValueType> d_values_out,
                                   uint                  num_items,
                                   Decomposer            decomposer,
                                   uint                  begin_bit,
                                   uint                  end_bit,
                                   int*                  key_selector,
                                   int*                  value_selector)
    {
        static_cast<void>(temp_storage);
        validate_decomposed_key_views<KeyType, Decomposer>(d_keys_in, d_keys_out, num_items, begin_bit, end_bit);
        LUISA_ASSERT(d_values_in.size() >= num_items, "DeviceRadixSort decomposed value input view is too small.");
        LUISA_ASSERT(d_values_out.size() >= num_items, "DeviceRadixSort decomposed value output view is too small.");
        validate_decomposed_non_overlap(d_keys_in, d_values_out, num_items);
        validate_decomposed_non_overlap(d_values_in, d_keys_out, num_items);
        validate_decomposed_non_overlap(d_values_in, d_values_out, num_items);
        if(num_items == 0u)
        {
            return;
        }
        if(begin_bit == end_bit)
        {
            if constexpr(!DOUBLE_BUFFER)
            {
                cmdlist
                    << d_keys_out.subview(0u, num_items).copy_from(d_keys_in.subview(0u, num_items))
                    << d_values_out.subview(0u, num_items).copy_from(d_values_in.subview(0u, num_items));
            }
            return;
        }

        auto better = [decomposer, begin_bit, end_bit](Var<KeyType> lhs, Var<KeyType> rhs) noexcept
        {
            auto lhs_tuple = decomposer(lhs);
            auto rhs_tuple = decomposer(rhs);
            if constexpr(IS_DESCENDING)
            {
                return decomposed_less(rhs_tuple, lhs_tuple, begin_bit, end_bit);
            }
            else
            {
                return decomposed_less(lhs_tuple, rhs_tuple, begin_bit, end_bit);
            }
        };
        using SortShader =
            Shader<1, Buffer<KeyType>, Buffer<KeyType>, Buffer<ValueType>, Buffer<ValueType>, uint>;
        auto key = luisa::format("device_radix_sort_decomposed_pairs+{}+{}+{}+{}",
                                 IS_DESCENDING ? "desc" : "asc",
                                 DOUBLE_BUFFER ? "double" : "copy",
                                 Type::of<KeyType>()->description(),
                                 Type::of<ValueType>()->description());
        key += luisa::format("+{}", get_binary_op_hash<KeyType>(better));
        auto shader_it = m_decomposed_pair_shaders.find(key);
        if(shader_it == m_decomposed_pair_shaders.end())
        {
            U<SortShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [better](BufferVar<KeyType>   keys_input,
                                  BufferVar<KeyType>   keys_output,
                                  BufferVar<ValueType> values_input,
                                  BufferVar<ValueType> values_output,
                                  UInt                 num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt         candidate     = dispatch_id().x;
                             Var<KeyType> candidate_key = keys_input.read(candidate);
                             UInt         rank          = 0u;
                             $for(other, 0u, num_items_var, 1u)
                             {
                                 Var<KeyType> other_key        = keys_input.read(other);
                                 Bool         other_better     = better(other_key, candidate_key);
                                 Bool         candidate_better = better(candidate_key, other_key);
                                 Bool         equivalent       = !other_better & !candidate_better;
                                 rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                             };
                             keys_output.write(rank, candidate_key);
                             values_output.write(rank, values_input.read(candidate));
                         });
            m_decomposed_pair_shaders.try_emplace(key, std::move(shader));
            shader_it = m_decomposed_pair_shaders.find(key);
        }
        auto shader = reinterpret_cast<SortShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile decomposed DeviceRadixSort pair shader.");
        cmdlist << (*shader)(d_keys_in, d_keys_out, d_values_in, d_values_out, num_items).dispatch(num_items);
        if constexpr(DOUBLE_BUFFER)
        {
            LUISA_ASSERT(key_selector != nullptr && value_selector != nullptr,
                         "Missing DeviceRadixSort pair DoubleBuffer selectors.");
            *key_selector ^= 1;
            *value_selector ^= 1;
        }
    }

    template <typename T>
    static void validate_double_buffer(const DoubleBuffer<T>& buffer, uint num_items, luisa::string_view name)
    {
        LUISA_ASSERT(buffer.selector == 0 || buffer.selector == 1,
                     "DeviceRadixSort {} DoubleBuffer selector must be 0 or 1.",
                     name);
        LUISA_ASSERT(buffer.d_buffer[0].size() >= num_items, "DeviceRadixSort {} DoubleBuffer slot 0 is too small.", name);
        LUISA_ASSERT(buffer.d_buffer[1].size() >= num_items, "DeviceRadixSort {} DoubleBuffer slot 1 is too small.", name);
    }

    template <NumericT KeyType, NumericT ValueType, bool KEY_ONLY, bool IS_DESCENDING>
    void dispatch_double_buffer(CommandList&             cmdlist,
                                BufferView<uint>         temp_storage,
                                DoubleBuffer<KeyType>&   d_keys,
                                DoubleBuffer<ValueType>& d_values,
                                uint                     num_items,
                                uint                     begin_bit,
                                uint                     end_bit)
    {
        validate_double_buffer(d_keys, num_items, "key");
        validate_double_buffer(d_values, num_items, "value");
        LUISA_ASSERT(d_keys.selector == d_values.selector,
                     "DeviceRadixSort key and value DoubleBuffer selectors must match.");
        validate_pair_views(
            temp_storage, d_keys.current(), d_keys.alternate(), d_values.current(), d_values.alternate(), num_items, begin_bit, end_bit);
        if(num_items == 0u || begin_bit == end_bit)
        {
            return;
        }
        lcpp_check(onesweep_radix_sort<KeyType, ValueType, KEY_ONLY, IS_DESCENDING>(
                       cmdlist, temp_storage, d_keys, d_values, begin_bit, end_bit, num_items, true),
                   cmdlist,
                   debug_stream());
    }

    template <NumericT KeyType>
    void validate_key_views(BufferView<uint>    temp_storage,
                            BufferView<KeyType> d_keys_in,
                            BufferView<KeyType> d_keys_out,
                            uint                num_items,
                            uint                begin_bit,
                            uint                end_bit) const
    {
        LUISA_ASSERT(m_created, "DeviceRadixSort::create must be called before dispatch.");
        LUISA_ASSERT(begin_bit <= end_bit, "DeviceRadixSort begin_bit must not exceed end_bit.");
        LUISA_ASSERT(end_bit <= sizeof(KeyType) * 8u, "DeviceRadixSort end_bit exceeds the key width.");
        LUISA_ASSERT(d_keys_in.size() >= num_items, "DeviceRadixSort key input view is too small.");
        LUISA_ASSERT(d_keys_out.size() >= num_items, "DeviceRadixSort key output view is too small.");
        if(num_items > 0u && begin_bit < end_bit)
        {
            LUISA_ASSERT(temp_storage.size()
                             >= bytes_to_uint_count(GetSortKeysTempStorageBytes<KeyType>(num_items)),
                         "DeviceRadixSort temporary storage is too small.");
        }
    }

    template <NumericT KeyType, NumericT ValueType>
    void validate_pair_views(BufferView<uint>      temp_storage,
                             BufferView<KeyType>   d_keys_in,
                             BufferView<KeyType>   d_keys_out,
                             BufferView<ValueType> d_values_in,
                             BufferView<ValueType> d_values_out,
                             uint                  num_items,
                             uint                  begin_bit,
                             uint                  end_bit) const
    {
        validate_key_views(temp_storage, d_keys_in, d_keys_out, num_items, begin_bit, end_bit);
        LUISA_ASSERT(d_values_in.size() >= num_items, "DeviceRadixSort value input view is too small.");
        LUISA_ASSERT(d_values_out.size() >= num_items, "DeviceRadixSort value output view is too small.");
        if(num_items > 0u && begin_bit < end_bit)
        {
            LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(
                             GetSortPairsTempStorageBytes<KeyType, ValueType>(num_items)),
                         "DeviceRadixSort temporary storage is too small.");
        }
    }

    template <NumericT KeyType, typename ValueType, bool KEY_ONLY, bool IS_DESCENDING>
    [[nodiscard]] int onesweep_radix_sort(CommandList&             cmdlist,
                                          BufferView<uint>         temp_storage,
                                          DoubleBuffer<KeyType>&   d_keys,
                                          DoubleBuffer<ValueType>& d_values,
                                          uint                     begin_bit,
                                          uint                     end_bit,
                                          uint                     num_items,
                                          bool                     is_overwrite_okay)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        constexpr uint ONESWEEP_BLOCK_THREADS =
            KEY_ONLY && CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 384u : BLOCK_SIZE;
        constexpr uint ONESWEEP_ITEMS_PER_THREAD =
            KEY_ONLY && CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 20u : ITEMS_PER_THREAD;
        constexpr uint ONESWEEP_TILE_ITEMS = ONESWEEP_ITEMS_PER_THREAD * ONESWEEP_BLOCK_THREADS;

        const auto PORTION_SIZE = ((1u << 28u) - 1u) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div(end_bit - begin_bit, RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        // Carve out sub-buffers from temp_storage
        size_t offset_bytes = 0;

        // d_bins: num_portions * num_passes * RADIX_DIGITS * uint
        size_t bins_count  = (size_t)num_portions * num_passes * RADIX_DIGITS;
        auto   d_bins_view = temp_storage.subview(offset_bytes / sizeof(uint), bins_count);
        offset_bytes += bins_count * sizeof(uint);

        // d_lookback: max_num_blocks * RADIX_DIGITS * uint
        size_t lookback_count  = (size_t)max_num_blocks * RADIX_DIGITS;
        auto   d_lookback_view = temp_storage.subview(offset_bytes / sizeof(uint), lookback_count);
        offset_bytes += lookback_count * sizeof(uint);

        // extra key/value buffers for multi-pass
        BufferView<KeyType>   d_keys_tmp2_view;
        BufferView<ValueType> d_values_tmp2_view;
        if(!is_overwrite_okay && num_passes > 1)
        {
            offset_bytes           = align_up_uint(offset_bytes, alignof(KeyType));
            size_t keys_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(KeyType));
            d_keys_tmp2_view =
                temp_storage.subview(offset_bytes / sizeof(uint), keys_uint_count).template as<KeyType>();
            offset_bytes += keys_uint_count * sizeof(uint);

            offset_bytes             = align_up_uint(offset_bytes, alignof(ValueType));
            size_t values_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(ValueType));
            d_values_tmp2_view =
                temp_storage.subview(offset_bytes / sizeof(uint), values_uint_count).template as<ValueType>();
            offset_bytes += values_uint_count * sizeof(uint);
        }

        // d_ctrs: num_portions * num_passes * uint
        size_t ctrs_count  = (size_t)num_portions * num_passes;
        auto   d_ctrs_view = temp_storage.subview(offset_bytes / sizeof(uint), ctrs_count);

        auto radix_sort_key = get_type_and_op_desc<KeyType, ValueType>()
                              + luisa::string(IS_DESCENDING ? "_desc" : "_asc");


        // reset keys
        using RadixSortReset        = details::RadixSortResetModule<uint>;
        using RadixSortResetKernel  = RadixSortReset::RadixSortResetKernel;
        auto ms_radix_sort_reset_it = ms_radix_sort_reset_map.find(radix_sort_key);
        if(ms_radix_sort_reset_it == ms_radix_sort_reset_map.end())
        {
            auto shader = RadixSortReset().compile(m_device);
            if(!shader)
            {
                return -1;
            }
            auto [it, inserted] = ms_radix_sort_reset_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_reset_it = it;
        }
        if(ms_radix_sort_reset_it == ms_radix_sort_reset_map.end())
        {
            return -1;
        }
        auto ms_radix_sort_reset_ptr =
            reinterpret_cast<RadixSortResetKernel*>(&(*ms_radix_sort_reset_it->second));
        if(!ms_radix_sort_reset_ptr)
        {
            return -1;
        }

        cmdlist << (*ms_radix_sort_reset_ptr)(d_bins_view, 0u).dispatch(bins_count)
                << (*ms_radix_sort_reset_ptr)(d_ctrs_view, 0u).dispatch(ctrs_count);

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
        using RadixSortHistogramKernel  = RadixSortHistogram::RadixSortHistogramKernel;
        auto ms_radix_sort_histogram_it = ms_radix_sort_histogram_map.find(radix_sort_key);
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            auto shader = RadixSortHistogram().compile(m_device);
            if(!shader)
            {
                return -1;
            }
            auto [it, inserted] = ms_radix_sort_histogram_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_histogram_it = it;
        }
        if(ms_radix_sort_histogram_it == ms_radix_sort_histogram_map.end())
        {
            return -1;
        }
        auto ms_radix_sort_histogram_ptr =
            reinterpret_cast<RadixSortHistogramKernel*>(&(*ms_radix_sort_histogram_it->second));
        if(!ms_radix_sort_histogram_ptr)
        {
            return -1;
        }
        const auto num_sms             = BLOCK_SIZE;
        const auto histo_blocks_per_sm = 1;
        cmdlist << (*ms_radix_sort_histogram_ptr)(
                       d_bins_view, ByteBufferView{d_keys.current()}, num_items, begin_bit, end_bit)
                       .dispatch(num_sms * histo_blocks_per_sm * m_block_size);

        // exclusive scan
        using RadixSortExclusiveSum = details::RadixSortExclusiveSumModule<RADIX_DIGITS, BLOCK_SIZE, WARP_NUMS>;
        using RadixSortExclusiveSumKernel   = RadixSortExclusiveSum::RadixSortExclusiveSumKernel;
        auto ms_radix_sort_exclusive_sum_it = ms_radix_sort_exclusive_sum_map.find(radix_sort_key);
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end())
        {
            auto shader = RadixSortExclusiveSum().compile(m_device);
            if(!shader)
            {
                return -1;
            }
            auto [it, inserted] =
                ms_radix_sort_exclusive_sum_map.try_emplace(radix_sort_key, std::move(shader));
            ms_radix_sort_exclusive_sum_it = it;
        }
        if(ms_radix_sort_exclusive_sum_it == ms_radix_sort_exclusive_sum_map.end())
        {
            return -1;
        }
        auto ms_radix_sort_exclusive_sum_ptr =
            reinterpret_cast<RadixSortExclusiveSumKernel*>(&(*ms_radix_sort_exclusive_sum_it->second));
        if(!ms_radix_sort_exclusive_sum_ptr)
        {
            return -1;
        }

        cmdlist << (*ms_radix_sort_exclusive_sum_ptr)(d_bins_view).dispatch(num_passes * m_block_size);

        // one sweep
        auto d_keys_tmp   = d_keys.alternate();
        auto d_values_tmp = d_values.alternate();
        if(!is_overwrite_okay && num_passes % 2 == 0)
        {
            d_keys.d_buffer[1]   = d_keys_tmp2_view;
            d_values.d_buffer[1] = d_values_tmp2_view;
        }

        using RadixSortOneSweep =
            details::RadixSortOneSweepModule<KeyType,
                                             ValueType,
                                             KEY_ONLY,
                                             IS_DESCENDING,
                                             RADIX_BITS,
                                             ONESWEEP_BLOCK_THREADS,
                                             WARP_NUMS,
                                             ONESWEEP_ITEMS_PER_THREAD>;
        using RadixSortOneSweepKernel = RadixSortOneSweep::RadixSortOneSweepKernel;

        auto onesweep_radix_sort_key   = radix_sort_key + (KEY_ONLY ? "_keys" : "_pairs");
        auto ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(onesweep_radix_sort_key);
        if(ms_radix_sort_onesweep_it == ms_radix_sort_one_sweep_map.end())
        {
            auto shader = RadixSortOneSweep().compile(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_radix_sort_one_sweep_map.try_emplace(onesweep_radix_sort_key, std::move(shader));
            ms_radix_sort_onesweep_it = ms_radix_sort_one_sweep_map.find(onesweep_radix_sort_key);
        }
        if(ms_radix_sort_onesweep_it == ms_radix_sort_one_sweep_map.end())
        {
            return -1;
        }
        auto ms_radix_sort_onesweep_ptr =
            reinterpret_cast<RadixSortOneSweepKernel*>(&(*ms_radix_sort_onesweep_it->second));
        if(!ms_radix_sort_onesweep_ptr)
        {
            return -1;
        }

        for(uint current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            uint num_bit = std::min(end_bit - current_bit, RADIX_BITS);

            for(uint portion = 0; portion < num_portions; ++portion)
            {
                uint portion_num_items = std::min(num_items - portion * PORTION_SIZE, PORTION_SIZE);
                uint num_blocks        = ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

                cmdlist << (*ms_radix_sort_reset_ptr)(d_lookback_view, 0u).dispatch(lookback_count);

                // dispatch
                cmdlist
                    << (*ms_radix_sort_onesweep_ptr)(
                           d_lookback_view,
                           d_ctrs_view.subview(portion * num_passes + pass, 1),
                           d_bins_view.subview((portion * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS),
                           portion < num_portions - 1 ?
                               d_bins_view.subview(((portion + 1) * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS) :
                               d_bins_view.subview(0, RADIX_DIGITS),  // dummy: last portion, bins_out unused but must be valid-sized
                           ByteBufferView{d_keys.current().subview(portion * PORTION_SIZE, portion_num_items)},
                           ByteBufferView{d_keys.alternate()},
                           KEY_ONLY ? d_values.current().subview(0u, 1u) :
                                      d_values.current().subview(portion * PORTION_SIZE, portion_num_items),
                           KEY_ONLY ? d_values.alternate().subview(0u, 1u) :
                                      d_values.alternate().subview(portion * PORTION_SIZE, portion_num_items),
                           portion_num_items,
                           current_bit,
                           num_bit)
                           .dispatch(num_blocks * ONESWEEP_BLOCK_THREADS);
            }
            if(!is_overwrite_okay && pass == 0)
            {
                d_keys = num_passes % 2 == 0 ? DoubleBuffer<KeyType>(d_keys_tmp, d_keys_tmp2_view) :
                                               DoubleBuffer<KeyType>(d_keys_tmp2_view, d_keys_tmp);
                d_values = num_passes % 2 == 0 ?
                               DoubleBuffer<ValueType>(d_values_tmp, d_values_tmp2_view) :
                               DoubleBuffer<ValueType>(d_values_tmp2_view, d_values_tmp);
            }
            d_keys.selector ^= 1;
            d_values.selector ^= 1;
        }

        return 0;
    }

  private:
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_histogram_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_exclusive_sum_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_one_sweep_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_reset_map;
};
}  // namespace luisa::parallel_primitive
