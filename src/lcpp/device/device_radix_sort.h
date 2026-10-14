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
#include <lcpp/runtime/interop_dispatch.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/utils.h>
#include <lcpp/agent/policy.h>
#include <lcpp/device/details/radix_sort.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
// Radix one-sweep policy default (epoch-3): BLOCK_SIZE x ITEMS_PER_THREAD tile.
// Tuned on RTX 5060 (4M uint32 SortKeys): 256x16 is the best portable policy
// (cuda 0.43 ms ~ CUB parity, dx 0.67 ms, vk 2.48 ms) -- larger tiles cut the
// block count (hence the per-pass lookback chain length and the 1M-uint lookback
// reset volume) and improve per-thread ILP. Shared bytes ~= BLOCK*ITEMS*sizeof(k)
// + (BLOCK/32)*RADIX_DIGITS*4 + RADIX_DIGITS*4; 256x16 uint32 ~= 26 KiB, under the
// 32 KiB DX/HLSL floor. See docs/epoch3_radix_sort.md.
static constexpr size_t RADIX_ITEMS_PER_THREAD = 16;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE,
          size_t WARP_NUMS = details::WARP_SIZE,
          size_t ITEMS_PER_THREAD = RADIX_ITEMS_PER_THREAD,
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
    // Optional vk-cuda interop sink (epoch B).  Null => pure-vulkan DSL path.
    InteropContext* m_interop = nullptr;

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
    // `interop` is optional: pass a valid InteropContext (see
    // lcpp/runtime/interop_dispatch.h) to launch the four one-sweep radix
    // kernel families as cuda-backend modules inside the vk command buffer.
    // Without it (or when the context is null because the fast path is
    // unavailable) every op keeps its pure-vulkan DSL dispatch path.
    void create(Device& device, Stream* debug_stream = nullptr, InteropContext* interop = nullptr)
    {
        m_device                   = device;
        // Epoch C per-op policy: `LCPP_INTEROP` (1|all|auto|a,b,c) decides
        // per module whether this one takes the fast path; see
        // lcpp/runtime/interop_dispatch.h.
        m_interop = interop_for(interop, "radix");
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    /// True when this module dispatches its radix kernels through the vk-cuda
    /// interop fast path (cuda-compiled module, vk command buffer).
    [[nodiscard]] bool interop_active() const noexcept { return m_interop != nullptr; }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for SortPairs / SortPairsDescending
    template <typename KeyType, typename ValueType>
    static size_t GetSortPairsTempStorageBytes(uint num_items)
    {
        const uint RADIX_BITS   = OneSweepSmallKeyTunedPolicy<KeyType>::ONESWEEP_RADIX_BITS;
        const uint RADIX_DIGITS = 1 << RADIX_BITS;
        // Effective tile: scaled down for >4-byte keys (see radix_effective_items_v).
        const uint ONESWEEP_TILE_ITEMS =
            static_cast<uint>(radix_effective_items_v<ITEMS_PER_THREAD, KeyType> * BLOCK_SIZE);
        const auto PORTION_SIZE = ((1u << 28u) - 1u) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;

        auto num_passes     = ceil_div((uint)(sizeof(KeyType) * 8), RADIX_BITS);
        auto num_portions   = ceil_div(num_items, PORTION_SIZE);
        auto max_num_blocks = ceil_div(std::min(num_items, PORTION_SIZE), ONESWEEP_TILE_ITEMS);

        size_t bytes = 0;
        // d_bins: num_portions * num_passes * RADIX_DIGITS * uint
        size_t bins_count = (size_t)num_portions * num_passes * RADIX_DIGITS;
        bytes += bins_count * sizeof(uint);
        // d_lookback: max_num_blocks * RADIX_DIGITS * uint (descriptor-aligned base)
        bytes = align_up_view(bytes);
        size_t lookback_count = (size_t)max_num_blocks * RADIX_DIGITS;
        bytes += lookback_count * sizeof(uint);
        // extra key buffer (for multi-pass): num_items * sizeof(KeyType)
        if(num_passes > 1)
        {
            bytes = align_up_view(bytes);
            bytes += (size_t)num_items * sizeof(KeyType);
            // extra value buffer
            bytes = align_up_view(bytes);
            bytes += (size_t)num_items * sizeof(ValueType);
        }
        // d_ctrs: one descriptor-aligned 256-byte slot per (portion, pass);
        // must mirror the runtime carving in onesweep_radix_sort().
        bytes = align_up_view(bytes);
        size_t ctrs_count = (size_t)num_portions * num_passes * descriptor_view_alignment_uints;
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
        // Remote CUB parity: the tuned 32-bit key-only one-sweep policy
        // uses 384 threads x 20 items unless the caller opts out
        // (CUB_SMALL_KEY_POLICY=false). The local epoch-3/3.5 tuning then
        // scales items-per-thread down for >4-byte keys so the tile, the
        // shared footprint and the look-back chain stay bounded
        // (see radix_effective_items_v).
        constexpr uint ONESWEEP_BLOCK_THREADS =
            KEY_ONLY && CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 384u : BLOCK_SIZE;
        constexpr uint ONESWEEP_POLICY_ITEMS_PER_THREAD =
            KEY_ONLY && CUB_SMALL_KEY_POLICY && sizeof(KeyType) == 4u ? 20u : ITEMS_PER_THREAD;
        constexpr uint ONESWEEP_ITEMS_PER_THREAD = static_cast<uint>(
            radix_effective_items_v<ONESWEEP_POLICY_ITEMS_PER_THREAD, KeyType>);
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

        // d_lookback: max_num_blocks * RADIX_DIGITS * uint (descriptor-aligned base)
        offset_bytes = align_up_view(offset_bytes);
        size_t lookback_count = (size_t)max_num_blocks * RADIX_DIGITS;
        auto d_lookback_view = temp_storage.subview(offset_bytes / sizeof(uint), lookback_count);
        offset_bytes += lookback_count * sizeof(uint);

        // extra key/value buffers for multi-pass (descriptor-aligned bases)
        BufferView<KeyType>   d_keys_tmp2_view;
        BufferView<ValueType> d_values_tmp2_view;
        if(!is_overwrite_okay && num_passes > 1)
        {
            offset_bytes = align_up_view(offset_bytes);
            size_t keys_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(KeyType));
            d_keys_tmp2_view =
                temp_storage.subview(offset_bytes / sizeof(uint), keys_uint_count).template as<KeyType>();
            offset_bytes += keys_uint_count * sizeof(uint);

            offset_bytes = align_up_view(offset_bytes);
            size_t values_uint_count = bytes_to_uint_count((size_t)num_items * sizeof(ValueType));
            d_values_tmp2_view =
                temp_storage.subview(offset_bytes / sizeof(uint), values_uint_count).template as<ValueType>();
            offset_bytes += values_uint_count * sizeof(uint);
        }

        // d_ctrs: one descriptor-aligned 256-byte slot per (portion, pass) -- the
        // per-pass counter is passed to the one-sweep kernel as a 1-element
        // subview, whose offset must satisfy the descriptor-view alignment on
        // the legacy Vulkan HLSL path. Mirrors GetSortPairsTempStorageBytes.
        offset_bytes = align_up_view(offset_bytes);
        size_t ctrs_count = (size_t)num_portions * num_passes * descriptor_view_alignment_uints;
        auto d_ctrs_view = temp_storage.subview(offset_bytes / sizeof(uint), ctrs_count);

        auto radix_sort_key = get_radix_sort_key<KeyType, ValueType, KEY_ONLY, IS_DESCENDING>();


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

        // ---- interop fast path: reset kernels ----
        // Usage table: the reset kernel writes exactly one entry per dispatch
        // thread (`d_bins_out.write(dispatch_id().x, init_value)`) and never
        // reads the buffer, so WRITE is the accurate contract.
        auto reset_interop_it = ms_radix_sort_reset_interop_map.end();
        if(m_interop != nullptr)
        {
            reset_interop_it = ms_radix_sort_reset_interop_map.find(radix_sort_key);
            if(reset_interop_it == ms_radix_sort_reset_interop_map.end())
            {
                reset_interop_it = import_into_interop_cache(
                    m_interop,
                    ms_radix_sort_reset_interop_map,
                    radix_sort_key,
                    RadixSortReset().compile(m_interop->cuda_device()));
            }
        }
        if(reset_interop_it != ms_radix_sort_reset_interop_map.end())
        {
            auto kernel = reset_interop_it->second.shader.template kernel<
                vk_cuda_interop::CudaArg<Buffer<uint>, Usage::WRITE>,
                uint>();
            cmdlist << kernel(d_bins_view, 0u).dispatch(static_cast<uint>(bins_count))
                    << kernel(d_ctrs_view, 0u).dispatch(static_cast<uint>(ctrs_count));
        }
        else
        {
            cmdlist << (*ms_radix_sort_reset_ptr)(d_bins_view, 0u).dispatch(bins_count)
                    << (*ms_radix_sort_reset_ptr)(d_ctrs_view, 0u).dispatch(ctrs_count);
        }

        // radix sort histogram
        using RadixSortHistogram =
            details::RadixSortHistogramModule<KeyType, IS_DESCENDING, RADIX_BITS, BLOCK_SIZE, WARP_NUMS,
                                          radix_effective_items_v<ITEMS_PER_THREAD, KeyType>>;
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
        const auto histo_threads       = static_cast<uint>(num_sms * histo_blocks_per_sm * m_block_size);
        auto histogram_interop_it      = ms_radix_sort_histogram_interop_map.end();
        // ---- interop fast path: histogram ----
        // Usage table: d_bins_out is accumulated atomically
        // (`atomic(...).fetch_add(count)`) hence READ_WRITE; the key channel is
        // a read-only ByteBuffer.
        if(m_interop != nullptr)
        {
            histogram_interop_it = ms_radix_sort_histogram_interop_map.find(radix_sort_key);
            if(histogram_interop_it == ms_radix_sort_histogram_interop_map.end())
            {
                histogram_interop_it = import_into_interop_cache(
                    m_interop,
                    ms_radix_sort_histogram_interop_map,
                    radix_sort_key,
                    RadixSortHistogram().compile(m_interop->cuda_device()));
            }
        }
        if(histogram_interop_it != ms_radix_sort_histogram_interop_map.end())
        {
            auto kernel = histogram_interop_it->second.shader.template kernel<
                vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                vk_cuda_interop::CudaArg<ByteBuffer, Usage::READ>,
                uint,
                uint,
                uint>();
            cmdlist << kernel(d_bins_view, ByteBufferView{d_keys.current()}, num_items, begin_bit, end_bit)
                           .dispatch(histo_threads);
        }
        else
        {
            cmdlist << (*ms_radix_sort_histogram_ptr)(
                           d_bins_view, ByteBufferView{d_keys.current()}, num_items, begin_bit, end_bit)
                           .dispatch(histo_threads);
        }

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

        const auto exclusive_sum_threads = static_cast<uint>(num_passes) * m_block_size;
        auto       exclusive_sum_interop_it = ms_radix_sort_exclusive_sum_interop_map.end();
        // ---- interop fast path: per-digit exclusive scan ----
        // Usage table: the kernel reads its 256-bin block out of d_bins and
        // writes the exclusive prefix back into the same range => READ_WRITE.
        if(m_interop != nullptr)
        {
            exclusive_sum_interop_it = ms_radix_sort_exclusive_sum_interop_map.find(radix_sort_key);
            if(exclusive_sum_interop_it == ms_radix_sort_exclusive_sum_interop_map.end())
            {
                exclusive_sum_interop_it = import_into_interop_cache(
                    m_interop,
                    ms_radix_sort_exclusive_sum_interop_map,
                    radix_sort_key,
                    RadixSortExclusiveSum().compile(m_interop->cuda_device()));
            }
        }
        if(exclusive_sum_interop_it != ms_radix_sort_exclusive_sum_interop_map.end())
        {
            auto kernel = exclusive_sum_interop_it->second.shader.template kernel<
                vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>>();
            cmdlist << kernel(d_bins_view).dispatch(exclusive_sum_threads);
        }
        else
        {
            cmdlist << (*ms_radix_sort_exclusive_sum_ptr)(d_bins_view).dispatch(exclusive_sum_threads);
        }

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

        // the local pointer-based radix key already folds in KEY_ONLY (see
        // get_radix_sort_key), so the remote "_keys"/"_pairs" suffix is
        // redundant and does not apply to the allocation-free key type.
        const auto &onesweep_radix_sort_key = radix_sort_key;
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

        // ---- interop fast path: the one-sweep kernel ----
        // This is the kernel the whole epoch is about: its rank (MatchAny) and
        // its volatile look-back spin-polling are exactly the constructs the
        // XIR->SPIR-V path lowers poorly and native CUDA lowers well.
        //
        // Usage table (per launch, cross-portion carry included):
        //   d_lookback        READ_WRITE  partial/global look-back channel
        //                                 (volatile_read spin + volatile_write)
        //   d_ctrs            READ_WRITE  atomic block-index counter
        //   d_bins_in         READ        this portion/pass digit offsets
        //   d_bins_out        WRITE       the (portion+1) carry for this pass
        //   d_keys_in  (Byte) READ        keys of this portion
        //   d_keys_out (Byte) WRITE       scatter destination (the alternate
        //                                 buffer, a different allocation from
        //                                 d_keys_in even when the ranges
        //                                 overlap)
        //   d_values_in       READ        values of this portion
        //   d_values_out      WRITE       scatter destination
        auto onesweep_interop_it = ms_radix_sort_one_sweep_interop_map.end();
        if(m_interop != nullptr)
        {
            onesweep_interop_it = ms_radix_sort_one_sweep_interop_map.find(radix_sort_key);
            if(onesweep_interop_it == ms_radix_sort_one_sweep_interop_map.end())
            {
                onesweep_interop_it = import_into_interop_cache(
                    m_interop,
                    ms_radix_sort_one_sweep_interop_map,
                    radix_sort_key,
                    RadixSortOneSweep().compile(m_interop->cuda_device()));
            }
        }

        for(uint current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            uint num_bit = std::min(end_bit - current_bit, RADIX_BITS);

            for(uint portion = 0; portion < num_portions; ++portion)
            {
                uint portion_num_items = std::min(num_items - portion * PORTION_SIZE, PORTION_SIZE);
                uint num_blocks        = ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

                if(reset_interop_it != ms_radix_sort_reset_interop_map.end())
                {
                    auto kernel = reset_interop_it->second.shader.template kernel<
                        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::WRITE>,
                        uint>();
                    cmdlist << kernel(d_lookback_view, 0u).dispatch(static_cast<uint>(lookback_count));
                }
                else
                {
                    cmdlist << (*ms_radix_sort_reset_ptr)(d_lookback_view, 0u).dispatch(lookback_count);
                }

                auto bins_out_view =
                    portion < num_portions - 1 ?
                        d_bins_view.subview(((portion + 1) * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS) :
                        d_bins_view.subview(0, RADIX_DIGITS);  // dummy: last portion, bins_out unused but must be valid-sized
                auto ctrs_view = d_ctrs_view.subview((portion * num_passes + pass) * descriptor_view_alignment_uints, 1);
                auto bins_in_view = d_bins_view.subview((portion * num_passes + pass) * RADIX_DIGITS, RADIX_DIGITS);
                // KEY_ONLY: pass a 1-element dummy view instead of an empty one --
                // Vulkan forbids empty storage-buffer descriptors, and the
                // values are never read/written when KEY_ONLY is true.
                auto values_in_view = KEY_ONLY ? d_values.current().subview(0, 1) :
                                                 d_values.current().subview(portion * PORTION_SIZE, portion_num_items);
                auto values_out_view = KEY_ONLY ? d_values.alternate().subview(0, 1) :
                                                  d_values.alternate().subview(portion * PORTION_SIZE, portion_num_items);
                auto keys_in_view =
                    ByteBufferView{d_keys.current().subview(portion * PORTION_SIZE, portion_num_items)};
                auto keys_out_view = ByteBufferView{d_keys.alternate()};

                // dispatch
                if(onesweep_interop_it != ms_radix_sort_one_sweep_interop_map.end())
                {
                    auto kernel = onesweep_interop_it->second.shader.template kernel<
                        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ>,
                        vk_cuda_interop::CudaArg<Buffer<uint>, Usage::WRITE>,
                        vk_cuda_interop::CudaArg<ByteBuffer, Usage::READ>,
                        vk_cuda_interop::CudaArg<ByteBuffer, Usage::WRITE>,
                        vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::READ>,
                        vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::WRITE>,
                        uint,
                        uint,
                        uint>();
                    cmdlist << kernel(d_lookback_view,
                                      ctrs_view,
                                      bins_in_view,
                                      bins_out_view,
                                      keys_in_view,
                                      keys_out_view,
                                      values_in_view,
                                      values_out_view,
                                      portion_num_items,
                                      current_bit,
                                      num_bit)
                                   .dispatch(num_blocks * ONESWEEP_BLOCK_THREADS);
                }
                else
                {
                    cmdlist << (*ms_radix_sort_onesweep_ptr)(d_lookback_view,
                                                             ctrs_view,
                                                             bins_in_view,
                                                             bins_out_view,
                                                             keys_in_view,
                                                             keys_out_view,
                                                             values_in_view,
                                                             values_out_view,
                                                             portion_num_items,
                                                             current_bit,
                                                             num_bit)
                                   .dispatch(num_blocks * ONESWEEP_BLOCK_THREADS);
                }
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
    detail::ShaderCacheMap ms_radix_sort_histogram_map;
    detail::ShaderCacheMap ms_radix_sort_exclusive_sum_map;
    detail::ShaderCacheMap ms_radix_sort_one_sweep_map;
    detail::ShaderCacheMap ms_radix_sort_reset_map;

    // epoch B: vk-cuda interop kernels shadowing the four maps above (risk R6:
    // destroyed with the module, hence before the vk device they borrow).
    detail::InteropShaderMap ms_radix_sort_histogram_interop_map;
    detail::InteropShaderMap ms_radix_sort_exclusive_sum_interop_map;
    detail::InteropShaderMap ms_radix_sort_one_sweep_interop_map;
    detail::InteropShaderMap ms_radix_sort_reset_interop_map;
};
}  // namespace luisa::parallel_primitive
