/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/device/device_segmented_sort.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceSegmentedRadixSort : public LuisaModule
{
  private:
    DeviceSegmentedSort<BLOCK_SIZE> m_segmented_sort;

  public:
    DeviceSegmentedRadixSort()  = default;
    ~DeviceSegmentedRadixSort() = default;

    void create(Device& device) { m_segmented_sort.create(device); }

    template <NumericT KeyT>
    [[nodiscard]] static size_t GetSortKeysTempStorageBytes(size_t num_items)
    {
        return DeviceSegmentedSort<BLOCK_SIZE>::template GetSortKeysTempStorageBytes<KeyT>(num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    [[nodiscard]] static size_t GetSortPairsTempStorageBytes(size_t num_items)
    {
        return DeviceSegmentedSort<BLOCK_SIZE>::template GetSortPairsTempStorageBytes<KeyT, ValueT>(num_items);
    }

    template <NumericT KeyT>
    void SortKeys(CommandList&     cmdlist,
                  BufferView<uint> temp_storage,
                  BufferView<KeyT> keys_input,
                  BufferView<KeyT> keys_output,
                  size_t           num_items,
                  size_t           num_segments,
                  BufferView<uint> begin_offsets,
                  BufferView<uint> end_offsets,
                  uint             begin_bit = 0u,
                  uint             end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        auto compare = radix_compare<KeyT, false>(begin_bit, end_bit);
        m_segmented_sort.SortKeys(
            cmdlist, temp_storage, keys_input, keys_output, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT>
    void SortKeys(CommandList&        cmdlist,
                  BufferView<uint>    temp_storage,
                  DoubleBuffer<KeyT>& keys,
                  size_t              num_items,
                  size_t              num_segments,
                  BufferView<uint>    begin_offsets,
                  BufferView<uint>    end_offsets,
                  uint                begin_bit = 0u,
                  uint                end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        if(begin_bit == end_bit)
        {
            m_segmented_sort.validate_key_double_buffer(keys, num_items, num_segments, begin_offsets, end_offsets);
            return;
        }
        auto compare = radix_compare<KeyT, false>(begin_bit, end_bit);
        m_segmented_sort.SortKeys(
            cmdlist, temp_storage, keys, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT>
    void SortKeysDescending(CommandList&     cmdlist,
                            BufferView<uint> temp_storage,
                            BufferView<KeyT> keys_input,
                            BufferView<KeyT> keys_output,
                            size_t           num_items,
                            size_t           num_segments,
                            BufferView<uint> begin_offsets,
                            BufferView<uint> end_offsets,
                            uint             begin_bit = 0u,
                            uint             end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        auto compare = radix_compare<KeyT, true>(begin_bit, end_bit);
        m_segmented_sort.SortKeys(
            cmdlist, temp_storage, keys_input, keys_output, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT>
    void SortKeysDescending(CommandList&        cmdlist,
                            BufferView<uint>    temp_storage,
                            DoubleBuffer<KeyT>& keys,
                            size_t              num_items,
                            size_t              num_segments,
                            BufferView<uint>    begin_offsets,
                            BufferView<uint>    end_offsets,
                            uint                begin_bit = 0u,
                            uint                end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        if(begin_bit == end_bit)
        {
            m_segmented_sort.validate_key_double_buffer(keys, num_items, num_segments, begin_offsets, end_offsets);
            return;
        }
        auto compare = radix_compare<KeyT, true>(begin_bit, end_bit);
        m_segmented_sort.SortKeys(
            cmdlist, temp_storage, keys, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&       cmdlist,
                   BufferView<uint>   temp_storage,
                   BufferView<KeyT>   keys_input,
                   BufferView<KeyT>   keys_output,
                   BufferView<ValueT> values_input,
                   BufferView<ValueT> values_output,
                   size_t             num_items,
                   size_t             num_segments,
                   BufferView<uint>   begin_offsets,
                   BufferView<uint>   end_offsets,
                   uint               begin_bit = 0u,
                   uint               end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        auto compare = radix_compare<KeyT, false>(begin_bit, end_bit);
        m_segmented_sort.SortPairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&          cmdlist,
                   BufferView<uint>      temp_storage,
                   DoubleBuffer<KeyT>&   keys,
                   DoubleBuffer<ValueT>& values,
                   size_t                num_items,
                   size_t                num_segments,
                   BufferView<uint>      begin_offsets,
                   BufferView<uint>      end_offsets,
                   uint                  begin_bit = 0u,
                   uint                  end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        if(begin_bit == end_bit)
        {
            m_segmented_sort.validate_pair_double_buffers(keys, values, num_items, num_segments, begin_offsets, end_offsets);
            return;
        }
        auto compare = radix_compare<KeyT, false>(begin_bit, end_bit);
        m_segmented_sort.SortPairs(
            cmdlist, temp_storage, keys, values, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairsDescending(CommandList&       cmdlist,
                             BufferView<uint>   temp_storage,
                             BufferView<KeyT>   keys_input,
                             BufferView<KeyT>   keys_output,
                             BufferView<ValueT> values_input,
                             BufferView<ValueT> values_output,
                             size_t             num_items,
                             size_t             num_segments,
                             BufferView<uint>   begin_offsets,
                             BufferView<uint>   end_offsets,
                             uint               begin_bit = 0u,
                             uint               end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        auto compare = radix_compare<KeyT, true>(begin_bit, end_bit);
        m_segmented_sort.SortPairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairsDescending(CommandList&          cmdlist,
                             BufferView<uint>      temp_storage,
                             DoubleBuffer<KeyT>&   keys,
                             DoubleBuffer<ValueT>& values,
                             size_t                num_items,
                             size_t                num_segments,
                             BufferView<uint>      begin_offsets,
                             BufferView<uint>      end_offsets,
                             uint                  begin_bit = 0u,
                             uint                  end_bit   = sizeof(KeyT) * 8u)
    {
        validate_bit_range<KeyT>(begin_bit, end_bit);
        if(begin_bit == end_bit)
        {
            m_segmented_sort.validate_pair_double_buffers(keys, values, num_items, num_segments, begin_offsets, end_offsets);
            return;
        }
        auto compare = radix_compare<KeyT, true>(begin_bit, end_bit);
        m_segmented_sort.SortPairs(
            cmdlist, temp_storage, keys, values, num_items, num_segments, begin_offsets, end_offsets, compare);
    }

  private:
    template <NumericT KeyT>
    static void validate_bit_range(uint begin_bit, uint end_bit)
    {
        constexpr auto bit_width = static_cast<uint>(sizeof(KeyT) * 8u);
        LUISA_ASSERT(begin_bit <= end_bit, "DeviceSegmentedRadixSort begin_bit must not exceed end_bit.");
        LUISA_ASSERT(end_bit <= bit_width, "DeviceSegmentedRadixSort end_bit exceeds the key width.");
    }

    template <NumericT KeyT, bool DESCENDING>
    [[nodiscard]] static Callable<bool(KeyT, KeyT)> radix_compare(uint begin_bit, uint end_bit)
    {
        using UnsignedBits       = typename Traits<KeyT>::UnsignedBits;
        constexpr auto bit_width = static_cast<uint>(sizeof(KeyT) * 8u);
        auto           bit_count = end_bit - begin_bit;
        if(bit_count == 0u)
        {
            return [](Var<KeyT>, Var<KeyT>) noexcept { return false; };
        }
        auto mask = UnsignedBits(-1);
        if(bit_count < bit_width)
        {
            mask = static_cast<UnsignedBits>((UnsignedBits{1} << bit_count) - UnsignedBits{1});
        }
        return [begin_bit, mask](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        {
            auto shift    = Var<UnsignedBits>(static_cast<UnsignedBits>(begin_bit));
            auto mask_var = Var<UnsignedBits>(mask);
            auto lhs_bits = (Traits<KeyT>::TwiddleIn(compute::as<UnsignedBits>(lhs)) >> shift) & mask_var;
            auto rhs_bits = (Traits<KeyT>::TwiddleIn(compute::as<UnsignedBits>(rhs)) >> shift) & mask_var;
            if constexpr(DESCENDING)
            {
                return lhs_bits > rhs_bits;
            }
            else
            {
                return lhs_bits < rhs_bits;
            }
        };
    }
};

}  // namespace luisa::parallel_primitive
