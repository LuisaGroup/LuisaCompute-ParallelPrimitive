/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceMergeSort : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u);

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_key_merge_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_pair_merge_shaders;

  public:
    DeviceMergeSort()  = default;
    ~DeviceMergeSort() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <NumericT KeyT>
    [[nodiscard]] static size_t GetSortKeysTempStorageBytes(size_t num_items)
    {
        validate_storage_count<KeyT>(num_items);
        if(num_items <= 1u)
        {
            return 0u;
        }
        return rounded_storage_bytes<KeyT>(num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    [[nodiscard]] static size_t GetSortPairsTempStorageBytes(size_t num_items)
    {
        validate_storage_count<KeyT>(num_items);
        validate_storage_count<ValueT>(num_items);
        if(num_items <= 1u)
        {
            return 0u;
        }
        auto key_bytes    = rounded_storage_bytes<KeyT>(num_items);
        auto value_offset = checked_align_up(key_bytes, alignof(ValueT));
        auto value_bytes  = rounded_storage_bytes<ValueT>(num_items);
        LUISA_ASSERT(value_offset <= std::numeric_limits<size_t>::max() - value_bytes,
                     "DeviceMergeSort temporary storage size overflow.");
        return value_offset + value_bytes;
    }

    template <NumericT KeyT>
    void SortKeys(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<KeyT> keys, size_t num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortKeys(cmdlist, temp_storage, keys, num_items, compare);
    }

    template <NumericT KeyT>
    void SortKeys(CommandList&                      cmdlist,
                  BufferView<uint>                  temp_storage,
                  BufferView<KeyT>                  keys,
                  size_t                            num_items,
                  const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_in_place_key_view(temp_storage, keys, num_items);
        if(num_items <= 1u)
        {
            return;
        }

        auto scratch        = key_scratch<KeyT>(temp_storage, num_items);
        auto source         = keys.subview(0u, num_items);
        bool source_is_keys = true;
        for(size_t width = 1u; width < num_items;)
        {
            auto destination = source_is_keys ? scratch : keys.subview(0u, num_items);
            dispatch_key_merge(cmdlist, source, destination, num_items, width, compare);
            source         = destination;
            source_is_keys = !source_is_keys;
            if(width > num_items / 2u)
            {
                break;
            }
            width *= 2u;
        }
        if(!source_is_keys)
        {
            cmdlist << keys.subview(0u, num_items).copy_from(source);
        }
    }

    template <NumericT KeyT>
    void SortKeys(CommandList&     cmdlist,
                  BufferView<uint> temp_storage,
                  BufferView<KeyT> keys_input,
                  BufferView<KeyT> keys_output,
                  size_t           num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortKeys(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT>
    void SortKeys(CommandList&                      cmdlist,
                  BufferView<uint>                  temp_storage,
                  BufferView<KeyT>                  keys_input,
                  BufferView<KeyT>                  keys_output,
                  size_t                            num_items,
                  const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_key_views(temp_storage, keys_input, keys_output, num_items);
        if(num_items == 0u)
        {
            return;
        }
        if(num_items == 1u)
        {
            cmdlist << keys_output.subview(0u, 1u).copy_from(keys_input.subview(0u, 1u));
            return;
        }

        auto scratch      = key_scratch<KeyT>(temp_storage, num_items);
        auto passes       = merge_pass_count(num_items);
        auto source       = keys_input.subview(0u, num_items);
        bool write_output = (passes & 1u) != 0u;
        for(size_t width = 1u; width < num_items;)
        {
            auto destination = write_output ? keys_output.subview(0u, num_items) : scratch;
            dispatch_key_merge(cmdlist, source, destination, num_items, width, compare);
            source       = destination;
            write_output = !write_output;
            if(width > num_items / 2u)
            {
                break;
            }
            width *= 2u;
        }
    }

    template <NumericT KeyT>
    void StableSortKeys(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<KeyT> keys, size_t num_items)
    {
        SortKeys(cmdlist, temp_storage, keys, num_items);
    }

    template <NumericT KeyT>
    void StableSortKeys(CommandList&                      cmdlist,
                        BufferView<uint>                  temp_storage,
                        BufferView<KeyT>                  keys,
                        size_t                            num_items,
                        const Callable<bool(KeyT, KeyT)>& compare)
    {
        SortKeys(cmdlist, temp_storage, keys, num_items, compare);
    }

    template <NumericT KeyT>
    void StableSortKeys(CommandList&     cmdlist,
                        BufferView<uint> temp_storage,
                        BufferView<KeyT> keys_input,
                        BufferView<KeyT> keys_output,
                        size_t           num_items)
    {
        SortKeys(cmdlist, temp_storage, keys_input, keys_output, num_items);
    }

    template <NumericT KeyT>
    void StableSortKeys(CommandList&                      cmdlist,
                        BufferView<uint>                  temp_storage,
                        BufferView<KeyT>                  keys_input,
                        BufferView<KeyT>                  keys_output,
                        size_t                            num_items,
                        const Callable<bool(KeyT, KeyT)>& compare)
    {
        SortKeys(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&       cmdlist,
                   BufferView<uint>   temp_storage,
                   BufferView<KeyT>   keys,
                   BufferView<ValueT> values,
                   size_t             num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortPairs(cmdlist, temp_storage, keys, values, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&                      cmdlist,
                   BufferView<uint>                  temp_storage,
                   BufferView<KeyT>                  keys,
                   BufferView<ValueT>                values,
                   size_t                            num_items,
                   const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_in_place_pair_views(temp_storage, keys, values, num_items);
        if(num_items <= 1u)
        {
            return;
        }

        auto [scratch_keys, scratch_values] = pair_scratch<KeyT, ValueT>(temp_storage, num_items);
        auto source_keys                    = keys.subview(0u, num_items);
        auto source_values                  = values.subview(0u, num_items);
        bool source_is_input                = true;
        for(size_t width = 1u; width < num_items;)
        {
            auto destination_keys = source_is_input ? scratch_keys : keys.subview(0u, num_items);
            auto destination_values = source_is_input ? scratch_values : values.subview(0u, num_items);
            dispatch_pair_merge(
                cmdlist, source_keys, source_values, destination_keys, destination_values, num_items, width, compare);
            source_keys     = destination_keys;
            source_values   = destination_values;
            source_is_input = !source_is_input;
            if(width > num_items / 2u)
            {
                break;
            }
            width *= 2u;
        }
        if(!source_is_input)
        {
            cmdlist << keys.subview(0u, num_items).copy_from(source_keys)
                    << values.subview(0u, num_items).copy_from(source_values);
        }
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&       cmdlist,
                   BufferView<uint>   temp_storage,
                   BufferView<KeyT>   keys_input,
                   BufferView<KeyT>   keys_output,
                   BufferView<ValueT> values_input,
                   BufferView<ValueT> values_output,
                   size_t             num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortPairs(cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairs(CommandList&                      cmdlist,
                   BufferView<uint>                  temp_storage,
                   BufferView<KeyT>                  keys_input,
                   BufferView<KeyT>                  keys_output,
                   BufferView<ValueT>                values_input,
                   BufferView<ValueT>                values_output,
                   size_t                            num_items,
                   const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_pair_views(temp_storage, keys_input, keys_output, values_input, values_output, num_items);
        if(num_items == 0u)
        {
            return;
        }
        if(num_items == 1u)
        {
            cmdlist << keys_output.subview(0u, 1u).copy_from(keys_input.subview(0u, 1u))
                    << values_output.subview(0u, 1u).copy_from(values_input.subview(0u, 1u));
            return;
        }

        auto [scratch_keys, scratch_values] = pair_scratch<KeyT, ValueT>(temp_storage, num_items);
        auto passes                         = merge_pass_count(num_items);
        auto source_keys                    = keys_input.subview(0u, num_items);
        auto source_values                  = values_input.subview(0u, num_items);
        bool write_output                   = (passes & 1u) != 0u;
        for(size_t width = 1u; width < num_items;)
        {
            auto destination_keys = write_output ? keys_output.subview(0u, num_items) : scratch_keys;
            auto destination_values = write_output ? values_output.subview(0u, num_items) : scratch_values;
            dispatch_pair_merge(
                cmdlist, source_keys, source_values, destination_keys, destination_values, num_items, width, compare);
            source_keys   = destination_keys;
            source_values = destination_values;
            write_output  = !write_output;
            if(width > num_items / 2u)
            {
                break;
            }
            width *= 2u;
        }
    }

    template <NumericT KeyT, NumericT ValueT>
    void StableSortPairs(CommandList&       cmdlist,
                         BufferView<uint>   temp_storage,
                         BufferView<KeyT>   keys,
                         BufferView<ValueT> values,
                         size_t             num_items)
    {
        SortPairs(cmdlist, temp_storage, keys, values, num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    void StableSortPairs(CommandList&                      cmdlist,
                         BufferView<uint>                  temp_storage,
                         BufferView<KeyT>                  keys,
                         BufferView<ValueT>                values,
                         size_t                            num_items,
                         const Callable<bool(KeyT, KeyT)>& compare)
    {
        SortPairs(cmdlist, temp_storage, keys, values, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void StableSortPairs(CommandList&       cmdlist,
                         BufferView<uint>   temp_storage,
                         BufferView<KeyT>   keys_input,
                         BufferView<KeyT>   keys_output,
                         BufferView<ValueT> values_input,
                         BufferView<ValueT> values_output,
                         size_t             num_items)
    {
        SortPairs(cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    void StableSortPairs(CommandList&                      cmdlist,
                         BufferView<uint>                  temp_storage,
                         BufferView<KeyT>                  keys_input,
                         BufferView<KeyT>                  keys_output,
                         BufferView<ValueT>                values_input,
                         BufferView<ValueT>                values_output,
                         size_t                            num_items,
                         const Callable<bool(KeyT, KeyT)>& compare)
    {
        SortPairs(cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, compare);
    }

    template <NumericT KeyT>
    void SortKeysCopy(CommandList&     cmdlist,
                      BufferView<uint> temp_storage,
                      BufferView<KeyT> keys_input,
                      BufferView<KeyT> keys_output,
                      size_t           num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortKeysCopy(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT>
    void SortKeysCopy(CommandList&                      cmdlist,
                      BufferView<uint>                  temp_storage,
                      BufferView<KeyT>                  keys_input,
                      BufferView<KeyT>                  keys_output,
                      size_t                            num_items,
                      const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_copy_range(keys_input, keys_output, num_items, "key");
        SortKeys(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT>
    void StableSortKeysCopy(CommandList&     cmdlist,
                            BufferView<uint> temp_storage,
                            BufferView<KeyT> keys_input,
                            BufferView<KeyT> keys_output,
                            size_t           num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        StableSortKeysCopy(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT>
    void StableSortKeysCopy(CommandList&                      cmdlist,
                            BufferView<uint>                  temp_storage,
                            BufferView<KeyT>                  keys_input,
                            BufferView<KeyT>                  keys_output,
                            size_t                            num_items,
                            const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_copy_range(keys_input, keys_output, num_items, "key");
        StableSortKeys(cmdlist, temp_storage, keys_input, keys_output, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairsCopy(CommandList&       cmdlist,
                       BufferView<uint>   temp_storage,
                       BufferView<KeyT>   keys_input,
                       BufferView<KeyT>   keys_output,
                       BufferView<ValueT> values_input,
                       BufferView<ValueT> values_output,
                       size_t             num_items)
    {
        Callable<bool(KeyT, KeyT)> compare = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs < rhs; };
        SortPairsCopy(cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, compare);
    }

    template <NumericT KeyT, NumericT ValueT>
    void SortPairsCopy(CommandList&                      cmdlist,
                       BufferView<uint>                  temp_storage,
                       BufferView<KeyT>                  keys_input,
                       BufferView<KeyT>                  keys_output,
                       BufferView<ValueT>                values_input,
                       BufferView<ValueT>                values_output,
                       size_t                            num_items,
                       const Callable<bool(KeyT, KeyT)>& compare)
    {
        validate_copy_range(keys_input, keys_output, num_items, "key");
        validate_copy_range(keys_input, values_output, num_items, "input-key/output-value");
        validate_copy_range(values_input, keys_output, num_items, "input-value/output-key");
        validate_copy_range(values_input, values_output, num_items, "value");
        SortPairs(cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, compare);
    }

  private:
    template <NumericT T>
    static void validate_storage_count(size_t num_items)
    {
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceMergeSort.");
        LUISA_ASSERT(num_items <= std::numeric_limits<size_t>::max() / sizeof(T),
                     "DeviceMergeSort temporary storage size overflow.");
    }

    template <NumericT T>
    [[nodiscard]] static size_t rounded_storage_bytes(size_t num_items)
    {
        auto raw_bytes = num_items * sizeof(T);
        LUISA_ASSERT(raw_bytes <= std::numeric_limits<size_t>::max() - (sizeof(uint) - 1u),
                     "DeviceMergeSort temporary storage size overflow.");
        return bytes_to_uint_count(raw_bytes) * sizeof(uint);
    }

    [[nodiscard]] static size_t checked_align_up(size_t byte_offset, size_t alignment)
    {
        LUISA_ASSERT(byte_offset <= std::numeric_limits<size_t>::max() - (alignment - 1u),
                     "DeviceMergeSort temporary storage size overflow.");
        return align_up_uint(byte_offset, alignment);
    }

    [[nodiscard]] static size_t merge_pass_count(size_t num_items) noexcept
    {
        size_t passes = 0u;
        for(size_t width = 1u; width < num_items;)
        {
            ++passes;
            if(width > num_items / 2u)
            {
                break;
            }
            width *= 2u;
        }
        return passes;
    }

    template <NumericT KeyT>
    [[nodiscard]] static BufferView<KeyT> key_scratch(BufferView<uint> temp_storage, size_t num_items)
    {
        auto key_uints = bytes_to_uint_count(num_items * sizeof(KeyT));
        return temp_storage.subview(0u, key_uints).template as<KeyT>().subview(0u, num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    [[nodiscard]] static std::pair<BufferView<KeyT>, BufferView<ValueT>> pair_scratch(BufferView<uint> temp_storage,
                                                                                      size_t num_items)
    {
        auto key_uints = bytes_to_uint_count(num_items * sizeof(KeyT));
        auto scratch_keys = temp_storage.subview(0u, key_uints).template as<KeyT>().subview(0u, num_items);
        size_t value_offset_bytes = checked_align_up(key_uints * sizeof(uint), alignof(ValueT));
        auto   value_uints        = bytes_to_uint_count(num_items * sizeof(ValueT));
        auto   scratch_values = temp_storage.subview(value_offset_bytes / sizeof(uint), value_uints)
                                  .template as<ValueT>()
                                  .subview(0u, num_items);
        return {scratch_keys, scratch_values};
    }

    template <NumericT KeyT>
    void validate_in_place_key_view(BufferView<uint> temp_storage, BufferView<KeyT> keys, size_t num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceMergeSort::create must be called before dispatch.");
        validate_storage_count<KeyT>(num_items);
        LUISA_ASSERT(keys.size() >= num_items, "DeviceMergeSort key view is too small.");
        if(num_items > 1u)
        {
            LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetSortKeysTempStorageBytes<KeyT>(num_items)),
                         "DeviceMergeSort temporary storage is too small.");
        }
    }

    template <NumericT KeyT, NumericT ValueT>
    void validate_in_place_pair_views(BufferView<uint>   temp_storage,
                                      BufferView<KeyT>   keys,
                                      BufferView<ValueT> values,
                                      size_t             num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceMergeSort::create must be called before dispatch.");
        validate_storage_count<KeyT>(num_items);
        validate_storage_count<ValueT>(num_items);
        LUISA_ASSERT(keys.size() >= num_items, "DeviceMergeSort key view is too small.");
        LUISA_ASSERT(values.size() >= num_items, "DeviceMergeSort value view is too small.");
        if(num_items > 1u)
        {
            LUISA_ASSERT(temp_storage.size()
                             >= bytes_to_uint_count(GetSortPairsTempStorageBytes<KeyT, ValueT>(num_items)),
                         "DeviceMergeSort temporary storage is too small.");
        }
    }

    template <typename InputT, typename OutputT>
    static void validate_copy_range(BufferView<InputT> input, BufferView<OutputT> output, size_t num_items, luisa::string_view name)
    {
        if(num_items == 0u || input.handle() != output.handle())
        {
            return;
        }
        auto input_begin  = input.offset_bytes();
        auto input_end    = input_begin + num_items * input.stride();
        auto output_begin = output.offset_bytes();
        auto output_end   = output_begin + num_items * output.stride();
        LUISA_ASSERT(input_end <= output_begin || output_end <= input_begin,
                     "DeviceMergeSort Copy {} input and output ranges must not overlap.",
                     name);
    }

    template <NumericT KeyT>
    void validate_key_views(BufferView<uint> temp_storage,
                            BufferView<KeyT> keys_input,
                            BufferView<KeyT> keys_output,
                            size_t           num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceMergeSort::create must be called before dispatch.");
        validate_storage_count<KeyT>(num_items);
        LUISA_ASSERT(keys_input.size() >= num_items, "DeviceMergeSort key input view is too small.");
        LUISA_ASSERT(keys_output.size() >= num_items, "DeviceMergeSort key output view is too small.");
        if(num_items > 1u)
        {
            LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetSortKeysTempStorageBytes<KeyT>(num_items)),
                         "DeviceMergeSort temporary storage is too small.");
        }
    }

    template <NumericT KeyT, NumericT ValueT>
    void validate_pair_views(BufferView<uint>   temp_storage,
                             BufferView<KeyT>   keys_input,
                             BufferView<KeyT>   keys_output,
                             BufferView<ValueT> values_input,
                             BufferView<ValueT> values_output,
                             size_t             num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceMergeSort::create must be called before dispatch.");
        validate_storage_count<KeyT>(num_items);
        validate_storage_count<ValueT>(num_items);
        LUISA_ASSERT(keys_input.size() >= num_items, "DeviceMergeSort key input view is too small.");
        LUISA_ASSERT(keys_output.size() >= num_items, "DeviceMergeSort key output view is too small.");
        LUISA_ASSERT(values_input.size() >= num_items, "DeviceMergeSort value input view is too small.");
        LUISA_ASSERT(values_output.size() >= num_items, "DeviceMergeSort value output view is too small.");
        if(num_items > 1u)
        {
            LUISA_ASSERT(temp_storage.size()
                             >= bytes_to_uint_count(GetSortPairsTempStorageBytes<KeyT, ValueT>(num_items)),
                         "DeviceMergeSort temporary storage is too small.");
        }
    }

    template <typename... T>
    [[nodiscard]] static luisa::string shader_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    template <NumericT KeyT>
    void dispatch_key_merge(CommandList&                      cmdlist,
                            BufferView<KeyT>                  source,
                            BufferView<KeyT>                  destination,
                            size_t                            num_items,
                            size_t                            width,
                            const Callable<bool(KeyT, KeyT)>& compare)
    {
        using MergeShader = Shader<1, Buffer<KeyT>, Buffer<KeyT>, uint, uint>;
        auto key          = shader_key<KeyT>("device_merge_sort_keys");
        key += luisa::format("+{}", compare.function().hash());
        auto shader_it = m_key_merge_shaders.find(key);
        if(shader_it == m_key_merge_shaders.end())
        {
            U<MergeShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [compare](BufferVar<KeyT> source_buffer, BufferVar<KeyT> destination_buffer, UInt num_items_var, UInt width_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt index       = dispatch_id().x;
                    UInt run         = index / width_var;
                    UInt left_begin  = (run - run % 2u) * width_var;
                    UInt right_begin = min(left_begin + width_var, num_items_var);
                    UInt right_end   = right_begin + min(width_var, num_items_var - right_begin);
                    UInt left_count  = right_begin - left_begin;
                    UInt right_count = right_end - right_begin;
                    UInt diagonal    = index - left_begin;
                    UInt low         = select(0u, diagonal - right_count, diagonal > right_count);
                    UInt high        = min(diagonal, left_count);
                    $while(low < high)
                    {
                        UInt left_taken     = (low + high) / 2u;
                        UInt right_taken    = diagonal - left_taken;
                        Bool take_more_left = false;
                        $if(right_taken > 0u & left_taken < left_count)
                        {
                            take_more_left = !compare(source_buffer.read(right_begin + right_taken - 1u),
                                                      source_buffer.read(left_begin + left_taken));
                        };
                        $if(take_more_left)
                        {
                            low = left_taken + 1u;
                        }
                        $else
                        {
                            high = left_taken;
                        };
                    };

                    UInt left_index  = low;
                    UInt right_index = diagonal - left_index;
                    Bool take_left   = left_index < left_count;
                    $if(take_left & right_index < right_count)
                    {
                        take_left = !compare(source_buffer.read(right_begin + right_index),
                                             source_buffer.read(left_begin + left_index));
                    };
                    UInt source_index = right_begin + right_index;
                    $if(take_left)
                    {
                        source_index = left_begin + left_index;
                    };
                    destination_buffer.write(index, source_buffer.read(source_index));
                });
            m_key_merge_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_merge_shaders.find(key);
        }
        auto shader = reinterpret_cast<MergeShader*>(&(*shader_it->second));
        cmdlist << (*shader)(source, destination, static_cast<uint>(num_items), static_cast<uint>(width))
                       .dispatch(num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    void dispatch_pair_merge(CommandList&                      cmdlist,
                             BufferView<KeyT>                  source_keys,
                             BufferView<ValueT>                source_values,
                             BufferView<KeyT>                  destination_keys,
                             BufferView<ValueT>                destination_values,
                             size_t                            num_items,
                             size_t                            width,
                             const Callable<bool(KeyT, KeyT)>& compare)
    {
        using MergeShader = Shader<1, Buffer<KeyT>, Buffer<ValueT>, Buffer<KeyT>, Buffer<ValueT>, uint, uint>;
        auto key = shader_key<KeyT, ValueT>("device_merge_sort_pairs");
        key += luisa::format("+{}", compare.function().hash());
        auto shader_it = m_pair_merge_shaders.find(key);
        if(shader_it == m_pair_merge_shaders.end())
        {
            U<MergeShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [compare](BufferVar<KeyT>   source_keys_buffer,
                                   BufferVar<ValueT> source_values_buffer,
                                   BufferVar<KeyT>   destination_keys_buffer,
                                   BufferVar<ValueT> destination_values_buffer,
                                   UInt              num_items_var,
                                   UInt              width_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index       = dispatch_id().x;
                             UInt run         = index / width_var;
                             UInt left_begin  = (run - run % 2u) * width_var;
                             UInt right_begin = min(left_begin + width_var, num_items_var);
                             UInt right_end = right_begin + min(width_var, num_items_var - right_begin);
                             UInt left_count  = right_begin - left_begin;
                             UInt right_count = right_end - right_begin;
                             UInt diagonal    = index - left_begin;
                             UInt low  = select(0u, diagonal - right_count, diagonal > right_count);
                             UInt high = min(diagonal, left_count);
                             $while(low < high)
                             {
                                 UInt left_taken     = (low + high) / 2u;
                                 UInt right_taken    = diagonal - left_taken;
                                 Bool take_more_left = false;
                                 $if(right_taken > 0u & left_taken < left_count)
                                 {
                                     take_more_left =
                                         !compare(source_keys_buffer.read(right_begin + right_taken - 1u),
                                                  source_keys_buffer.read(left_begin + left_taken));
                                 };
                                 $if(take_more_left)
                                 {
                                     low = left_taken + 1u;
                                 }
                                 $else
                                 {
                                     high = left_taken;
                                 };
                             };

                             UInt left_index  = low;
                             UInt right_index = diagonal - left_index;
                             Bool take_left   = left_index < left_count;
                             $if(take_left & right_index < right_count)
                             {
                                 take_left = !compare(source_keys_buffer.read(right_begin + right_index),
                                                      source_keys_buffer.read(left_begin + left_index));
                             };
                             UInt source_index = right_begin + right_index;
                             $if(take_left)
                             {
                                 source_index = left_begin + left_index;
                             };
                             destination_keys_buffer.write(index, source_keys_buffer.read(source_index));
                             destination_values_buffer.write(index, source_values_buffer.read(source_index));
                         });
            m_pair_merge_shaders.try_emplace(key, std::move(shader));
            shader_it = m_pair_merge_shaders.find(key);
        }
        auto shader = reinterpret_cast<MergeShader*>(&(*shader_it->second));
        cmdlist << (*shader)(source_keys,
                             source_values,
                             destination_keys,
                             destination_values,
                             static_cast<uint>(num_items),
                             static_cast<uint>(width))
                       .dispatch(num_items);
    }
};

}  // namespace luisa::parallel_primitive
