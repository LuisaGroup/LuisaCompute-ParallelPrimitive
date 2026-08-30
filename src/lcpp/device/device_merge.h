/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceMerge : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceMerge block size must be positive.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_key_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_pair_shaders;

  public:
    DeviceMerge()  = default;
    ~DeviceMerge() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <typename KeyT>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t) noexcept
    {
        return 0u;
    }

    template <typename KeyT, typename ValueT>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t) noexcept
    {
        return 0u;
    }

    template <typename KeyT>
    void MergeKeys(CommandList&     cmdlist,
                   BufferView<uint> temp_storage,
                   BufferView<KeyT> keys_input1,
                   size_t           num_keys1,
                   BufferView<KeyT> keys_input2,
                   size_t           num_keys2,
                   BufferView<KeyT> keys_output)
    {
        auto compare = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        MergeKeys(cmdlist, temp_storage, keys_input1, num_keys1, keys_input2, num_keys2, keys_output, compare);
    }

    template <typename KeyT, typename Compare>
    void MergeKeys(CommandList& cmdlist,
                   BufferView<uint>,
                   BufferView<KeyT> keys_input1,
                   size_t           num_keys1,
                   BufferView<KeyT> keys_input2,
                   size_t           num_keys2,
                   BufferView<KeyT> keys_output,
                   Compare          compare)
    {
        size_t total_items = validate_counts(num_keys1, num_keys2);
        if(total_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_output.size() >= total_items, "DeviceMerge key output view is too small.");
        if(num_keys1 == 0u)
        {
            LUISA_ASSERT(keys_input2.size() >= num_keys2, "DeviceMerge second key input view is too small.");
            cmdlist << keys_output.subview(0u, num_keys2).copy_from(keys_input2.subview(0u, num_keys2));
            return;
        }
        if(num_keys2 == 0u)
        {
            LUISA_ASSERT(keys_input1.size() >= num_keys1, "DeviceMerge first key input view is too small.");
            cmdlist << keys_output.subview(0u, num_keys1).copy_from(keys_input1.subview(0u, num_keys1));
            return;
        }
        LUISA_ASSERT(keys_input1.size() >= num_keys1, "DeviceMerge first key input view is too small.");
        LUISA_ASSERT(keys_input2.size() >= num_keys2, "DeviceMerge second key input view is too small.");

        using MergeKeysShader = Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<KeyT>, uint, uint>;
        auto key              = luisa::format("device_merge_keys+{}+{}",
                                 Type::of<KeyT>()->description(),
                                 get_binary_op_hash<KeyT>(compare));
        auto shader_it        = m_key_shaders.find(key);
        if(shader_it == m_key_shaders.end())
        {
            U<MergeKeysShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [compare](BufferVar<KeyT> keys_input1_buffer,
                                   BufferVar<KeyT> keys_input2_buffer,
                                   BufferVar<KeyT> keys_output_buffer,
                                   UInt            num_keys1_var,
                                   UInt            num_keys2_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt diagonal = dispatch_id().x;
                             UInt low = select(0u, diagonal - num_keys2_var, diagonal > num_keys2_var);
                             UInt high = min(diagonal, num_keys1_var);
                             $while(low < high)
                             {
                                 UInt left_taken     = (low + high) / 2u;
                                 UInt right_taken    = diagonal - left_taken;
                                 Bool take_more_left = false;
                                 $if(right_taken > 0u & left_taken < num_keys1_var)
                                 {
                                     take_more_left = !compare(keys_input2_buffer.read(right_taken - 1u),
                                                               keys_input1_buffer.read(left_taken));
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
                             Bool take_left   = left_index < num_keys1_var;
                             $if(take_left & right_index < num_keys2_var)
                             {
                                 take_left = !compare(keys_input2_buffer.read(right_index),
                                                      keys_input1_buffer.read(left_index));
                             };
                             $if(take_left)
                             {
                                 keys_output_buffer.write(diagonal, keys_input1_buffer.read(left_index));
                             }
                             $else
                             {
                                 keys_output_buffer.write(diagonal, keys_input2_buffer.read(right_index));
                             };
                         });
            m_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_shaders.find(key);
        }

        auto shader = reinterpret_cast<MergeKeysShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceMerge::MergeKeys shader.");
        cmdlist << (*shader)(keys_input1, keys_input2, keys_output, static_cast<uint>(num_keys1), static_cast<uint>(num_keys2))
                       .dispatch(total_items);
    }

    template <typename KeyT, typename ValueT>
    void MergePairs(CommandList&       cmdlist,
                    BufferView<uint>   temp_storage,
                    BufferView<KeyT>   keys_input1,
                    BufferView<ValueT> values_input1,
                    size_t             num_pairs1,
                    BufferView<KeyT>   keys_input2,
                    BufferView<ValueT> values_input2,
                    size_t             num_pairs2,
                    BufferView<KeyT>   keys_output,
                    BufferView<ValueT> values_output)
    {
        auto compare = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        MergePairs(cmdlist, temp_storage, keys_input1, values_input1, num_pairs1, keys_input2, values_input2, num_pairs2, keys_output, values_output, compare);
    }

    template <typename KeyT, typename ValueT, typename Compare>
    void MergePairs(CommandList& cmdlist,
                    BufferView<uint>,
                    BufferView<KeyT>   keys_input1,
                    BufferView<ValueT> values_input1,
                    size_t             num_pairs1,
                    BufferView<KeyT>   keys_input2,
                    BufferView<ValueT> values_input2,
                    size_t             num_pairs2,
                    BufferView<KeyT>   keys_output,
                    BufferView<ValueT> values_output,
                    Compare            compare)
    {
        size_t total_items = validate_counts(num_pairs1, num_pairs2);
        if(total_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_output.size() >= total_items, "DeviceMerge pair key output view is too small.");
        LUISA_ASSERT(values_output.size() >= total_items, "DeviceMerge pair value output view is too small.");
        if(num_pairs1 == 0u)
        {
            validate_pair_input(keys_input2, values_input2, num_pairs2, "second");
            cmdlist
                << keys_output.subview(0u, num_pairs2).copy_from(keys_input2.subview(0u, num_pairs2))
                << values_output.subview(0u, num_pairs2).copy_from(values_input2.subview(0u, num_pairs2));
            return;
        }
        if(num_pairs2 == 0u)
        {
            validate_pair_input(keys_input1, values_input1, num_pairs1, "first");
            cmdlist
                << keys_output.subview(0u, num_pairs1).copy_from(keys_input1.subview(0u, num_pairs1))
                << values_output.subview(0u, num_pairs1).copy_from(values_input1.subview(0u, num_pairs1));
            return;
        }
        validate_pair_input(keys_input1, values_input1, num_pairs1, "first");
        validate_pair_input(keys_input2, values_input2, num_pairs2, "second");

        using MergePairsShader =
            Shader<1, Buffer<KeyT>, Buffer<ValueT>, Buffer<KeyT>, Buffer<ValueT>, Buffer<KeyT>, Buffer<ValueT>, uint, uint>;
        auto key       = luisa::format("device_merge_pairs+{}+{}+{}",
                                 Type::of<KeyT>()->description(),
                                 Type::of<ValueT>()->description(),
                                 get_binary_op_hash<KeyT>(compare));
        auto shader_it = m_pair_shaders.find(key);
        if(shader_it == m_pair_shaders.end())
        {
            U<MergePairsShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [compare](BufferVar<KeyT>   keys_input1_buffer,
                          BufferVar<ValueT> values_input1_buffer,
                          BufferVar<KeyT>   keys_input2_buffer,
                          BufferVar<ValueT> values_input2_buffer,
                          BufferVar<KeyT>   keys_output_buffer,
                          BufferVar<ValueT> values_output_buffer,
                          UInt              num_pairs1_var,
                          UInt              num_pairs2_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt diagonal = dispatch_id().x;
                    UInt low  = select(0u, diagonal - num_pairs2_var, diagonal > num_pairs2_var);
                    UInt high = min(diagonal, num_pairs1_var);
                    $while(low < high)
                    {
                        UInt left_taken     = (low + high) / 2u;
                        UInt right_taken    = diagonal - left_taken;
                        Bool take_more_left = false;
                        $if(right_taken > 0u & left_taken < num_pairs1_var)
                        {
                            take_more_left = !compare(keys_input2_buffer.read(right_taken - 1u),
                                                      keys_input1_buffer.read(left_taken));
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
                    Bool take_left   = left_index < num_pairs1_var;
                    $if(take_left & right_index < num_pairs2_var)
                    {
                        take_left = !compare(keys_input2_buffer.read(right_index),
                                             keys_input1_buffer.read(left_index));
                    };
                    $if(take_left)
                    {
                        keys_output_buffer.write(diagonal, keys_input1_buffer.read(left_index));
                        values_output_buffer.write(diagonal, values_input1_buffer.read(left_index));
                    }
                    $else
                    {
                        keys_output_buffer.write(diagonal, keys_input2_buffer.read(right_index));
                        values_output_buffer.write(diagonal, values_input2_buffer.read(right_index));
                    };
                });
            m_pair_shaders.try_emplace(key, std::move(shader));
            shader_it = m_pair_shaders.find(key);
        }

        auto shader = reinterpret_cast<MergePairsShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceMerge::MergePairs shader.");
        cmdlist << (*shader)(keys_input1,
                             values_input1,
                             keys_input2,
                             values_input2,
                             keys_output,
                             values_output,
                             static_cast<uint>(num_pairs1),
                             static_cast<uint>(num_pairs2))
                       .dispatch(total_items);
    }

  private:
    [[nodiscard]] size_t validate_counts(size_t first_count, size_t second_count) const
    {
        LUISA_ASSERT(m_created, "DeviceMerge::create must be called before dispatch.");
        LUISA_ASSERT(first_count <= std::numeric_limits<uint>::max(),
                     "DeviceMerge first input exceeds the shader index range.");
        LUISA_ASSERT(second_count <= std::numeric_limits<uint>::max(),
                     "DeviceMerge second input exceeds the shader index range.");
        LUISA_ASSERT(first_count <= std::numeric_limits<uint>::max() - second_count,
                     "DeviceMerge output exceeds the shader index range.");
        return first_count + second_count;
    }

    template <typename KeyT, typename ValueT>
    static void validate_pair_input(BufferView<KeyT> keys, BufferView<ValueT> values, size_t count, luisa::string_view name)
    {
        LUISA_ASSERT(keys.size() >= count, "DeviceMerge {} pair key input view is too small.", name);
        LUISA_ASSERT(values.size() >= count, "DeviceMerge {} pair value input view is too small.", name);
    }
};
}  // namespace luisa::parallel_primitive
