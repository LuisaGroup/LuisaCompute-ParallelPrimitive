/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <lcpp/block/block_store.h>
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

        if constexpr(BLOCK_SIZE % details::WARP_SIZE == 0u && BLOCK_SIZE <= 256u && sizeof(KeyT) <= 8u)
        {
            if(m_device.backend_name() == "cuda")
            {
                dispatch_tiled_keys(cmdlist, keys_input1, num_keys1, keys_input2, num_keys2, keys_output, compare);
                return;
            }
        }

        using MergeKeysShader = Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<KeyT>, uint, uint>;
        auto key              = luisa::format("device_merge_keys_tile4+{}+{}",
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
                             constexpr size_t items_per_thread = details::ITEMS_PER_THREAD;
                             UInt             group = block_id().x * block_size_x() + thread_id().x;
                             UInt             diagonal = group * UInt(items_per_thread);
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
                             UInt total_items = num_keys1_var + num_keys2_var;
                             for(size_t item = 0u; item < items_per_thread; ++item)
                             {
                                 UInt output_index = diagonal + UInt(item);
                                 $if(output_index < total_items)
                                 {
                                     Bool take_left = left_index < num_keys1_var;
                                     $if(take_left & right_index < num_keys2_var)
                                     {
                                         take_left = !compare(keys_input2_buffer.read(right_index),
                                                              keys_input1_buffer.read(left_index));
                                     };
                                     $if(take_left)
                                     {
                                         keys_output_buffer.write(output_index,
                                                                  keys_input1_buffer.read(left_index));
                                         left_index += 1u;
                                     }
                                     $else
                                     {
                                         keys_output_buffer.write(output_index,
                                                                  keys_input2_buffer.read(right_index));
                                         right_index += 1u;
                                     };
                                 };
                             }
                         });
            m_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_shaders.find(key);
        }

        auto shader = reinterpret_cast<MergeKeysShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceMerge::MergeKeys shader.");
        size_t dispatch_threads = total_items / details::ITEMS_PER_THREAD
                                  + static_cast<size_t>(total_items % details::ITEMS_PER_THREAD != 0u);
        cmdlist << (*shader)(keys_input1, keys_input2, keys_output, static_cast<uint>(num_keys1), static_cast<uint>(num_keys2))
                       .dispatch(dispatch_threads);
    }

  private:
    template <typename Compare, typename ReadLeft, typename ReadRight>
    static UInt merge_path(UInt diagonal, UInt left_count, UInt right_count, Compare compare, ReadLeft read_left, ReadRight read_right)
    {
        UInt low  = select(0u, diagonal - right_count, diagonal > right_count);
        UInt high = min(diagonal, left_count);
        $while(low < high)
        {
            UInt left           = low + (high - low) / 2u;
            UInt right          = diagonal - left;
            Bool take_more_left = false;
            $if(right > 0u & left < left_count)
            {
                take_more_left = !compare(read_right(right - 1u), read_left(left));
            };
            $if(take_more_left)
            {
                low = left + 1u;
            }
            $else
            {
                high = left;
            };
        };
        return low;
    }

    template <typename KeyT, typename Compare>
    void dispatch_tiled_keys(CommandList&     cmdlist,
                             BufferView<KeyT> input1,
                             size_t           count1,
                             BufferView<KeyT> input2,
                             size_t           count2,
                             BufferView<KeyT> output,
                             Compare          compare)
    {
        constexpr size_t items_per_thread = 8u;
        constexpr size_t tile_items       = BLOCK_SIZE * items_per_thread;
        using Kernel   = Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<KeyT>, uint, uint>;
        auto key       = luisa::format("device_merge_keys_shared+{}+{}",
                                 Type::of<KeyT>()->description(),
                                 get_binary_op_hash<KeyT>(compare));
        auto shader_it = m_key_shaders.find(key);
        if(shader_it == m_key_shaders.end())
        {
            U<Kernel> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [compare](BufferVar<KeyT> left_buffer,
                          BufferVar<KeyT> right_buffer,
                          BufferVar<KeyT> output_buffer,
                          UInt            left_count,
                          UInt            right_count) noexcept
                {
                    constexpr size_t items_per_thread = 8u;
                    constexpr size_t tile_items       = BLOCK_SIZE * items_per_thread;
                    set_block_size(BLOCK_SIZE);
                    UInt tile_start  = block_id().x * UInt(tile_items);
                    UInt valid_items = min(UInt(tile_items), left_count + right_count - tile_start);
                    Shared<uint> partition{2u};
                    $if(thread_id().x < 2u)
                    {
                        auto read_left  = [&left_buffer](UInt i) { return left_buffer.read(i); };
                        auto read_right = [&right_buffer](UInt i) { return right_buffer.read(i); };
                        UInt diagonal   = tile_start + select(0u, valid_items, thread_id().x == 1u);
                        partition.write(thread_id().x,
                                        merge_path(diagonal, left_count, right_count, compare, read_left, read_right));
                    };
                    sync_block();
                    UInt         left_begin  = partition.read(0u);
                    UInt         right_begin = tile_start - left_begin;
                    UInt         left_items  = partition.read(1u) - left_begin;
                    UInt         right_items = valid_items - left_items;
                    Shared<KeyT> tile_keys{tile_items};
                    for(size_t item = 0u; item < items_per_thread; ++item)
                    {
                        UInt local = thread_id().x + UInt(item * BLOCK_SIZE);
                        $if(local < left_items)
                        {
                            tile_keys.write(local, left_buffer.read(left_begin + local));
                        };
                        $if(local < right_items)
                        {
                            tile_keys.write(left_items + local, right_buffer.read(right_begin + local));
                        };
                    }
                    sync_block();

                    ArrayVar<KeyT, items_per_thread> values;
                    for(size_t item = 0u; item < items_per_thread; ++item)
                    {
                        values[item] = KeyT{};
                    }
                    UInt diagonal = thread_id().x * UInt(items_per_thread);
                    $if(diagonal < valid_items)
                    {
                        auto read_left  = [&tile_keys](UInt i) { return tile_keys.read(i); };
                        auto read_right = [&tile_keys, &left_items](UInt i)
                        { return tile_keys.read(left_items + i); };
                        UInt left = merge_path(diagonal, left_items, right_items, compare, read_left, read_right);
                        UInt right = diagonal - left;
                        for(size_t item = 0u; item < items_per_thread; ++item)
                        {
                            $if(diagonal + UInt(item) < valid_items)
                            {
                                Bool take_left = left < left_items;
                                $if(take_left & right < right_items)
                                {
                                    take_left = !compare(read_right(right), read_left(left));
                                };
                                UInt source  = select(left_items + right, left, take_left);
                                values[item] = tile_keys.read(source);
                                left += select(0u, 1u, take_left);
                                right += select(1u, 0u, take_left);
                            };
                        }
                    };
                    BlockStore<KeyT, BLOCK_SIZE, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE, true> store;
                    store.Store(values, output_buffer, tile_start, valid_items);
                });
            m_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_shaders.find(key);
        }
        auto shader = reinterpret_cast<Kernel*>(&(*shader_it->second));
        cmdlist << (*shader)(input1, input2, output, static_cast<uint>(count1), static_cast<uint>(count2))
                       .dispatch(ceil_div(count1 + count2, tile_items) * BLOCK_SIZE);
    }

  public:
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
