/*
 * @Author: Ligo
 * @Date: 2026-10-12
 */
#pragma once

#include <cstddef>
#include <limits>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <utility>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

/// Portable flat-buffer form of CCCL/CUB v3.5's DeviceBatchedTopK.
///
/// Each input segment is `[begin_offsets[i], end_offsets[i])`. The selected
/// values for segment `i` are written to `[i * k, (i + 1) * k)` in the output
/// buffer; only `min(k, segment_size)` entries are written. The output order
/// and tied subset are unspecified by the public contract, even though this
/// implementation currently resolves ties by input position.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceBatchedTopK : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceBatchedTopK block size must be positive.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_key_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_pair_shaders;

  public:
    DeviceBatchedTopK()  = default;
    ~DeviceBatchedTopK() = default;

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

    template <NumericT KeyT>
    void MinKeys(CommandList&     cmdlist,
                 BufferView<uint> temp_storage,
                 BufferView<KeyT> keys_input,
                 BufferView<KeyT> keys_output,
                 BufferView<uint> begin_offsets,
                 BufferView<uint> end_offsets,
                 size_t           num_segments,
                 size_t           k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, begin_offsets, end_offsets, num_segments, k, better, "min");
    }

    template <NumericT KeyT>
    void MaxKeys(CommandList&     cmdlist,
                 BufferView<uint> temp_storage,
                 BufferView<KeyT> keys_input,
                 BufferView<KeyT> keys_output,
                 BufferView<uint> begin_offsets,
                 BufferView<uint> end_offsets,
                 size_t           num_segments,
                 size_t           k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs > rhs; };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, begin_offsets, end_offsets, num_segments, k, better, "max");
    }

    template <NumericT KeyT, typename ValueT>
    void MinPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  BufferView<uint>   begin_offsets,
                  BufferView<uint>   end_offsets,
                  size_t             num_segments,
                  size_t             k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        dispatch_pairs(cmdlist,
                       temp_storage,
                       keys_input,
                       keys_output,
                       values_input,
                       values_output,
                       begin_offsets,
                       end_offsets,
                       num_segments,
                       k,
                       better,
                       "min");
    }

    template <NumericT KeyT, typename ValueT>
    void MaxPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  BufferView<uint>   begin_offsets,
                  BufferView<uint>   end_offsets,
                  size_t             num_segments,
                  size_t             k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs > rhs; };
        dispatch_pairs(cmdlist,
                       temp_storage,
                       keys_input,
                       keys_output,
                       values_input,
                       values_output,
                       begin_offsets,
                       end_offsets,
                       num_segments,
                       k,
                       better,
                       "max");
    }

  private:
    [[nodiscard]] size_t validate_counts(size_t num_segments, size_t k) const
    {
        LUISA_ASSERT(m_created, "DeviceBatchedTopK::create must be called before dispatch.");
        LUISA_ASSERT(num_segments <= std::numeric_limits<uint>::max(),
                     "DeviceBatchedTopK segment count exceeds the shader index range.");
        LUISA_ASSERT(k <= std::numeric_limits<uint>::max(), "DeviceBatchedTopK k exceeds the shader index range.");
        LUISA_ASSERT(num_segments == 0u || k <= std::numeric_limits<uint>::max() / num_segments,
                     "DeviceBatchedTopK packed output size exceeds the shader index range.");
        return num_segments * k;
    }

    template <typename KeyT, typename Better>
    void dispatch_keys(CommandList&       cmdlist,
                       BufferView<uint>   temp_storage,
                       BufferView<KeyT>   keys_input,
                       BufferView<KeyT>   keys_output,
                       BufferView<uint>   begin_offsets,
                       BufferView<uint>   end_offsets,
                       size_t             num_segments,
                       size_t             k,
                       Better             better,
                       luisa::string_view direction)
    {
        static_cast<void>(temp_storage);
        auto output_size = validate_counts(num_segments, k);
        if(output_size == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_input.size() <= std::numeric_limits<uint>::max(),
                     "DeviceBatchedTopK key input exceeds the shader index range.");
        LUISA_ASSERT(keys_output.size() >= output_size, "DeviceBatchedTopK key output view is too small.");
        LUISA_ASSERT(begin_offsets.size() >= num_segments, "DeviceBatchedTopK begin-offset view is too small.");
        LUISA_ASSERT(end_offsets.size() >= num_segments, "DeviceBatchedTopK end-offset view is too small.");

        using BatchedKeysShader =
            Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<uint>, Buffer<uint>, uint, uint>;
        auto key       = luisa::format("device_batched_topk_keys+{}+{}+{}",
                                 direction,
                                 Type::of<KeyT>()->description(),
                                 get_binary_op_hash<KeyT>(better));
        auto shader_it = m_key_shaders.find(key);
        if(shader_it == m_key_shaders.end())
        {
            U<BatchedKeysShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [better](BufferVar<KeyT> keys_input_buffer,
                         BufferVar<KeyT> keys_output_buffer,
                         BufferVar<uint> begin_offsets_buffer,
                         BufferVar<uint> end_offsets_buffer,
                         UInt            num_segments_var,
                         UInt            k_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt segment = dispatch_id().x;
                    $if(segment < num_segments_var)
                    {
                        UInt input_begin = begin_offsets_buffer.read(segment);
                        UInt input_end   = end_offsets_buffer.read(segment);
                        $if(input_begin < input_end)
                        {
                            UInt segment_size = input_end - input_begin;
                            UInt output_count = select(k_var, segment_size, k_var > segment_size);
                            $for(candidate, input_begin, input_end)
                            {
                                Var<KeyT> candidate_key = keys_input_buffer.read(candidate);
                                UInt      rank          = 0u;
                                $for(other, input_begin, input_end)
                                {
                                    Var<KeyT> other_key        = keys_input_buffer.read(other);
                                    Bool      other_better     = better(other_key, candidate_key);
                                    Bool      candidate_better = better(candidate_key, other_key);
                                    Bool      equivalent       = !other_better & !candidate_better;
                                    rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                                };
                                $if(rank < output_count)
                                {
                                    keys_output_buffer.write(segment * k_var + rank, candidate_key);
                                };
                            };
                        };
                    };
                });
            m_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_shaders.find(key);
        }

        auto shader = reinterpret_cast<BatchedKeysShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceBatchedTopK key shader.");
        cmdlist << (*shader)(keys_input,
                             keys_output,
                             begin_offsets,
                             end_offsets,
                             static_cast<uint>(num_segments),
                             static_cast<uint>(k))
                       .dispatch(num_segments);
    }

    template <typename KeyT, typename ValueT, typename Better>
    void dispatch_pairs(CommandList&       cmdlist,
                        BufferView<uint>   temp_storage,
                        BufferView<KeyT>   keys_input,
                        BufferView<KeyT>   keys_output,
                        BufferView<ValueT> values_input,
                        BufferView<ValueT> values_output,
                        BufferView<uint>   begin_offsets,
                        BufferView<uint>   end_offsets,
                        size_t             num_segments,
                        size_t             k,
                        Better             better,
                        luisa::string_view direction)
    {
        static_cast<void>(temp_storage);
        auto output_size = validate_counts(num_segments, k);
        if(output_size == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_input.size() <= std::numeric_limits<uint>::max(),
                     "DeviceBatchedTopK pair key input exceeds the shader index range.");
        LUISA_ASSERT(values_input.size() >= keys_input.size(),
                     "DeviceBatchedTopK pair value input view is too small.");
        LUISA_ASSERT(keys_output.size() >= output_size, "DeviceBatchedTopK pair key output view is too small.");
        LUISA_ASSERT(values_output.size() >= output_size, "DeviceBatchedTopK pair value output view is too small.");
        LUISA_ASSERT(begin_offsets.size() >= num_segments, "DeviceBatchedTopK begin-offset view is too small.");
        LUISA_ASSERT(end_offsets.size() >= num_segments, "DeviceBatchedTopK end-offset view is too small.");

        using BatchedPairsShader =
            Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<ValueT>, Buffer<ValueT>, Buffer<uint>, Buffer<uint>, uint, uint>;
        auto key       = luisa::format("device_batched_topk_pairs+{}+{}+{}+{}",
                                 direction,
                                 Type::of<KeyT>()->description(),
                                 Type::of<ValueT>()->description(),
                                 get_binary_op_hash<KeyT>(better));
        auto shader_it = m_pair_shaders.find(key);
        if(shader_it == m_pair_shaders.end())
        {
            U<BatchedPairsShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [better](BufferVar<KeyT>   keys_input_buffer,
                         BufferVar<KeyT>   keys_output_buffer,
                         BufferVar<ValueT> values_input_buffer,
                         BufferVar<ValueT> values_output_buffer,
                         BufferVar<uint>   begin_offsets_buffer,
                         BufferVar<uint>   end_offsets_buffer,
                         UInt              num_segments_var,
                         UInt              k_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt segment = dispatch_id().x;
                    $if(segment < num_segments_var)
                    {
                        UInt input_begin = begin_offsets_buffer.read(segment);
                        UInt input_end   = end_offsets_buffer.read(segment);
                        $if(input_begin < input_end)
                        {
                            UInt segment_size = input_end - input_begin;
                            UInt output_count = select(k_var, segment_size, k_var > segment_size);
                            $for(candidate, input_begin, input_end)
                            {
                                Var<KeyT> candidate_key = keys_input_buffer.read(candidate);
                                UInt      rank          = 0u;
                                $for(other, input_begin, input_end)
                                {
                                    Var<KeyT> other_key        = keys_input_buffer.read(other);
                                    Bool      other_better     = better(other_key, candidate_key);
                                    Bool      candidate_better = better(candidate_key, other_key);
                                    Bool      equivalent       = !other_better & !candidate_better;
                                    rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                                };
                                $if(rank < output_count)
                                {
                                    UInt output_index = segment * k_var + rank;
                                    keys_output_buffer.write(output_index, candidate_key);
                                    values_output_buffer.write(output_index, values_input_buffer.read(candidate));
                                };
                            };
                        };
                    };
                });
            m_pair_shaders.try_emplace(key, std::move(shader));
            shader_it = m_pair_shaders.find(key);
        }

        auto shader = reinterpret_cast<BatchedPairsShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceBatchedTopK pair shader.");
        cmdlist << (*shader)(keys_input,
                             keys_output,
                             values_input,
                             values_output,
                             begin_offsets,
                             end_offsets,
                             static_cast<uint>(num_segments),
                             static_cast<uint>(k))
                       .dispatch(num_segments);
    }
};

}  // namespace luisa::parallel_primitive
