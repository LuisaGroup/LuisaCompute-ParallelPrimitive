/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

namespace details
{
    template <size_t INDEX = 0u, typename LhsTuple, typename RhsTuple>
    [[nodiscard]] Bool topk_tuple_less(const LhsTuple& lhs, const RhsTuple& rhs) noexcept
    {
        using Lhs = std::remove_cvref_t<LhsTuple>;
        using Rhs = std::remove_cvref_t<RhsTuple>;
        static_assert(std::tuple_size_v<Lhs> == std::tuple_size_v<Rhs>,
                      "DeviceTopK decomposer results must have matching arity.");
        static_assert(std::tuple_size_v<Lhs> > 0u, "DeviceTopK decomposer results must not be empty.");
        auto less = std::get<INDEX>(lhs) < std::get<INDEX>(rhs);
        if constexpr(INDEX + 1u == std::tuple_size_v<Lhs>)
        {
            return less;
        }
        else
        {
            auto equal = std::get<INDEX>(lhs) == std::get<INDEX>(rhs);
            return less | (equal & topk_tuple_less<INDEX + 1u>(lhs, rhs));
        }
    }
}  // namespace details

// Selects up to k best items without requiring caller-provided scratch. The
// implementation assigns each input a unique global rank and writes ranks less
// than k. Equal keys are disambiguated by input position. Although that yields
// deterministic sorted output here, callers must treat output order and the
// selected subset at a tied boundary as unspecified, matching CUB's contract.
// Input and output ranges must not overlap.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceTopK : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceTopK block size must be positive.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_key_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_pair_shaders;

  public:
    DeviceTopK()  = default;
    ~DeviceTopK() = default;

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
                 size_t           num_items,
                 size_t           k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, num_items, k, better, "min");
    }

    template <typename KeyT, typename Decomposer>
    void MinKeys(CommandList&     cmdlist,
                 BufferView<uint> temp_storage,
                 BufferView<KeyT> keys_input,
                 BufferView<KeyT> keys_output,
                 size_t           num_items,
                 size_t           k,
                 Decomposer       decomposer)
    {
        auto better = [decomposer](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept
        { return details::topk_tuple_less(decomposer(lhs), decomposer(rhs)); };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, num_items, k, better, "min_decomposed");
    }

    template <NumericT KeyT>
    void MaxKeys(CommandList&     cmdlist,
                 BufferView<uint> temp_storage,
                 BufferView<KeyT> keys_input,
                 BufferView<KeyT> keys_output,
                 size_t           num_items,
                 size_t           k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs > rhs; };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, num_items, k, better, "max");
    }

    template <typename KeyT, typename Decomposer>
    void MaxKeys(CommandList&     cmdlist,
                 BufferView<uint> temp_storage,
                 BufferView<KeyT> keys_input,
                 BufferView<KeyT> keys_output,
                 size_t           num_items,
                 size_t           k,
                 Decomposer       decomposer)
    {
        auto better = [decomposer](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept
        { return details::topk_tuple_less(decomposer(rhs), decomposer(lhs)); };
        dispatch_keys(cmdlist, temp_storage, keys_input, keys_output, num_items, k, better, "max_decomposed");
    }

    template <NumericT KeyT, typename ValueT>
    void MinPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  size_t             num_items,
                  size_t             k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs < rhs; };
        dispatch_pairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, k, better, "min");
    }

    template <typename KeyT, typename ValueT, typename Decomposer>
    void MinPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  size_t             num_items,
                  size_t             k,
                  Decomposer         decomposer)
    {
        auto better = [decomposer](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept
        { return details::topk_tuple_less(decomposer(lhs), decomposer(rhs)); };
        dispatch_pairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, k, better, "min_decomposed");
    }

    template <NumericT KeyT, typename ValueT>
    void MaxPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  size_t             num_items,
                  size_t             k)
    {
        auto better = [](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept { return lhs > rhs; };
        dispatch_pairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, k, better, "max");
    }

    template <typename KeyT, typename ValueT, typename Decomposer>
    void MaxPairs(CommandList&       cmdlist,
                  BufferView<uint>   temp_storage,
                  BufferView<KeyT>   keys_input,
                  BufferView<KeyT>   keys_output,
                  BufferView<ValueT> values_input,
                  BufferView<ValueT> values_output,
                  size_t             num_items,
                  size_t             k,
                  Decomposer         decomposer)
    {
        auto better = [decomposer](const Var<KeyT>& lhs, const Var<KeyT>& rhs) noexcept
        { return details::topk_tuple_less(decomposer(rhs), decomposer(lhs)); };
        dispatch_pairs(
            cmdlist, temp_storage, keys_input, keys_output, values_input, values_output, num_items, k, better, "max_decomposed");
    }

  private:
    [[nodiscard]] size_t validate_counts(size_t num_items, size_t k) const
    {
        LUISA_ASSERT(m_created, "DeviceTopK::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(),
                     "DeviceTopK input exceeds the shader index range.");
        return std::min(num_items, k);
    }

    template <typename KeyT, typename Better>
    void dispatch_keys(CommandList&       cmdlist,
                       BufferView<uint>   temp_storage,
                       BufferView<KeyT>   keys_input,
                       BufferView<KeyT>   keys_output,
                       size_t             num_items,
                       size_t             k,
                       Better             better,
                       luisa::string_view direction)
    {
        static_cast<void>(temp_storage);
        auto output_count = validate_counts(num_items, k);
        if(output_count == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_input.size() >= num_items, "DeviceTopK key input view is too small.");
        LUISA_ASSERT(keys_output.size() >= output_count, "DeviceTopK key output view is too small.");

        using TopKKeysShader = Shader<1, Buffer<KeyT>, Buffer<KeyT>, uint, uint>;
        auto key             = luisa::format("device_topk_keys+{}+{}+{}",
                                 direction,
                                 Type::of<KeyT>()->description(),
                                 get_binary_op_hash<KeyT>(better));
        auto shader_it       = m_key_shaders.find(key);
        if(shader_it == m_key_shaders.end())
        {
            U<TopKKeysShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [better](BufferVar<KeyT> keys_input_buffer,
                                  BufferVar<KeyT> keys_output_buffer,
                                  UInt            num_items_var,
                                  UInt            output_count_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt candidate = dispatch_id().x;
                             $if(candidate < num_items_var)
                             {
                                 Var<KeyT> candidate_key = keys_input_buffer.read(candidate);
                                 UInt      rank          = 0u;
                                 $for(other, 0u, num_items_var, 1u)
                                 {
                                     Var<KeyT> other_key        = keys_input_buffer.read(other);
                                     Bool      other_better     = better(other_key, candidate_key);
                                     Bool      candidate_better = better(candidate_key, other_key);
                                     Bool      equivalent       = !other_better & !candidate_better;
                                     rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                                 };
                                 $if(rank < output_count_var)
                                 {
                                     keys_output_buffer.write(rank, candidate_key);
                                 };
                             };
                         });
            m_key_shaders.try_emplace(key, std::move(shader));
            shader_it = m_key_shaders.find(key);
        }
        auto shader = reinterpret_cast<TopKKeysShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTopK key shader.");
        cmdlist << (*shader)(keys_input, keys_output, static_cast<uint>(num_items), static_cast<uint>(output_count))
                       .dispatch(num_items);
    }

    template <typename KeyT, typename ValueT, typename Better>
    void dispatch_pairs(CommandList&       cmdlist,
                        BufferView<uint>   temp_storage,
                        BufferView<KeyT>   keys_input,
                        BufferView<KeyT>   keys_output,
                        BufferView<ValueT> values_input,
                        BufferView<ValueT> values_output,
                        size_t             num_items,
                        size_t             k,
                        Better             better,
                        luisa::string_view direction)
    {
        static_cast<void>(temp_storage);
        auto output_count = validate_counts(num_items, k);
        if(output_count == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys_input.size() >= num_items, "DeviceTopK pair key input view is too small.");
        LUISA_ASSERT(values_input.size() >= num_items, "DeviceTopK pair value input view is too small.");
        LUISA_ASSERT(keys_output.size() >= output_count, "DeviceTopK pair key output view is too small.");
        LUISA_ASSERT(values_output.size() >= output_count, "DeviceTopK pair value output view is too small.");

        using TopKPairsShader =
            Shader<1, Buffer<KeyT>, Buffer<KeyT>, Buffer<ValueT>, Buffer<ValueT>, uint, uint>;
        auto key       = luisa::format("device_topk_pairs+{}+{}+{}+{}",
                                 direction,
                                 Type::of<KeyT>()->description(),
                                 Type::of<ValueT>()->description(),
                                 get_binary_op_hash<KeyT>(better));
        auto shader_it = m_pair_shaders.find(key);
        if(shader_it == m_pair_shaders.end())
        {
            U<TopKPairsShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [better](BufferVar<KeyT>   keys_input_buffer,
                                  BufferVar<KeyT>   keys_output_buffer,
                                  BufferVar<ValueT> values_input_buffer,
                                  BufferVar<ValueT> values_output_buffer,
                                  UInt              num_items_var,
                                  UInt              output_count_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt candidate = dispatch_id().x;
                             $if(candidate < num_items_var)
                             {
                                 Var<KeyT> candidate_key = keys_input_buffer.read(candidate);
                                 UInt      rank          = 0u;
                                 $for(other, 0u, num_items_var, 1u)
                                 {
                                     Var<KeyT> other_key        = keys_input_buffer.read(other);
                                     Bool      other_better     = better(other_key, candidate_key);
                                     Bool      candidate_better = better(candidate_key, other_key);
                                     Bool      equivalent       = !other_better & !candidate_better;
                                     rank += select(0u, 1u, other_better | (equivalent & (other < candidate)));
                                 };
                                 $if(rank < output_count_var)
                                 {
                                     keys_output_buffer.write(rank, candidate_key);
                                     values_output_buffer.write(rank, values_input_buffer.read(candidate));
                                 };
                             };
                         });
            m_pair_shaders.try_emplace(key, std::move(shader));
            shader_it = m_pair_shaders.find(key);
        }
        auto shader = reinterpret_cast<TopKPairsShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTopK pair shader.");
        cmdlist << (*shader)(keys_input,
                             keys_output,
                             values_input,
                             values_output,
                             static_cast<uint>(num_items),
                             static_cast<uint>(output_count))
                       .dispatch(num_items);
    }
};

}  // namespace luisa::parallel_primitive
