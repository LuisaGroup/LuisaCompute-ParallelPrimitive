/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceFind : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceFind block size must be positive.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_fill_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_find_if_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_bound_shaders;

  public:
    DeviceFind()  = default;
    ~DeviceFind() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <typename T>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept
    {
        return 0u;
    }

    template <typename RangeT, typename ValueT>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t) noexcept
    {
        return 0u;
    }

    template <typename T, typename Predicate>
    void FindIf(CommandList& cmdlist, BufferView<uint>, BufferView<T> input, BufferView<uint> output, Predicate predicate, size_t num_items)
    {
        LUISA_ASSERT(m_created, "DeviceFind::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(),
                     "DeviceFind item count exceeds the output index range.");
        LUISA_ASSERT(output.size() >= 1u, "DeviceFind output view is empty.");

        fill_indices(cmdlist, output.subview(0u, 1u), static_cast<uint>(num_items));
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(input.size() >= num_items, "DeviceFind input view is too small.");

        using FindIfShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key =
            luisa::format("device_find_if+{}+{}", Type::of<T>()->description(), get_unary_op_hash<T>(predicate));
        auto shader_it = m_find_if_shaders.find(key);
        if(shader_it == m_find_if_shaders.end())
        {
            U<FindIfShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate](BufferVar<T> input_buffer, BufferVar<uint> output_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             $if(predicate(input_buffer.read(index)))
                             {
                                 output_buffer.atomic(0u).fetch_min(index);
                             };
                         });
            m_find_if_shaders.try_emplace(key, std::move(shader));
            shader_it = m_find_if_shaders.find(key);
        }

        auto shader = reinterpret_cast<FindIfShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFind::FindIf shader.");
        cmdlist << (*shader)(input, output).dispatch(num_items);
    }

    template <typename RangeT, typename ValueT, typename Compare>
    void LowerBound(CommandList&       cmdlist,
                    BufferView<uint>   temp_storage,
                    BufferView<RangeT> range,
                    size_t             range_num_items,
                    BufferView<ValueT> values,
                    size_t             values_num_items,
                    BufferView<uint>   output,
                    Compare            compare)
    {
        dispatch_bound<false>(
            cmdlist, temp_storage, range, range_num_items, values, values_num_items, output, std::move(compare));
    }

    template <typename RangeT, typename ValueT, typename Compare>
    void UpperBound(CommandList&       cmdlist,
                    BufferView<uint>   temp_storage,
                    BufferView<RangeT> range,
                    size_t             range_num_items,
                    BufferView<ValueT> values,
                    size_t             values_num_items,
                    BufferView<uint>   output,
                    Compare            compare)
    {
        dispatch_bound<true>(
            cmdlist, temp_storage, range, range_num_items, values, values_num_items, output, std::move(compare));
    }

    // v3.5 adds sorted-query variants. The portable implementation keeps the
    // same result contract and uses the existing per-query binary search;
    // callers must still provide ranges and queries sorted consistently.
    template <typename RangeT, typename ValueT, typename Compare>
    void LowerBoundSortedValues(CommandList&       cmdlist,
                                BufferView<uint>   temp_storage,
                                BufferView<RangeT> range,
                                size_t             range_num_items,
                                BufferView<ValueT> values,
                                size_t             values_num_items,
                                BufferView<uint>   output,
                                Compare            compare)
    {
        dispatch_bound<false>(
            cmdlist, temp_storage, range, range_num_items, values, values_num_items, output, std::move(compare));
    }

    template <typename RangeT, typename ValueT, typename Compare>
    void UpperBoundSortedValues(CommandList&       cmdlist,
                                BufferView<uint>   temp_storage,
                                BufferView<RangeT> range,
                                size_t             range_num_items,
                                BufferView<ValueT> values,
                                size_t             values_num_items,
                                BufferView<uint>   output,
                                Compare            compare)
    {
        dispatch_bound<true>(
            cmdlist, temp_storage, range, range_num_items, values, values_num_items, output, std::move(compare));
    }

  private:
    void fill_indices(CommandList& cmdlist, BufferView<uint> output, uint value)
    {
        if(output.size() == 0u)
        {
            return;
        }
        using FillShader = Shader<1, Buffer<uint>, uint>;
        luisa::string key{"device_find_fill_indices"};
        auto          shader_it = m_fill_shaders.find(key);
        if(shader_it == m_fill_shaders.end())
        {
            U<FillShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> output_buffer, UInt value_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             output_buffer.write(dispatch_id().x, value_var);
                         });
            m_fill_shaders.try_emplace(key, std::move(shader));
            shader_it = m_fill_shaders.find(key);
        }

        auto shader = reinterpret_cast<FillShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFind fill shader.");
        cmdlist << (*shader)(output, value).dispatch(output.size());
    }

    template <typename LhsT, typename RhsT, typename Compare>
    [[nodiscard]] static uint64_t relation_hash(Compare compare)
    {
        Callable relation = [compare](Var<LhsT> lhs, Var<RhsT> rhs) noexcept
        { return compare(lhs, rhs); };
        return relation.function().hash();
    }

    template <bool UPPER, typename RangeT, typename ValueT, typename Compare>
    void dispatch_bound(CommandList& cmdlist,
                        BufferView<uint>,
                        BufferView<RangeT> range,
                        size_t             range_num_items,
                        BufferView<ValueT> values,
                        size_t             values_num_items,
                        BufferView<uint>   output,
                        Compare            compare)
    {
        LUISA_ASSERT(m_created, "DeviceFind::create must be called before dispatch.");
        LUISA_ASSERT(range_num_items <= std::numeric_limits<uint>::max(),
                     "DeviceFind range size exceeds the output index range.");
        LUISA_ASSERT(values_num_items <= std::numeric_limits<uint>::max(),
                     "DeviceFind query count exceeds the shader index range.");
        if(values_num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(values.size() >= values_num_items, "DeviceFind query view is too small.");
        LUISA_ASSERT(output.size() >= values_num_items, "DeviceFind bounds output view is too small.");
        if(range_num_items == 0u)
        {
            fill_indices(cmdlist, output.subview(0u, values_num_items), 0u);
            return;
        }
        LUISA_ASSERT(range.size() >= range_num_items, "DeviceFind ordered-range view is too small.");

        using BoundShader = Shader<1, Buffer<RangeT>, Buffer<ValueT>, Buffer<uint>, uint>;
        uint64_t compare_hash;
        if constexpr(UPPER)
        {
            compare_hash = relation_hash<ValueT, RangeT>(compare);
        }
        else
        {
            compare_hash = relation_hash<RangeT, ValueT>(compare);
        }
        auto key       = luisa::format("device_find_bound+upper:{}+{}+{}+{}",
                                 UPPER,
                                 Type::of<RangeT>()->description(),
                                 Type::of<ValueT>()->description(),
                                 compare_hash);
        auto shader_it = m_bound_shaders.find(key);
        if(shader_it == m_bound_shaders.end())
        {
            U<BoundShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [compare](BufferVar<RangeT> range_buffer,
                                   BufferVar<ValueT> values_buffer,
                                   BufferVar<uint>   output_buffer,
                                   UInt              range_num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt        query_index = dispatch_id().x;
                             Var<ValueT> value       = values_buffer.read(query_index);
                             UInt        low         = 0u;
                             UInt        high        = range_num_items_var;
                             $while(low < high)
                             {
                                 UInt        mid  = low + (high - low) / 2u;
                                 Var<RangeT> item = range_buffer.read(mid);
                                 Bool        advance;
                                 if constexpr(UPPER)
                                 {
                                     advance = !compare(value, item);
                                 }
                                 else
                                 {
                                     advance = compare(item, value);
                                 }
                                 $if(advance)
                                 {
                                     low = mid + 1u;
                                 }
                                 $else
                                 {
                                     high = mid;
                                 };
                             };
                             output_buffer.write(query_index, low);
                         });
            m_bound_shaders.try_emplace(key, std::move(shader));
            shader_it = m_bound_shaders.find(key);
        }

        auto shader = reinterpret_cast<BoundShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFind bounds shader.");
        cmdlist << (*shader)(range, values, output, static_cast<uint>(range_num_items)).dispatch(values_num_items);
    }
};
}  // namespace luisa::parallel_primitive
