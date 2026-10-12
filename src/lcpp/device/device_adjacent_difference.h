/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceAdjacentDifference : public LuisaModule
{
  private:
    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_difference_shaders;

  public:
    DeviceAdjacentDifference()  = default;
    ~DeviceAdjacentDifference() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <NumericT T>
    [[nodiscard]] static size_t GetTempStorageBytes(size_t num_items)
    {
        LUISA_ASSERT(num_items <= std::numeric_limits<size_t>::max() / sizeof(T),
                     "DeviceAdjacentDifference temporary storage size overflow.");
        return num_items * sizeof(T);
    }

    template <NumericT T>
    [[nodiscard]] static constexpr size_t GetCopyTempStorageBytes(size_t) noexcept
    {
        return 0u;
    }

    template <NumericT T>
    void SubtractLeftCopy(CommandList&     cmdlist,
                          BufferView<uint> temp_storage,
                          BufferView<T>    input,
                          BufferView<T>    output,
                          size_t           num_items)
    {
        Callable<T(T, T)> difference = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs - rhs; };
        SubtractLeftCopy(cmdlist, temp_storage, input, output, num_items, difference);
    }

    template <NumericT T>
    void SubtractLeftCopy(CommandList&             cmdlist,
                          BufferView<uint>         temp_storage,
                          BufferView<T>            input,
                          BufferView<T>            output,
                          size_t                   num_items,
                          const Callable<T(T, T)>& difference)
    {
        static_cast<void>(temp_storage);
        validate_copy_views(input, output, num_items);
        if(num_items == 0u)
        {
            return;
        }
        dispatch_difference<true>(cmdlist, input, output, num_items, difference);
    }

    template <NumericT T>
    void SubtractRightCopy(CommandList&     cmdlist,
                           BufferView<uint> temp_storage,
                           BufferView<T>    input,
                           BufferView<T>    output,
                           size_t           num_items)
    {
        Callable<T(T, T)> difference = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs - rhs; };
        SubtractRightCopy(cmdlist, temp_storage, input, output, num_items, difference);
    }

    template <NumericT T>
    void SubtractRightCopy(CommandList&             cmdlist,
                           BufferView<uint>         temp_storage,
                           BufferView<T>            input,
                           BufferView<T>            output,
                           size_t                   num_items,
                           const Callable<T(T, T)>& difference)
    {
        static_cast<void>(temp_storage);
        validate_copy_views(input, output, num_items);
        if(num_items == 0u)
        {
            return;
        }
        dispatch_difference<false>(cmdlist, input, output, num_items, difference);
    }

    template <NumericT T>
    void SubtractLeft(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<T> data, size_t num_items)
    {
        Callable<T(T, T)> difference = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs - rhs; };
        SubtractLeft(cmdlist, temp_storage, data, num_items, difference);
    }

    template <NumericT T>
    void SubtractLeft(CommandList&             cmdlist,
                      BufferView<uint>         temp_storage,
                      BufferView<T>            data,
                      size_t                   num_items,
                      const Callable<T(T, T)>& difference)
    {
        dispatch_in_place<true>(cmdlist, temp_storage, data, num_items, difference);
    }

    template <NumericT T>
    void SubtractRight(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<T> data, size_t num_items)
    {
        Callable<T(T, T)> difference = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs - rhs; };
        SubtractRight(cmdlist, temp_storage, data, num_items, difference);
    }

    template <NumericT T>
    void SubtractRight(CommandList&             cmdlist,
                       BufferView<uint>         temp_storage,
                       BufferView<T>            data,
                       size_t                   num_items,
                       const Callable<T(T, T)>& difference)
    {
        dispatch_in_place<false>(cmdlist, temp_storage, data, num_items, difference);
    }

  private:
    template <NumericT T>
    void validate_copy_views(BufferView<T> input, BufferView<T> output, size_t num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceAdjacentDifference::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceAdjacentDifference.");
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(input.size() >= num_items, "DeviceAdjacentDifference input view is too small.");
        LUISA_ASSERT(output.size() >= num_items, "DeviceAdjacentDifference output view is too small.");
    }

    template <bool READ_LEFT, NumericT T>
    void dispatch_in_place(CommandList&             cmdlist,
                           BufferView<uint>         temp_storage,
                           BufferView<T>            data,
                           size_t                   num_items,
                           const Callable<T(T, T)>& difference)
    {
        validate_copy_views(data, data, num_items);
        if(num_items == 0u)
        {
            return;
        }
        auto temp_uint_count = bytes_to_uint_count(GetTempStorageBytes<T>(num_items));
        LUISA_ASSERT(temp_storage.size() >= temp_uint_count,
                     "DeviceAdjacentDifference temporary storage is too small.");
        auto scratch = temp_storage.subview(0u, temp_uint_count).template as<T>().subview(0u, num_items);
        dispatch_difference<READ_LEFT>(cmdlist, data, scratch, num_items, difference);
        cmdlist << data.subview(0u, num_items).copy_from(scratch);
    }

    template <bool READ_LEFT, NumericT T>
    void dispatch_difference(CommandList&             cmdlist,
                             BufferView<T>            input,
                             BufferView<T>            output,
                             size_t                   num_items,
                             const Callable<T(T, T)>& difference)
    {
        using DifferenceShader = Shader<1, Buffer<T>, Buffer<T>, uint>;
        luisa::string key{READ_LEFT ? "adjacent_difference_left" : "adjacent_difference_right"};
        key += "+";
        key += Type::of<T>()->description();
        key += luisa::format("+{}", difference.function().hash());
        auto shader_it = m_difference_shaders.find(key);
        if(shader_it == m_difference_shaders.end())
        {
            U<DifferenceShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [difference](BufferVar<T> input_buffer, BufferVar<T> output_buffer, UInt num_items_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt   index  = dispatch_id().x;
                    Var<T> result = input_buffer.read(index);
                    if constexpr(READ_LEFT)
                    {
                        $if(index > 0u)
                        {
                            result = difference(input_buffer.read(index), input_buffer.read(index - 1u));
                        };
                    }
                    else
                    {
                        $if(index + 1u < num_items_var)
                        {
                            result = difference(input_buffer.read(index), input_buffer.read(index + 1u));
                        };
                    }
                    output_buffer.write(index, result);
                });
            m_difference_shaders.try_emplace(key, std::move(shader));
            shader_it = m_difference_shaders.find(key);
        }
        auto shader = reinterpret_cast<DifferenceShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, output, static_cast<uint>(num_items)).dispatch(num_items);
    }
};

}  // namespace luisa::parallel_primitive
