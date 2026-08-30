/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>
#include <luisa/runtime/bindless_array.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

enum class DeviceCopyLayout
{
    ROW_MAJOR,
    COLUMN_MAJOR
};

// Slot i in each bindless array represents one source/destination range and
// sizes[i] supplies its element count. Binding a BufferView preserves its
// logical start and length, so ranges can use arbitrary backing buffers.
// As in CUB, input ranges may alias one another, while input/output ranges and
// distinct output ranges must not overlap. The caller updates and retains the
// bindless arrays and their buffers through command execution.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceCopy : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u);

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_copy_shaders;

  public:
    DeviceCopy()  = default;
    ~DeviceCopy() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept { return 0u; }

    template <typename InputT, typename OutputT, size_t RANK>
    void Copy(CommandList&                    cmdlist,
              BufferView<uint>                temp_storage,
              BufferView<InputT>              input,
              BufferView<OutputT>             output,
              const std::array<size_t, RANK>& extents,
              DeviceCopyLayout                input_layout  = DeviceCopyLayout::ROW_MAJOR,
              DeviceCopyLayout                output_layout = DeviceCopyLayout::ROW_MAJOR)
    {
        static_cast<void>(temp_storage);
        LUISA_ASSERT(m_created, "DeviceCopy::create must be called before dispatch.");
        auto num_items = extent_size(extents);
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(input.size() >= num_items,
                     "DeviceCopy source mdspan requires {} elements, but its buffer view contains {}.",
                     num_items,
                     input.size());
        LUISA_ASSERT(output.size() >= num_items,
                     "DeviceCopy destination mdspan requires {} elements, but its buffer view contains {}.",
                     num_items,
                     output.size());
        validate_non_overlapping(input, output, num_items);

        using CopyShader = Shader<1, Buffer<InputT>, Buffer<OutputT>>;
        auto key         = luisa::format("device_copy_mdspan+{}+{}+{}+{}+{}",
                                 Type::of<InputT>()->description(),
                                 Type::of<OutputT>()->description(),
                                 RANK,
                                 static_cast<uint>(input_layout),
                                 static_cast<uint>(output_layout));
        for(auto extent : extents)
        {
            key += luisa::format("+{}", extent);
        }

        auto shader_it = m_copy_shaders.find(key);
        if(shader_it == m_copy_shaders.end())
        {
            U<CopyShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [extents, input_layout, output_layout](BufferVar<InputT> input_buffer,
                                                                BufferVar<OutputT> output_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt                   remaining = dispatch_id().x;
                             std::array<UInt, RANK> coordinates;
                             for(size_t dimension = RANK; dimension > 0u; --dimension)
                             {
                                 auto index         = dimension - 1u;
                                 coordinates[index] = remaining % static_cast<uint>(extents[index]);
                                 remaining /= static_cast<uint>(extents[index]);
                             }

                             auto physical_index = [&](DeviceCopyLayout layout) noexcept
                             {
                                 UInt offset = 0u;
                                 UInt stride = 1u;
                                 if(layout == DeviceCopyLayout::ROW_MAJOR)
                                 {
                                     for(size_t dimension = RANK; dimension > 0u; --dimension)
                                     {
                                         auto index = dimension - 1u;
                                         offset += coordinates[index] * stride;
                                         stride *= static_cast<uint>(extents[index]);
                                     }
                                 }
                                 else
                                 {
                                     for(size_t dimension = 0u; dimension < RANK; ++dimension)
                                     {
                                         offset += coordinates[dimension] * stride;
                                         stride *= static_cast<uint>(extents[dimension]);
                                     }
                                 }
                                 return offset;
                             };

                             auto value = input_buffer.read(physical_index(input_layout));
                             if constexpr(std::is_same_v<InputT, OutputT>)
                             {
                                 output_buffer.write(physical_index(output_layout), value);
                             }
                             else
                             {
                                 output_buffer.write(physical_index(output_layout), cast<OutputT>(value));
                             }
                         });
            m_copy_shaders.try_emplace(key, std::move(shader));
            shader_it = m_copy_shaders.find(key);
        }
        auto shader = reinterpret_cast<CopyShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceCopy mdspan shader.");
        cmdlist << (*shader)(input, output).dispatch(num_items);
    }

    template <typename InputT, typename OutputT = InputT>
    void Batched(CommandList&         cmdlist,
                 BufferView<uint>     temp_storage,
                 const BindlessArray& input_ranges,
                 const BindlessArray& output_ranges,
                 BufferView<uint>     sizes,
                 size_t               num_ranges)
    {
        static_cast<void>(temp_storage);
        LUISA_ASSERT(m_created, "DeviceCopy::create must be called before dispatch.");
        LUISA_ASSERT(num_ranges <= std::numeric_limits<uint>::max(), "Too many ranges for DeviceCopy.");
        if(num_ranges == 0u)
        {
            return;
        }
        LUISA_ASSERT(input_ranges.size() >= num_ranges, "DeviceCopy input bindless array is too small.");
        LUISA_ASSERT(output_ranges.size() >= num_ranges, "DeviceCopy output bindless array is too small.");
        LUISA_ASSERT(sizes.size() >= num_ranges, "DeviceCopy range-size view is too small.");

        using CopyShader = Shader<2, BindlessArray, BindlessArray, Buffer<uint>, uint>;
        luisa::string key{"device_copy_batched+"};
        key += Type::of<InputT>()->description();
        key += "+";
        key += Type::of<OutputT>()->description();
        auto shader_it = m_copy_shaders.find(key);
        if(shader_it == m_copy_shaders.end())
        {
            U<CopyShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BindlessVar input_ranges_var, BindlessVar output_ranges_var, BufferVar<uint> sizes_var, UInt num_ranges_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE, 1u, 1u);
                             UInt lane  = dispatch_id().x;
                             UInt range = dispatch_id().y;
                             $if(range < num_ranges_var)
                             {
                                 auto input_range = input_ranges_var.template buffer<InputT>(range);
                                 auto output_range = output_ranges_var.template buffer<OutputT>(range);
                                 UInt range_size = sizes_var.read(range);
                                 $for(item, lane, range_size, UInt(BLOCK_SIZE))
                                 {
                                     output_range.write(item, input_range.read(item));
                                 };
                             };
                         });
            m_copy_shaders.try_emplace(key, std::move(shader));
            shader_it = m_copy_shaders.find(key);
        }
        auto shader = reinterpret_cast<CopyShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input_ranges, output_ranges, sizes, static_cast<uint>(num_ranges))
                       .dispatch(static_cast<uint>(BLOCK_SIZE), static_cast<uint>(num_ranges));
    }

  private:
    template <size_t RANK>
    [[nodiscard]] static size_t extent_size(const std::array<size_t, RANK>& extents)
    {
        size_t size = 1u;
        for(auto extent : extents)
        {
            LUISA_ASSERT(extent <= std::numeric_limits<uint>::max(),
                         "DeviceCopy extent exceeds the shader index range.");
            if(extent == 0u)
            {
                return 0u;
            }
            LUISA_ASSERT(size <= std::numeric_limits<uint>::max() / extent,
                         "DeviceCopy extents exceed the shader index range.");
            size *= extent;
        }
        return size;
    }

    template <typename InputT, typename OutputT>
    static void validate_non_overlapping(BufferView<InputT> input, BufferView<OutputT> output, size_t num_items)
    {
        if(input.handle() != output.handle())
        {
            return;
        }
        auto input_begin  = input.offset_bytes();
        auto input_end    = input_begin + num_items * input.stride();
        auto output_begin = output.offset_bytes();
        auto output_end   = output_begin + num_items * output.stride();
        LUISA_ASSERT(input_end <= output_begin || output_end <= input_begin,
                     "DeviceCopy source and destination mdspan memory ranges must not overlap.");
    }
};

}  // namespace luisa::parallel_primitive
