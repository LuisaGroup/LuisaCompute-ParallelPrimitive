/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>
#include <luisa/runtime/bindless_array.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

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
};

}  // namespace luisa::parallel_primitive
