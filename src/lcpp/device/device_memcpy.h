/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <lcpp/device/device_copy.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

// Slot i in each bindless array represents one source/destination byte range,
// and byte_counts[i] supplies the raw byte count. The caller updates and
// retains the bindless arrays and their buffers through command execution.
// As in CUB, input ranges may alias one another, while input/output ranges and
// distinct output ranges must not overlap.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceMemcpy : public LuisaModule
{
  private:
    DeviceCopy<BLOCK_SIZE> m_copy;

  public:
    DeviceMemcpy()  = default;
    ~DeviceMemcpy() = default;

    void create(Device& device) { m_copy.create(device); }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept { return 0u; }

    void Batched(CommandList&         cmdlist,
                 BufferView<uint>     temp_storage,
                 const BindlessArray& input_ranges,
                 const BindlessArray& output_ranges,
                 BufferView<uint>     byte_counts,
                 size_t               num_buffers)
    {
        m_copy.template Batched<luisa::ubyte>(cmdlist, temp_storage, input_ranges, output_ranges, byte_counts, num_buffers);
    }
};

}  // namespace luisa::parallel_primitive
