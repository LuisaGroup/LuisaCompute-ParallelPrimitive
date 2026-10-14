/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

/// Portable segmented scans with CUB-compatible left-to-right semantics.
///
/// Each segment is processed by one GPU worker. This favors semantic coverage
/// (including non-commutative operators and exact in-place execution) over the
/// throughput of CUB's CUDA-specific implementation.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceSegmentedScan : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceSegmentedScan block size must be positive.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_scan_shaders;

  public:
    DeviceSegmentedScan()  = default;
    ~DeviceSegmentedScan() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <NumericT T>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t) noexcept
    {
        return 0u;
    }

    template <NumericT T>
    void ExclusiveSegmentedSum(CommandList&     cmdlist,
                               BufferView<uint> temp_storage,
                               BufferView<T>    input,
                               BufferView<T>    output,
                               BufferView<uint> begin_offsets,
                               BufferView<uint> end_offsets,
                               size_t           num_segments)
    {
        ExclusiveSegmentedSum(cmdlist, temp_storage, input, output, begin_offsets, end_offsets, begin_offsets, num_segments);
    }

    template <NumericT T>
    void ExclusiveSegmentedSum(CommandList&     cmdlist,
                               BufferView<uint> temp_storage,
                               BufferView<T>    input,
                               BufferView<T>    output,
                               BufferView<uint> begin_offsets,
                               BufferView<uint> end_offsets,
                               BufferView<uint> output_begin_offsets,
                               size_t           num_segments)
    {
        dispatch_scan<false, true>(
            cmdlist, temp_storage, input, output, begin_offsets, end_offsets, output_begin_offsets, num_segments, SumOp{}, T{});
    }

    template <NumericT T, typename ScanOp>
    void ExclusiveSegmentedScan(CommandList&     cmdlist,
                                BufferView<uint> temp_storage,
                                BufferView<T>    input,
                                BufferView<T>    output,
                                BufferView<uint> begin_offsets,
                                BufferView<uint> end_offsets,
                                size_t           num_segments,
                                ScanOp           scan_op,
                                T                initial_value)
    {
        ExclusiveSegmentedScan(
            cmdlist, temp_storage, input, output, begin_offsets, end_offsets, begin_offsets, num_segments, std::move(scan_op), initial_value);
    }

    template <NumericT T, typename ScanOp>
    void ExclusiveSegmentedScan(CommandList&     cmdlist,
                                BufferView<uint> temp_storage,
                                BufferView<T>    input,
                                BufferView<T>    output,
                                BufferView<uint> begin_offsets,
                                BufferView<uint> end_offsets,
                                BufferView<uint> output_begin_offsets,
                                size_t           num_segments,
                                ScanOp           scan_op,
                                T                initial_value)
    {
        dispatch_scan<false, true>(cmdlist,
                                   temp_storage,
                                   input,
                                   output,
                                   begin_offsets,
                                   end_offsets,
                                   output_begin_offsets,
                                   num_segments,
                                   std::move(scan_op),
                                   initial_value);
    }

    template <NumericT T>
    void InclusiveSegmentedSum(CommandList&     cmdlist,
                               BufferView<uint> temp_storage,
                               BufferView<T>    input,
                               BufferView<T>    output,
                               BufferView<uint> begin_offsets,
                               BufferView<uint> end_offsets,
                               size_t           num_segments)
    {
        InclusiveSegmentedSum(cmdlist, temp_storage, input, output, begin_offsets, end_offsets, begin_offsets, num_segments);
    }

    template <NumericT T>
    void InclusiveSegmentedSum(CommandList&     cmdlist,
                               BufferView<uint> temp_storage,
                               BufferView<T>    input,
                               BufferView<T>    output,
                               BufferView<uint> begin_offsets,
                               BufferView<uint> end_offsets,
                               BufferView<uint> output_begin_offsets,
                               size_t           num_segments)
    {
        dispatch_scan<true, false>(
            cmdlist, temp_storage, input, output, begin_offsets, end_offsets, output_begin_offsets, num_segments, SumOp{}, T{});
    }

    template <NumericT T, typename ScanOp>
    void InclusiveSegmentedScan(CommandList&     cmdlist,
                                BufferView<uint> temp_storage,
                                BufferView<T>    input,
                                BufferView<T>    output,
                                BufferView<uint> begin_offsets,
                                BufferView<uint> end_offsets,
                                size_t           num_segments,
                                ScanOp           scan_op)
    {
        InclusiveSegmentedScan(
            cmdlist, temp_storage, input, output, begin_offsets, end_offsets, begin_offsets, num_segments, std::move(scan_op));
    }

    template <NumericT T, typename ScanOp>
    void InclusiveSegmentedScan(CommandList&     cmdlist,
                                BufferView<uint> temp_storage,
                                BufferView<T>    input,
                                BufferView<T>    output,
                                BufferView<uint> begin_offsets,
                                BufferView<uint> end_offsets,
                                BufferView<uint> output_begin_offsets,
                                size_t           num_segments,
                                ScanOp           scan_op)
    {
        dispatch_scan<true, false>(cmdlist,
                                   temp_storage,
                                   input,
                                   output,
                                   begin_offsets,
                                   end_offsets,
                                   output_begin_offsets,
                                   num_segments,
                                   std::move(scan_op),
                                   T{});
    }

    template <NumericT T, typename ScanOp>
    void InclusiveSegmentedScanInit(CommandList&     cmdlist,
                                    BufferView<uint> temp_storage,
                                    BufferView<T>    input,
                                    BufferView<T>    output,
                                    BufferView<uint> begin_offsets,
                                    BufferView<uint> end_offsets,
                                    size_t           num_segments,
                                    ScanOp           scan_op,
                                    T                initial_value)
    {
        InclusiveSegmentedScanInit(
            cmdlist, temp_storage, input, output, begin_offsets, end_offsets, begin_offsets, num_segments, std::move(scan_op), initial_value);
    }

    template <NumericT T, typename ScanOp>
    void InclusiveSegmentedScanInit(CommandList&     cmdlist,
                                    BufferView<uint> temp_storage,
                                    BufferView<T>    input,
                                    BufferView<T>    output,
                                    BufferView<uint> begin_offsets,
                                    BufferView<uint> end_offsets,
                                    BufferView<uint> output_begin_offsets,
                                    size_t           num_segments,
                                    ScanOp           scan_op,
                                    T                initial_value)
    {
        dispatch_scan<true, true>(cmdlist,
                                  temp_storage,
                                  input,
                                  output,
                                  begin_offsets,
                                  end_offsets,
                                  output_begin_offsets,
                                  num_segments,
                                  std::move(scan_op),
                                  initial_value);
    }

  private:
    template <bool INCLUSIVE, bool USE_INITIAL_VALUE, NumericT T, typename ScanOp>
    void dispatch_scan(CommandList& cmdlist,
                       BufferView<uint>,
                       BufferView<T>    input,
                       BufferView<T>    output,
                       BufferView<uint> begin_offsets,
                       BufferView<uint> end_offsets,
                       BufferView<uint> output_begin_offsets,
                       size_t           num_segments,
                       ScanOp           scan_op,
                       T                initial_value)
    {
        LUISA_ASSERT(m_created, "DeviceSegmentedScan::create must be called before dispatch.");
        LUISA_ASSERT(num_segments <= std::numeric_limits<uint>::max(),
                     "DeviceSegmentedScan segment count exceeds the shader index range.");
        if(num_segments == 0u)
        {
            return;
        }
        LUISA_ASSERT(begin_offsets.size() >= num_segments, "DeviceSegmentedScan begin-offset view is too small.");
        LUISA_ASSERT(end_offsets.size() >= num_segments, "DeviceSegmentedScan end-offset view is too small.");
        LUISA_ASSERT(output_begin_offsets.size() >= num_segments,
                     "DeviceSegmentedScan output-begin-offset view is too small.");

        using SegmentedScanShader =
            Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint, T>;
        auto key       = luisa::format("device_segmented_scan+inclusive:{}+initial:{}+{}",
                                 INCLUSIVE,
                                 USE_INITIAL_VALUE,
                                 get_type_and_op_desc<T>(scan_op));
        auto shader_it = m_scan_shaders.find(key);
        if(shader_it == m_scan_shaders.end())
        {
            U<SegmentedScanShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [scan_op](BufferVar<T>    input_buffer,
                                   BufferVar<T>    output_buffer,
                                   BufferVar<uint> begin_offsets_buffer,
                                   BufferVar<uint> end_offsets_buffer,
                                   BufferVar<uint> output_begin_offsets_buffer,
                                   UInt            num_segments_var,
                                   Var<T>          initial_value_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt segment = dispatch_id().x;
                             $if(segment < num_segments_var)
                             {
                                 UInt input_begin  = begin_offsets_buffer.read(segment);
                                 UInt input_end    = end_offsets_buffer.read(segment);
                                 UInt output_begin = output_begin_offsets_buffer.read(segment);
                                 $if(input_begin < input_end)
                                 {
                                     if constexpr(INCLUSIVE)
                                     {
                                         Var<T> aggregate;
                                         UInt   first_index;
                                         if constexpr(USE_INITIAL_VALUE)
                                         {
                                             aggregate   = initial_value_var;
                                             first_index = input_begin;
                                         }
                                         else
                                         {
                                             aggregate   = input_buffer.read(input_begin);
                                             first_index = input_begin + 1u;
                                             output_buffer.write(output_begin, aggregate);
                                         }
                                         $for(index, first_index, input_end)
                                         {
                                             Var<T> value = input_buffer.read(index);
                                             aggregate    = scan_op(aggregate, value);
                                             output_buffer.write(output_begin + index - input_begin, aggregate);
                                         };
                                     }
                                     else
                                     {
                                         Var<T> aggregate = initial_value_var;
                                         $for(index, input_begin, input_end)
                                         {
                                             Var<T> value = input_buffer.read(index);
                                             output_buffer.write(output_begin + index - input_begin, aggregate);
                                             aggregate = scan_op(aggregate, value);
                                         };
                                     }
                                 };
                             };
                         });
            m_scan_shaders.try_emplace(key, std::move(shader));
            shader_it = m_scan_shaders.find(key);
        }

        auto shader = reinterpret_cast<SegmentedScanShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceSegmentedScan shader.");
        cmdlist
            << (*shader)(input, output, begin_offsets, end_offsets, output_begin_offsets, static_cast<uint>(num_segments), initial_value)
                   .dispatch(num_segments);
    }
};
}  // namespace luisa::parallel_primitive
