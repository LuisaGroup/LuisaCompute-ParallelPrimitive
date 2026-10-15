/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

/// Portable segmented scans with CUB-compatible segment and initial-value semantics.
///
/// Built-in 32-bit sums use warp/block-cooperative scans. General operators retain
/// serial left-to-right evaluation, including non-associative operators.
/// Floating-point sums may round differently when evaluated in parallel.
template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceSegmentedScan : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceSegmentedScan block size must be positive.");
    static constexpr size_t LONG_ITEMS_PER_THREAD = 8u;

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
    [[nodiscard]] static UInt transpose_index(const UInt& logical_index)
    {
        // Offset consecutive 32-element rows by one shared-memory bank.
        return logical_index + logical_index / 32u;
    }

    static void sync_transpose(bool cuda_warp_sync)
    {
        if(cuda_warp_sync)
        {
            static ExternalCallable<void(uint)> syncwarp{"__syncwarp"};
            syncwarp(UInt(0xffffffffu));
        }
        else
        {
            sync_block();
        }
    }

    template <bool INCLUSIVE, bool USE_INITIAL_VALUE, NumericT T>
    static void emit_block_sum(const BufferVar<T>&    input_buffer,
                               const BufferVar<T>&    output_buffer,
                               const BufferVar<uint>& begin_offsets_buffer,
                               const BufferVar<uint>& end_offsets_buffer,
                               const BufferVar<uint>& output_begin_offsets_buffer,
                               const UInt&            num_segments,
                               const UInt&            segment_base,
                               const Var<T>&          initial_value,
                               uint                   warp_size,
                               uint                   warps_per_block,
                               bool                   cuda_warp_sync)
    {
        constexpr uint TILE_ITEMS   = static_cast<uint>(BLOCK_SIZE * LONG_ITEMS_PER_THREAD);
        constexpr uint SHARED_ITEMS = TILE_ITEMS + TILE_ITEMS / 32u;
        UInt segment = segment_base + block_id().x;
        $if(segment < num_segments)
        {
            UInt input_begin  = begin_offsets_buffer.read(segment);
            UInt input_end    = end_offsets_buffer.read(segment);
            UInt output_begin = output_begin_offsets_buffer.read(segment);
            $if(input_begin < input_end)
            {
                UInt thread     = thread_id().x;
                UInt lane       = warp_lane_id();
                UInt warp       = thread / warp_size;
                UInt warp_begin = warp * (warp_size * static_cast<uint>(LONG_ITEMS_PER_THREAD));
                UInt item_count = input_end - input_begin;
                UInt tile_begin = 0u;
                Var<T> carry    = T{};
                if constexpr(USE_INITIAL_VALUE)
                {
                    carry = initial_value;
                }
                Shared<T> transpose_storage{SHARED_ITEMS};
                Shared<T> warp_totals{warps_per_block};
                $while(tile_begin < item_count)
                {
                    UInt tile_size = min(item_count - tile_begin, static_cast<uint>(TILE_ITEMS));
                    ArrayVar<T, LONG_ITEMS_PER_THREAD> values;
                    // Each warp owns an independent padded shared window.
                    // Batch coalesced global loads before publishing to it.
                    ArrayVar<T, LONG_ITEMS_PER_THREAD> striped_values;
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = warp_begin + lane + static_cast<uint>(item) * warp_size;
                        striped_values[item] = T{};
                        $if(relative < tile_size)
                        {
                            striped_values[item] = input_buffer.read(input_begin + tile_begin + relative);
                        };
                    }
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = warp_begin + lane + static_cast<uint>(item) * warp_size;
                        transpose_storage.write(transpose_index(relative), striped_values[item]);
                    }
                    sync_transpose(cuda_warp_sync);
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = thread * static_cast<uint>(LONG_ITEMS_PER_THREAD) + static_cast<uint>(item);
                        values[item] = transpose_storage.read(transpose_index(relative));
                    }

                    Var<T> thread_total = values[0u];
                    for(size_t item = 1u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        thread_total += values[item];
                    }
                    Var<T> thread_prefix = warp_prefix_sum(thread_total);
                    $if(lane == warp_size - 1u)
                    {
                        warp_totals.write(warp, thread_prefix + thread_total);
                    };
                    // Publish warp totals and finish every shared-window read
                    // before the output transpose can reuse that window.
                    sync_block();
                    Var<T> warp_prefix = T{};
                    Var<T> tile_total  = T{};
                    for(uint preceding_warp = 0u; preceding_warp < warps_per_block; ++preceding_warp)
                    {
                        Var<T> partial = warp_totals.read(preceding_warp);
                        $if(preceding_warp < warp)
                        {
                            warp_prefix += partial;
                        };
                        tile_total += partial;
                    }
                    Var<T> prefix = carry + warp_prefix + thread_prefix;
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        Var<T> value = values[item];
                        if constexpr(INCLUSIVE)
                        {
                            prefix += value;
                            values[item] = prefix;
                        }
                        else
                        {
                            values[item] = prefix;
                            prefix += value;
                        }
                    }
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = thread * static_cast<uint>(LONG_ITEMS_PER_THREAD) + static_cast<uint>(item);
                        transpose_storage.write(transpose_index(relative), values[item]);
                    }
                    sync_transpose(cuda_warp_sync);
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = warp_begin + lane + static_cast<uint>(item) * warp_size;
                        striped_values[item] = transpose_storage.read(transpose_index(relative));
                    }
                    for(size_t item = 0u; item < LONG_ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = warp_begin + lane + static_cast<uint>(item) * warp_size;
                        $if(relative < tile_size)
                        {
                            output_buffer.write(output_begin + tile_begin + relative, striped_values[item]);
                        };
                    }
                    // Guard shared-window reuse and the next tile's warp-total
                    // writes, including CUDA's independently scheduled lanes.
                    sync_block();
                    carry += tile_total;
                    tile_begin += tile_size;
                };
            };
        };
    }

    template <bool INCLUSIVE, bool USE_INITIAL_VALUE, NumericT T>
    static void emit_warp_sum(const BufferVar<T>&    input_buffer,
                              const BufferVar<T>&    output_buffer,
                              const BufferVar<uint>& begin_offsets_buffer,
                              const BufferVar<uint>& end_offsets_buffer,
                              const BufferVar<uint>& output_begin_offsets_buffer,
                              const UInt&            num_segments,
                              const UInt&            segment_base,
                              const Var<T>&          initial_value,
                              uint                   warp_size,
                              uint                   warps_per_block)
    {
        constexpr uint ITEMS_PER_THREAD = 8u;
        UInt local_segment = block_id().x * warps_per_block + thread_id().x / warp_size;
        // Guard before adding the base so even UINT_MAX segments cannot wrap
        // the index of an idle warp in the final block.
        $if(local_segment < num_segments - segment_base)
        {
            UInt segment      = segment_base + local_segment;
            UInt input_begin  = begin_offsets_buffer.read(segment);
            UInt input_end    = end_offsets_buffer.read(segment);
            UInt output_begin = output_begin_offsets_buffer.read(segment);
            $if(input_begin < input_end)
            {
                UInt lane       = warp_lane_id();
                UInt item_count = input_end - input_begin;
                UInt tile_begin = 0u;
                Var<T> carry    = T{};
                if constexpr(USE_INITIAL_VALUE)
                {
                    carry = initial_value;
                }
                $while(tile_begin < item_count)
                {
                    UInt tile_size = min(item_count - tile_begin, warp_size * ITEMS_PER_THREAD);
                    std::array<Var<T>, ITEMS_PER_THREAD> values;
                    for(uint item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        UInt relative = lane + item * warp_size;
                        $if(relative < tile_size)
                        {
                            values[item] = input_buffer.read(input_begin + tile_begin + relative);
                        };
                    }
                    // Each row is contiguous across the warp. The row guard
                    // is warp-uniform, and tail lanes contribute zero.
                    for(uint item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        $if(item * warp_size < tile_size)
                        {
                            Var<T> prefix    = warp_prefix_sum(values[item]);
                            Var<T> inclusive = prefix + values[item];
                            UInt relative   = lane + item * warp_size;
                            $if(relative < tile_size)
                            {
                                if constexpr(INCLUSIVE)
                                {
                                    output_buffer.write(output_begin + tile_begin + relative, carry + inclusive);
                                }
                                else
                                {
                                    output_buffer.write(output_begin + tile_begin + relative, carry + prefix);
                                }
                            };
                            carry += warp_read_lane(inclusive, warp_size - 1u);
                        };
                    }
                    tile_begin += tile_size;
                };
            };
        };
    }

    template <bool INCLUSIVE, bool USE_INITIAL_VALUE, NumericT T>
    void dispatch_sum(CommandList&     cmdlist,
                      BufferView<T>    input,
                      BufferView<T>    output,
                      BufferView<uint> begin_offsets,
                      BufferView<uint> end_offsets,
                      BufferView<uint> output_begin_offsets,
                      uint             num_segments,
                      T                initial_value,
                      uint             warp_size)
    {
        constexpr uint SHORT_ITEMS_PER_THREAD = 8u;
        const uint     warps_per_block        = static_cast<uint>(BLOCK_SIZE) / warp_size;
        const bool     pin_warp_size          = m_device.backend_name() != "vk";
        const bool     cuda_warp_sync         = m_device.backend_name() == "cuda" && warp_size == details::WARP_SIZE;
        constexpr size_t TILE_ITEMS           = BLOCK_SIZE * LONG_ITEMS_PER_THREAD;
        const size_t shared_bytes             = (TILE_ITEMS + TILE_ITEMS / 32u + warps_per_block) * sizeof(T);
        // The transpose needs two register arrays as well as shared storage.
        // Keep register headroom for large custom blocks by using the warp
        // kernel, which also handles long ragged segments through more tiles.
        const bool warp_per_segment = input.size() / num_segments <= warp_size * SHORT_ITEMS_PER_THREAD
                                      || BLOCK_SIZE > 256u
                                      || shared_bytes > m_device.compute_max_shared_memory_size();
        using SegmentedScanShader =
            Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint, uint, T>;
        auto key = luisa::format("device_segmented_sum+inclusive:{}+initial:{}+warp:{}+pinned:{}+warp_segment:{}+long_items:{}+cuda_sync:{}+{}",
                                 INCLUSIVE,
                                 USE_INITIAL_VALUE,
                                 warp_size,
                                 pin_warp_size,
                                 warp_per_segment,
                                 LONG_ITEMS_PER_THREAD,
                                 cuda_warp_sync,
                                 get_type_and_op_desc<T>(SumOp{}));
        auto shader_it = m_scan_shaders.find(key);
        if(shader_it == m_scan_shaders.end())
        {
            U<SegmentedScanShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [warp_size, warps_per_block, pin_warp_size, warp_per_segment, cuda_warp_sync](
                             BufferVar<T>    input_buffer,
                             BufferVar<T>    output_buffer,
                             BufferVar<uint> begin_offsets_buffer,
                             BufferVar<uint> end_offsets_buffer,
                             BufferVar<uint> output_begin_offsets_buffer,
                             UInt            num_segments_var,
                             UInt            segment_base_var,
                             Var<T>          initial_value_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             // Vulkan uses its queried default subgroup width:
                             // LC emits SPIR-V 1.5 with no varying-size stage flag.
                             // Requesting it explicitly would require optional
                             // subgroup-size-control support even at that width.
                             if(pin_warp_size)
                             {
                                 set_warp_size(static_cast<uint8_t>(warp_size));
                             }
                             if(warp_per_segment)
                             {
                                 emit_warp_sum<INCLUSIVE, USE_INITIAL_VALUE>(input_buffer,
                                                                           output_buffer,
                                                                           begin_offsets_buffer,
                                                                           end_offsets_buffer,
                                                                           output_begin_offsets_buffer,
                                                                           num_segments_var,
                                                                           segment_base_var,
                                                                           initial_value_var,
                                                                           warp_size,
                                                                           warps_per_block);
                                 return;
                             }
                             emit_block_sum<INCLUSIVE, USE_INITIAL_VALUE>(input_buffer,
                                                                         output_buffer,
                                                                         begin_offsets_buffer,
                                                                         end_offsets_buffer,
                                                                         output_begin_offsets_buffer,
                                                                         num_segments_var,
                                                                         segment_base_var,
                                                                         initial_value_var,
                                                                         warp_size,
                                                                         warps_per_block,
                                                                         cuda_warp_sync);
                         });
            m_scan_shaders.try_emplace(key, std::move(shader));
            shader_it = m_scan_shaders.find(key);
        }
        auto shader = reinterpret_cast<SegmentedScanShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceSegmentedScan sum shader.");
        // 65535 workgroups along x is the portable Vulkan/DirectX minimum.
        // Keep offset views intact: arbitrary chunk boundaries need not meet
        // the backend's storage-buffer offset alignment.
        const size_t max_blocks_per_dispatch =
            std::min(size_t{65535u}, std::numeric_limits<uint>::max() / BLOCK_SIZE);
        const size_t segments_per_block = warp_per_segment ? warps_per_block : 1u;
        const size_t max_segments_per_dispatch = max_blocks_per_dispatch * segments_per_block;
        const uint threads_per_segment = warp_per_segment ? warp_size : static_cast<uint>(BLOCK_SIZE);
        for(size_t first = 0u; first < num_segments;)
        {
            auto count = static_cast<uint>(std::min(num_segments - first, max_segments_per_dispatch));
            cmdlist << (*shader)(input,
                                 output,
                                 begin_offsets,
                                 end_offsets,
                                 output_begin_offsets,
                                 num_segments,
                                 static_cast<uint>(first),
                                 initial_value)
                           .dispatch(count * threads_per_segment);
            first += count;
        }
    }

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

        if constexpr(std::is_same_v<std::remove_cvref_t<ScanOp>, SumOp>
                     && (std::is_same_v<T, int> || std::is_same_v<T, uint> || std::is_same_v<T, float>))
        {
            auto warp_size = m_device.compute_warp_size();
            if(warp_size >= 8u && warp_size <= 128u && (warp_size & (warp_size - 1u)) == 0u
               && BLOCK_SIZE % warp_size == 0u && BLOCK_SIZE <= std::numeric_limits<uint>::max())
            {
                dispatch_sum<INCLUSIVE, USE_INITIAL_VALUE>(cmdlist,
                                                          input,
                                                          output,
                                                          begin_offsets,
                                                          end_offsets,
                                                          output_begin_offsets,
                                                          static_cast<uint>(num_segments),
                                                          initial_value,
                                                          warp_size);
                return;
            }
        }

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
