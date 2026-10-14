/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/device/details/compact.h>
#include <lcpp/device/device_scan.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceRunLengthEncode : public LuisaModule
{
  private:
    using ScanT = DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
    using CompactT =
        details::CompactModule<BLOCK_SIZE == details::BLOCK_SIZE ? 384u : BLOCK_SIZE,
                               ITEMS_PER_THREAD == details::ITEMS_PER_THREAD ? 15u : ITEMS_PER_THREAD>;

    Device m_device;
    ScanT  m_scan;
    CompactT m_compact;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_head_flag_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_unique_scatter_shaders;
    luisa::shared_ptr<luisa::compute::Resource> m_run_offset_shader;
    luisa::shared_ptr<luisa::compute::Resource> m_run_length_shader;
    luisa::shared_ptr<luisa::compute::Resource> m_offset_length_shader;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_run_length_shader_map;
    luisa::shared_ptr<luisa::compute::Resource> m_non_trivial_flag_shader;
    luisa::shared_ptr<luisa::compute::Resource> m_non_trivial_scatter_shader;
    luisa::shared_ptr<luisa::compute::Resource> m_zero_count_shader;

  public:
    DeviceRunLengthEncode()  = default;
    ~DeviceRunLengthEncode() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
        m_scan.create(device);
        m_compact.create(device);
    }

    [[nodiscard]] static size_t GetTempStorageBytes(size_t num_items)
    {
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceRunLengthEncode.");
        if(num_items == 0u)
        {
            return 0u;
        }
        constexpr size_t arrays_per_item = 6u;
        LUISA_ASSERT(num_items <= (std::numeric_limits<size_t>::max() / sizeof(uint) - 1u) / arrays_per_item,
                     "DeviceRunLengthEncode temporary storage size overflow.");
        size_t fixed_bytes = (arrays_per_item * num_items + 1u) * sizeof(uint);
        fixed_bytes        = align_up_temp_storage(fixed_bytes);
        return fixed_bytes + ScanT::template GetTempStorageBytes<uint>(num_items);
    }

    template <NumericT T>
    void Encode(CommandList&     cmdlist,
                BufferView<uint> temp_storage,
                BufferView<T>    input,
                BufferView<T>    unique_output,
                BufferView<uint> counts_output,
                BufferView<uint> num_runs_output,
                size_t           num_items)
    {
        validate_encode_views(temp_storage, input, unique_output, counts_output, num_runs_output, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_runs_output);
            return;
        }

        auto views = temp_views(temp_storage, num_items);
        Callable<bool(T, T)> equality = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs == rhs; };
        if constexpr(sizeof(T) == sizeof(uint))
        {
            if(m_device.backend_name() == "cuda" && input.handle() != unique_output.handle())
            {
                m_compact.template Unique<true>(
                    cmdlist, views.scan_temp, input, unique_output, num_runs_output, views.run_offsets, num_items, equality);
                dispatch_lengths_from_offsets(cmdlist, views.run_offsets, counts_output, num_runs_output, num_items);
                return;
            }
        }
        if constexpr(sizeof(T) == sizeof(uint))
        {
            m_scan.ExclusiveSumAdjacent(cmdlist, views.scan_temp, input, views.head_positions, num_items, equality);
        }
        else
        {
            dispatch_head_flags(cmdlist, input, views.head_flags, num_items);
            m_scan.ExclusiveSum(cmdlist, views.scan_temp, views.head_flags, views.head_positions, num_items);
        }
        dispatch_unique_values_and_offsets(
            cmdlist, input, unique_output, views.head_positions, views.run_offsets, equality, num_items);
        dispatch_run_lengths_adjacent(
            cmdlist, input, views.head_positions, views.run_offsets, counts_output, num_runs_output, equality, num_items);
    }

    template <NumericT T>
    void NonTrivialRuns(CommandList&     cmdlist,
                        BufferView<uint> temp_storage,
                        BufferView<T>    input,
                        BufferView<uint> offsets_output,
                        BufferView<uint> lengths_output,
                        BufferView<uint> num_runs_output,
                        size_t           num_items)
    {
        validate_non_trivial_views(temp_storage, input, offsets_output, lengths_output, num_runs_output, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_runs_output);
            return;
        }

        auto views = temp_views(temp_storage, num_items);
        dispatch_head_flags(cmdlist, input, views.head_flags, num_items);
        m_scan.ExclusiveSum(cmdlist, views.scan_temp, views.head_flags, views.head_positions, num_items);
        dispatch_run_offsets(cmdlist, views.head_flags, views.head_positions, views.run_offsets, num_items);
        dispatch_run_lengths(cmdlist,
                             views.head_flags,
                             views.head_positions,
                             views.run_offsets,
                             views.run_lengths,
                             views.run_count,
                             num_items);
        dispatch_non_trivial_flags(cmdlist, views.run_lengths, views.run_count, views.non_trivial_flags, num_items);
        m_scan.ExclusiveSum(cmdlist, views.scan_temp, views.non_trivial_flags, views.non_trivial_positions, num_items);
        dispatch_non_trivial_scatter(cmdlist,
                                     views.run_offsets,
                                     views.run_lengths,
                                     views.non_trivial_flags,
                                     views.non_trivial_positions,
                                     offsets_output,
                                     lengths_output,
                                     num_runs_output,
                                     num_items);
    }

  private:
    void dispatch_lengths_from_offsets(CommandList&     cmdlist,
                                       BufferView<uint> offsets,
                                       BufferView<uint> lengths,
                                       BufferView<uint> count,
                                       size_t           num_items)
    {
        using Kernel = Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        if(!m_offset_length_shader)
        {
            U<Kernel> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> offsets_buffer,
                            BufferVar<uint> lengths_buffer,
                            BufferVar<uint> count_buffer,
                            UInt            num_elements) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             UInt runs  = count_buffer.read(0u);
                             $while(index < runs)
                             {
                                 UInt end = num_elements;
                                 $if(index + 1u < runs)
                                 {
                                     end = offsets_buffer.read(index + 1u);
                                 };
                                 lengths_buffer.write(index, end - offsets_buffer.read(index));
                                 index += dispatch_size().x;
                             };
                         });
            m_offset_length_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<Kernel*>(&(*m_offset_length_shader));
        size_t threads = std::min(num_items, ceil_div(num_items, CompactT::TILE_ITEMS) * BLOCK_SIZE);
        cmdlist << (*shader)(offsets, lengths, count, static_cast<uint>(num_items)).dispatch(threads);
    }

    struct TempViews
    {
        BufferView<uint> head_flags;
        BufferView<uint> head_positions;
        BufferView<uint> run_offsets;
        BufferView<uint> run_lengths;
        BufferView<uint> non_trivial_flags;
        BufferView<uint> non_trivial_positions;
        BufferView<uint> run_count;
        BufferView<uint> scan_temp;
    };

    [[nodiscard]] static TempViews temp_views(BufferView<uint> temp_storage, size_t num_items)
    {
        size_t offset     = 0u;
        auto   head_flags = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto head_positions = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto run_offsets = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto run_lengths = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto non_trivial_flags = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto non_trivial_positions = temp_storage.subview(offset, num_items);
        offset += num_items;
        auto run_count = temp_storage.subview(offset, 1u);
        offset += 1u;
        offset          = align_up_temp_storage(offset * sizeof(uint)) / sizeof(uint);
        auto scan_uints = bytes_to_uint_count(ScanT::template GetTempStorageBytes<uint>(num_items));
        auto scan_temp  = temp_storage.subview(offset, scan_uints);
        return {head_flags, head_positions, run_offsets, run_lengths, non_trivial_flags, non_trivial_positions, run_count, scan_temp};
    }

    void validate_common(BufferView<uint> temp_storage, BufferView<uint> num_runs_output, size_t num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceRunLengthEncode::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceRunLengthEncode.");
        LUISA_ASSERT(num_runs_output.size() >= 1u, "DeviceRunLengthEncode run-count output is empty.");
        if(num_items != 0u)
        {
            LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetTempStorageBytes(num_items)),
                         "DeviceRunLengthEncode temporary storage is too small.");
        }
    }

    template <NumericT T>
    void validate_encode_views(BufferView<uint> temp_storage,
                               BufferView<T>    input,
                               BufferView<T>    unique_output,
                               BufferView<uint> counts_output,
                               BufferView<uint> num_runs_output,
                               size_t           num_items) const
    {
        validate_common(temp_storage, num_runs_output, num_items);
        LUISA_ASSERT(input.size() >= num_items, "DeviceRunLengthEncode input view is too small.");
        LUISA_ASSERT(unique_output.size() >= num_items, "DeviceRunLengthEncode unique output view is too small.");
        LUISA_ASSERT(counts_output.size() >= num_items, "DeviceRunLengthEncode counts output view is too small.");
    }

    template <NumericT T>
    void validate_non_trivial_views(BufferView<uint> temp_storage,
                                    BufferView<T>    input,
                                    BufferView<uint> offsets_output,
                                    BufferView<uint> lengths_output,
                                    BufferView<uint> num_runs_output,
                                    size_t           num_items) const
    {
        validate_common(temp_storage, num_runs_output, num_items);
        LUISA_ASSERT(input.size() >= num_items, "DeviceRunLengthEncode input view is too small.");
        LUISA_ASSERT(offsets_output.size() >= num_items, "DeviceRunLengthEncode offsets output view is too small.");
        LUISA_ASSERT(lengths_output.size() >= num_items, "DeviceRunLengthEncode lengths output view is too small.");
    }

    template <typename... T>
    [[nodiscard]] static luisa::string shader_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    void write_zero_count(CommandList& cmdlist, BufferView<uint> num_runs_output)
    {
        using ZeroShader = Shader<1, Buffer<uint>>;
        if(!m_zero_count_shader)
        {
            U<ZeroShader> shader = nullptr;
            lazy_compile(
                m_device, shader, [](BufferVar<uint> output) noexcept { output.write(0u, 0u); });
            m_zero_count_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<ZeroShader*>(&(*m_zero_count_shader));
        cmdlist << (*shader)(num_runs_output).dispatch(1u);
    }

    template <NumericT T>
    void dispatch_head_flags(CommandList& cmdlist, BufferView<T> input, BufferView<uint> head_flags, size_t num_items)
    {
        using HeadFlagShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key             = shader_key<T>("run_length_encode_head_flags");
        auto shader_it       = m_head_flag_shaders.find(key);
        if(shader_it == m_head_flag_shaders.end())
        {
            U<HeadFlagShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<T> input_buffer, BufferVar<uint> flags_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index   = dispatch_id().x;
                             Bool is_head = index == 0u;
                             $if(index > 0u)
                             {
                                 is_head = input_buffer.read(index - 1u) != input_buffer.read(index);
                             };
                             flags_buffer.write(index, select(0u, 1u, is_head));
                         });
            m_head_flag_shaders.try_emplace(key, std::move(shader));
            shader_it = m_head_flag_shaders.find(key);
        }
        auto shader = reinterpret_cast<HeadFlagShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, head_flags).dispatch(num_items);
    }

    void dispatch_run_offsets(CommandList&     cmdlist,
                              BufferView<uint> head_flags,
                              BufferView<uint> head_positions,
                              BufferView<uint> run_offsets,
                              size_t           num_items)
    {
        using RunOffsetShader = Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>>;
        if(!m_run_offset_shader)
        {
            U<RunOffsetShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> flags_buffer, BufferVar<uint> positions_buffer, BufferVar<uint> offsets_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             $if(flags_buffer.read(index) != 0u)
                             {
                                 offsets_buffer.write(positions_buffer.read(index), index);
                             };
                         });
            m_run_offset_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<RunOffsetShader*>(&(*m_run_offset_shader));
        cmdlist << (*shader)(head_flags, head_positions, run_offsets).dispatch(num_items);
    }

    template <NumericT T>
    void dispatch_unique_values_and_offsets(CommandList&                cmdlist,
                                            BufferView<T>               input,
                                            BufferView<T>               unique_output,
                                            BufferView<uint>            head_positions,
                                            BufferView<uint>            run_offsets,
                                            const Callable<bool(T, T)>& equality,
                                            size_t                      num_items)
    {
        using UniqueShader = Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>>;
        auto key           = shader_key<T>("run_length_encode_unique_values_and_offsets_adjacent");
        key += luisa::format("+{}", equality.function().hash());
        auto shader_it = m_unique_scatter_shaders.find(key);
        if(shader_it == m_unique_scatter_shaders.end())
        {
            U<UniqueShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [equality](BufferVar<T>    input_buffer,
                                    BufferVar<T>    output_buffer,
                                    BufferVar<uint> positions_buffer,
                                    BufferVar<uint> offsets_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             Bool is_head = index == 0u;
                             $if(index > 0u)
                             {
                                 is_head = !equality(input_buffer.read(index - 1u), input_buffer.read(index));
                             };
                             $if(is_head)
                             {
                                 UInt position = positions_buffer.read(index);
                                 output_buffer.write(position, input_buffer.read(index));
                                 offsets_buffer.write(position, index);
                             };
                         });
            m_unique_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = m_unique_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<UniqueShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, unique_output, head_positions, run_offsets).dispatch(num_items);
    }

    template <NumericT T>
    void dispatch_run_lengths_adjacent(CommandList&                cmdlist,
                                       BufferView<T>               input,
                                       BufferView<uint>            head_positions,
                                       BufferView<uint>            run_offsets,
                                       BufferView<uint>            run_lengths,
                                       BufferView<uint>            run_count,
                                       const Callable<bool(T, T)>& equality,
                                       size_t                      num_items)
    {
        using RunLengthShader =
            Shader<1, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        auto key = shader_key<T>("run_length_encode_lengths_adjacent");
        key += luisa::format("+{}", equality.function().hash());
        auto shader_it = m_run_length_shader_map.find(key);
        if(shader_it == m_run_length_shader_map.end())
        {
            U<RunLengthShader> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [equality](BufferVar<T>    input_buffer,
                           BufferVar<uint> positions_buffer,
                           BufferVar<uint> offsets_buffer,
                           BufferVar<uint> lengths_buffer,
                           BufferVar<uint> count_buffer,
                           UInt            num_items_var) noexcept
                {
                    set_block_size(BLOCK_SIZE);
                    UInt index   = dispatch_id().x;
                    Bool is_head = index == 0u;
                    Bool is_tail = index + 1u == num_items_var;
                    $if(index > 0u)
                    {
                        is_head = !equality(input_buffer.read(index - 1u), input_buffer.read(index));
                    };
                    $if(index + 1u < num_items_var)
                    {
                        is_tail = !equality(input_buffer.read(index), input_buffer.read(index + 1u));
                    };
                    $if(is_tail)
                    {
                        UInt run_index = positions_buffer.read(index) + select(0u, 1u, is_head) - 1u;
                        lengths_buffer.write(run_index, index - offsets_buffer.read(run_index) + 1u);
                    };
                    $if(index + 1u == num_items_var)
                    {
                        count_buffer.write(0u, positions_buffer.read(index) + select(0u, 1u, is_head));
                    };
                });
            m_run_length_shader_map.try_emplace(key, std::move(shader));
            shader_it = m_run_length_shader_map.find(key);
        }
        auto shader = reinterpret_cast<RunLengthShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, head_positions, run_offsets, run_lengths, run_count, static_cast<uint>(num_items))
                       .dispatch(num_items);
    }

    void dispatch_run_lengths(CommandList&     cmdlist,
                              BufferView<uint> head_flags,
                              BufferView<uint> head_positions,
                              BufferView<uint> run_offsets,
                              BufferView<uint> run_lengths,
                              BufferView<uint> run_count,
                              size_t           num_items)
    {
        using RunLengthShader =
            Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        if(!m_run_length_shader)
        {
            U<RunLengthShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> flags_buffer,
                            BufferVar<uint> positions_buffer,
                            BufferVar<uint> offsets_buffer,
                            BufferVar<uint> lengths_buffer,
                            BufferVar<uint> count_buffer,
                            UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index   = dispatch_id().x;
                             UInt is_head = flags_buffer.read(index);
                             Bool is_tail = index + 1u == num_items_var;
                             $if(index + 1u < num_items_var)
                             {
                                 is_tail = flags_buffer.read(index + 1u) != 0u;
                             };
                             $if(is_tail)
                             {
                                 UInt run_index = positions_buffer.read(index) + is_head - 1u;
                                 lengths_buffer.write(run_index, index - offsets_buffer.read(run_index) + 1u);
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_buffer.write(0u, positions_buffer.read(index) + is_head);
                             };
                         });
            m_run_length_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<RunLengthShader*>(&(*m_run_length_shader));
        cmdlist << (*shader)(head_flags, head_positions, run_offsets, run_lengths, run_count, static_cast<uint>(num_items))
                       .dispatch(num_items);
    }

    void dispatch_non_trivial_flags(CommandList&     cmdlist,
                                    BufferView<uint> run_lengths,
                                    BufferView<uint> run_count,
                                    BufferView<uint> non_trivial_flags,
                                    size_t           num_items)
    {
        using NonTrivialFlagShader = Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>>;
        if(!m_non_trivial_flag_shader)
        {
            U<NonTrivialFlagShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> lengths_buffer, BufferVar<uint> count_buffer, BufferVar<uint> flags_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index  = dispatch_id().x;
                             UInt length = 0u;
                             $if(index < count_buffer.read(0u))
                             {
                                 length = lengths_buffer.read(index);
                             };
                             flags_buffer.write(index, select(0u, 1u, length > 1u));
                         });
            m_non_trivial_flag_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<NonTrivialFlagShader*>(&(*m_non_trivial_flag_shader));
        cmdlist << (*shader)(run_lengths, run_count, non_trivial_flags).dispatch(num_items);
    }

    void dispatch_non_trivial_scatter(CommandList&     cmdlist,
                                      BufferView<uint> run_offsets,
                                      BufferView<uint> run_lengths,
                                      BufferView<uint> non_trivial_flags,
                                      BufferView<uint> non_trivial_positions,
                                      BufferView<uint> offsets_output,
                                      BufferView<uint> lengths_output,
                                      BufferView<uint> num_runs_output,
                                      size_t           num_items)
    {
        using NonTrivialScatterShader =
            Shader<1, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        if(!m_non_trivial_scatter_shader)
        {
            U<NonTrivialScatterShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<uint> run_offsets_buffer,
                            BufferVar<uint> run_lengths_buffer,
                            BufferVar<uint> flags_buffer,
                            BufferVar<uint> positions_buffer,
                            BufferVar<uint> offsets_output_buffer,
                            BufferVar<uint> lengths_output_buffer,
                            BufferVar<uint> count_output_buffer,
                            UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             UInt selected = flags_buffer.read(index);
                             UInt position = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 offsets_output_buffer.write(position, run_offsets_buffer.read(index));
                                 lengths_output_buffer.write(position, run_lengths_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output_buffer.write(0u, position + selected);
                             };
                         });
            m_non_trivial_scatter_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<NonTrivialScatterShader*>(&(*m_non_trivial_scatter_shader));
        cmdlist << (*shader)(run_offsets,
                             run_lengths,
                             non_trivial_flags,
                             non_trivial_positions,
                             offsets_output,
                             lengths_output,
                             num_runs_output,
                             static_cast<uint>(num_items))
                       .dispatch(num_items);
    }
};

}  // namespace luisa::parallel_primitive
