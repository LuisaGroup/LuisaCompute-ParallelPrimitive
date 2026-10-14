/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

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
class DevicePartition : public LuisaModule
{
  private:
    using ScanT = DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
    using CompactT =
        details::CompactModule<BLOCK_SIZE == details::BLOCK_SIZE ? 384u : BLOCK_SIZE,
                               ITEMS_PER_THREAD == details::ITEMS_PER_THREAD ? 15u : ITEMS_PER_THREAD>;

    Device _device;
    ScanT  _scan;
    CompactT _compact;
    bool   _created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _flag_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _predicate_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _scatter_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _predicate_scatter_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _flag_scatter_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _three_way_flag_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _three_way_scatter_shaders;
    luisa::shared_ptr<luisa::compute::Resource> _zero_count_shader;
    luisa::shared_ptr<luisa::compute::Resource> _zero_counts_shader;

  public:
    DevicePartition()  = default;
    ~DevicePartition() = default;

    void create(Device& device)
    {
        _device  = device;
        _created = true;
        _scan.create(device);
        _compact.create(device);
    }

    [[nodiscard]] static size_t GetTempStorageBytes(size_t num_items)
    {
        if(num_items == 0u)
        {
            return 0u;
        }
        return 2u * num_items * sizeof(uint) + ScanT::template GetTempStorageBytes<uint>(num_items);
    }

    [[nodiscard]] static size_t GetThreeWayTempStorageBytes(size_t num_items)
    {
        if(num_items == 0u)
        {
            return 0u;
        }
        return 3u * num_items * sizeof(uint) + ScanT::template GetTempStorageBytes<uint>(num_items);
    }

    template <NumericT T, NumericT FlagT>
    void Flagged(CommandList&      cmdlist,
                 BufferView<uint>  temp_storage,
                 BufferView<T>     input,
                 BufferView<FlagT> selection_flags,
                 BufferView<T>     output,
                 BufferView<uint>  num_selected_out,
                 size_t            num_items)
    {
        _validate_views(temp_storage, input, output, num_selected_out, num_items);
        LUISA_ASSERT(selection_flags.size() >= num_items, "Partition flag view is too small.");
        if(num_items == 0u)
        {
            _write_zero_count(cmdlist, num_selected_out);
            return;
        }

        Callable<bool(FlagT)> flag_predicate = [](Var<FlagT> flag) noexcept
        { return flag != FlagT{}; };
        if constexpr(sizeof(T) == sizeof(uint) && sizeof(FlagT) == sizeof(uint))
        {
            if(_device.backend_name() == "cuda" && input.handle() != output.handle()
               && details::can_batch_callable(flag_predicate.function()))
            {
                _compact.template Flagged<true>(
                    cmdlist, temp_storage, input, selection_flags, output, num_selected_out, num_items, flag_predicate);
                return;
            }
        }
        auto [flags, positions, scan_temp] = _temp_views(temp_storage, num_items);
        if constexpr(sizeof(FlagT) == sizeof(uint))
        {
            _scan.ExclusiveSumTransform(cmdlist, scan_temp, selection_flags, positions, num_items, flag_predicate);
        }
        else
        {
            _dispatch_flags(cmdlist, selection_flags, flags, num_items);
            _scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        }
        _dispatch_flag_scatter(cmdlist, input, selection_flags, positions, output, num_selected_out, num_items);
    }

    template <NumericT T>
    void If(CommandList&             cmdlist,
            BufferView<uint>         temp_storage,
            BufferView<T>            input,
            BufferView<T>            output,
            BufferView<uint>         num_selected_out,
            size_t                   num_items,
            const Callable<bool(T)>& predicate)
    {
        _validate_views(temp_storage, input, output, num_selected_out, num_items);
        if(num_items == 0u)
        {
            _write_zero_count(cmdlist, num_selected_out);
            return;
        }

        if constexpr(sizeof(T) == sizeof(uint))
        {
            if(_device.backend_name() == "cuda" && input.handle() != output.handle()
               && details::can_batch_callable(predicate.function()))
            {
                _compact.template If<true>(cmdlist, temp_storage, input, output, num_selected_out, num_items, predicate);
                return;
            }
        }
        auto [flags, positions, scan_temp] = _temp_views(temp_storage, num_items);
        if constexpr(sizeof(T) == sizeof(uint))
        {
            _scan.ExclusiveSumTransform(cmdlist, scan_temp, input, positions, num_items, predicate);
        }
        else
        {
            _dispatch_predicate(cmdlist, input, flags, num_items, predicate);
            _scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        }
        _dispatch_predicate_scatter(cmdlist, input, predicate, positions, output, num_selected_out, num_items);
    }

    template <NumericT T>
    void If(CommandList&             cmdlist,
            BufferView<uint>         temp_storage,
            BufferView<T>            input,
            BufferView<T>            first_output,
            BufferView<T>            second_output,
            BufferView<T>            unselected_output,
            BufferView<uint>         num_selected_out,
            size_t                   num_items,
            const Callable<bool(T)>& select_first,
            const Callable<bool(T)>& select_second)
    {
        _validate_three_way_views(
            temp_storage, input, first_output, second_output, unselected_output, num_selected_out, num_items);
        if(num_items == 0u)
        {
            _write_zero_counts(cmdlist, num_selected_out);
            return;
        }

        auto [flags, first_positions, second_positions, scan_temp] =
            _three_way_temp_views(temp_storage, num_items);
        _dispatch_three_way_flags(cmdlist, input, flags, num_items, select_first, select_second, false);
        _scan.ExclusiveSum(cmdlist, scan_temp, flags, first_positions, num_items);
        _dispatch_three_way_flags(cmdlist, input, flags, num_items, select_first, select_second, true);
        _scan.ExclusiveSum(cmdlist, scan_temp, flags, second_positions, num_items);
        _dispatch_three_way_scatter(cmdlist,
                                    input,
                                    first_output,
                                    second_output,
                                    unselected_output,
                                    first_positions,
                                    second_positions,
                                    num_selected_out,
                                    num_items,
                                    select_first,
                                    select_second);
    }

  private:
    struct TempViews
    {
        BufferView<uint> flags;
        BufferView<uint> positions;
        BufferView<uint> scan_temp;
    };

    struct ThreeWayTempViews
    {
        BufferView<uint> flags;
        BufferView<uint> first_positions;
        BufferView<uint> second_positions;
        BufferView<uint> scan_temp;
    };

    [[nodiscard]] static TempViews _temp_views(BufferView<uint> temp_storage, size_t num_items)
    {
        size_t scan_uints = bytes_to_uint_count(ScanT::template GetTempStorageBytes<uint>(num_items));
        return {temp_storage.subview(0u, num_items),
                temp_storage.subview(num_items, num_items),
                temp_storage.subview(2u * num_items, scan_uints)};
    }

    [[nodiscard]] static ThreeWayTempViews _three_way_temp_views(BufferView<uint> temp_storage, size_t num_items)
    {
        size_t scan_uints = bytes_to_uint_count(ScanT::template GetTempStorageBytes<uint>(num_items));
        return {temp_storage.subview(0u, num_items),
                temp_storage.subview(num_items, num_items),
                temp_storage.subview(2u * num_items, num_items),
                temp_storage.subview(3u * num_items, scan_uints)};
    }

    template <NumericT T>
    void _validate_views(BufferView<uint> temp_storage,
                         BufferView<T>    input,
                         BufferView<T>    output,
                         BufferView<uint> num_selected_out,
                         size_t           num_items) const
    {
        LUISA_ASSERT(_created, "DevicePartition::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DevicePartition.");
        LUISA_ASSERT(input.size() >= num_items, "Partition input view is too small.");
        LUISA_ASSERT(output.size() >= num_items, "Partition output view is too small.");
        LUISA_ASSERT(num_selected_out.size() >= 1u, "Partition selected-count output is empty.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetTempStorageBytes(num_items)),
                     "DevicePartition temporary storage is too small.");
    }

    template <NumericT T>
    void _validate_three_way_views(BufferView<uint> temp_storage,
                                   BufferView<T>    input,
                                   BufferView<T>    first_output,
                                   BufferView<T>    second_output,
                                   BufferView<T>    unselected_output,
                                   BufferView<uint> num_selected_out,
                                   size_t           num_items) const
    {
        LUISA_ASSERT(_created, "DevicePartition::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DevicePartition.");
        LUISA_ASSERT(input.size() >= num_items, "Partition input view is too small.");
        LUISA_ASSERT(first_output.size() >= num_items, "Partition first output view is too small.");
        LUISA_ASSERT(second_output.size() >= num_items, "Partition second output view is too small.");
        LUISA_ASSERT(unselected_output.size() >= num_items, "Partition unselected output view is too small.");
        LUISA_ASSERT(num_selected_out.size() >= 2u, "Three-way partition selected-count output requires two items.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetThreeWayTempStorageBytes(num_items)),
                     "DevicePartition three-way temporary storage is too small.");
        _validate_non_overlapping(input, first_output, num_items);
        _validate_non_overlapping(input, second_output, num_items);
        _validate_non_overlapping(input, unselected_output, num_items);
        _validate_non_overlapping(first_output, second_output, num_items);
        _validate_non_overlapping(first_output, unselected_output, num_items);
        _validate_non_overlapping(second_output, unselected_output, num_items);
    }

    template <typename FirstT, typename SecondT>
    static void _validate_non_overlapping(BufferView<FirstT> first, BufferView<SecondT> second, size_t num_items)
    {
        if(num_items == 0u || first.handle() != second.handle())
        {
            return;
        }
        auto first_begin  = first.offset_bytes();
        auto first_end    = first_begin + num_items * first.stride();
        auto second_begin = second.offset_bytes();
        auto second_end   = second_begin + num_items * second.stride();
        LUISA_ASSERT(first_end <= second_begin || second_end <= first_begin,
                     "DevicePartition three-way input and output ranges must not overlap.");
    }

    template <typename... T>
    [[nodiscard]] static luisa::string _shader_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    void _write_zero_count(CommandList& cmdlist, BufferView<uint> num_selected_out)
    {
        using ZeroShader = Shader<1, Buffer<uint>>;
        if(!_zero_count_shader)
        {
            U<ZeroShader> shader = nullptr;
            lazy_compile(
                _device, shader, [](BufferVar<uint> output) noexcept { output.write(0u, 0u); });
            _zero_count_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<ZeroShader*>(&(*_zero_count_shader));
        cmdlist << (*shader)(num_selected_out).dispatch(1u);
    }

    void _write_zero_counts(CommandList& cmdlist, BufferView<uint> num_selected_out)
    {
        using ZeroShader = Shader<1, Buffer<uint>>;
        if(!_zero_counts_shader)
        {
            U<ZeroShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [](BufferVar<uint> output) noexcept { output.write(dispatch_id().x, 0u); });
            _zero_counts_shader = std::move(shader);
        }
        auto shader = reinterpret_cast<ZeroShader*>(&(*_zero_counts_shader));
        cmdlist << (*shader)(num_selected_out).dispatch(2u);
    }

    template <NumericT FlagT>
    void _dispatch_flags(CommandList& cmdlist, BufferView<FlagT> selection_flags, BufferView<uint> flags, size_t num_items)
    {
        using FlagShader = Shader<1, Buffer<FlagT>, Buffer<uint>>;
        auto key         = _shader_key<FlagT>("partition_flags");
        auto shader_it   = _flag_shaders.find(key);
        if(shader_it == _flag_shaders.end())
        {
            U<FlagShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [](BufferVar<FlagT> input, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             output.write(index, select(0u, 1u, input.read(index) != FlagT{}));
                         });
            _flag_shaders.try_emplace(key, std::move(shader));
            shader_it = _flag_shaders.find(key);
        }
        auto shader = reinterpret_cast<FlagShader*>(&(*shader_it->second));
        cmdlist << (*shader)(selection_flags, flags).dispatch(num_items);
    }

    template <NumericT T>
    void _dispatch_predicate(CommandList&             cmdlist,
                             BufferView<T>            input,
                             BufferView<uint>         flags,
                             size_t                   num_items,
                             const Callable<bool(T)>& predicate)
    {
        using PredicateShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key              = _shader_key<T>("partition_predicate");
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = _predicate_shaders.find(key);
        if(shader_it == _predicate_shaders.end())
        {
            U<PredicateShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [predicate](BufferVar<T> input_buffer, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             output.write(index, select(0u, 1u, predicate(input_buffer.read(index))));
                         });
            _predicate_shaders.try_emplace(key, std::move(shader));
            shader_it = _predicate_shaders.find(key);
        }
        auto shader = reinterpret_cast<PredicateShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, flags).dispatch(num_items);
    }

    template <NumericT T>
    void _dispatch_three_way_flags(CommandList&             cmdlist,
                                   BufferView<T>            input,
                                   BufferView<uint>         flags,
                                   size_t                   num_items,
                                   const Callable<bool(T)>& select_first,
                                   const Callable<bool(T)>& select_second,
                                   bool                     second_pass)
    {
        using FlagShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key = _shader_key<T>(second_pass ? "partition_three_way_second" : "partition_three_way_first");
        key += luisa::format("+{}+{}", select_first.function().hash(), select_second.function().hash());
        auto shader_it = _three_way_flag_shaders.find(key);
        if(shader_it == _three_way_flag_shaders.end())
        {
            U<FlagShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [select_first, select_second, second_pass](BufferVar<T> input_buffer,
                                                                    BufferVar<uint> flags_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index          = dispatch_id().x;
                             auto value          = input_buffer.read(index);
                             Bool first_selected = select_first(value);
                             Bool selected       = first_selected;
                             if(second_pass)
                             {
                                 selected = !first_selected & select_second(value);
                             }
                             flags_buffer.write(index, select(0u, 1u, selected));
                         });
            _three_way_flag_shaders.try_emplace(key, std::move(shader));
            shader_it = _three_way_flag_shaders.find(key);
        }
        auto shader = reinterpret_cast<FlagShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, flags).dispatch(num_items);
    }

    template <NumericT T>
    void _dispatch_predicate_scatter(CommandList&             cmdlist,
                                     BufferView<T>            input,
                                     const Callable<bool(T)>& predicate,
                                     BufferView<uint>         positions,
                                     BufferView<T>            output,
                                     BufferView<uint>         num_selected_out,
                                     size_t                   num_items)
    {
        using ScatterShader = Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, uint>;
        auto key            = _shader_key<T>("partition_predicate_scatter");
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = _predicate_scatter_shaders.find(key);
        if(shader_it == _predicate_scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [predicate](BufferVar<T>    input_buffer,
                                     BufferVar<T>    output_buffer,
                                     BufferVar<uint> positions_buffer,
                                     BufferVar<uint> count_output,
                                     UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             UInt selected = select(0u, 1u, predicate(input_buffer.read(index)));
                             UInt selected_before = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 output_buffer.write(selected_before, input_buffer.read(index));
                             }
                             $else
                             {
                                 UInt unselected_before = index - selected_before;
                                 output_buffer.write(num_items_var - 1u - unselected_before,
                                                     input_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output.write(0u, selected_before + selected);
                             };
                         });
            _predicate_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = _predicate_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist
            << (*shader)(input, output, positions, num_selected_out, static_cast<uint>(num_items)).dispatch(num_items);
    }

    template <NumericT T, NumericT FlagT>
    void _dispatch_flag_scatter(CommandList&      cmdlist,
                                BufferView<T>     input,
                                BufferView<FlagT> flags,
                                BufferView<uint>  positions,
                                BufferView<T>     output,
                                BufferView<uint>  num_selected_out,
                                size_t            num_items)
    {
        using ScatterShader =
            Shader<1, Buffer<T>, Buffer<FlagT>, Buffer<T>, Buffer<uint>, Buffer<uint>, uint>;
        auto key       = _shader_key<T, FlagT>("partition_flag_scatter");
        auto shader_it = _flag_scatter_shaders.find(key);
        if(shader_it == _flag_scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [](BufferVar<T>     input_buffer,
                            BufferVar<FlagT> flags_buffer,
                            BufferVar<T>     output_buffer,
                            BufferVar<uint>  positions_buffer,
                            BufferVar<uint>  count_output,
                            UInt             num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             UInt selected = select(0u, 1u, flags_buffer.read(index) != FlagT{});
                             UInt selected_before = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 output_buffer.write(selected_before, input_buffer.read(index));
                             }
                             $else
                             {
                                 UInt unselected_before = index - selected_before;
                                 output_buffer.write(num_items_var - 1u - unselected_before,
                                                     input_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output.write(0u, selected_before + selected);
                             };
                         });
            _flag_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = _flag_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, flags, output, positions, num_selected_out, static_cast<uint>(num_items))
                       .dispatch(num_items);
    }

    template <NumericT T>
    void _dispatch_scatter(CommandList&     cmdlist,
                           BufferView<T>    input,
                           BufferView<T>    output,
                           BufferView<uint> flags,
                           BufferView<uint> positions,
                           BufferView<uint> num_selected_out,
                           size_t           num_items)
    {
        using ScatterShader = Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        auto key       = _shader_key<T>("partition_scatter");
        auto shader_it = _scatter_shaders.find(key);
        if(shader_it == _scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [](BufferVar<T>    input_buffer,
                            BufferVar<T>    output_buffer,
                            BufferVar<uint> flags_buffer,
                            BufferVar<uint> positions_buffer,
                            BufferVar<uint> count_output,
                            UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index           = dispatch_id().x;
                             UInt selected        = flags_buffer.read(index);
                             UInt selected_before = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 output_buffer.write(selected_before, input_buffer.read(index));
                             }
                             $else
                             {
                                 UInt unselected_before = index - selected_before;
                                 output_buffer.write(num_items_var - 1u - unselected_before,
                                                     input_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output.write(0u, selected_before + selected);
                             };
                         });
            _scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = _scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, output, flags, positions, num_selected_out, static_cast<uint>(num_items))
                       .dispatch(num_items);
    }

    template <NumericT T>
    void _dispatch_three_way_scatter(CommandList&             cmdlist,
                                     BufferView<T>            input,
                                     BufferView<T>            first_output,
                                     BufferView<T>            second_output,
                                     BufferView<T>            unselected_output,
                                     BufferView<uint>         first_positions,
                                     BufferView<uint>         second_positions,
                                     BufferView<uint>         num_selected_out,
                                     size_t                   num_items,
                                     const Callable<bool(T)>& select_first,
                                     const Callable<bool(T)>& select_second)
    {
        using ScatterShader =
            Shader<1, Buffer<T>, Buffer<T>, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        auto key = _shader_key<T>("partition_three_way_scatter");
        key += luisa::format("+{}+{}", select_first.function().hash(), select_second.function().hash());
        auto shader_it = _three_way_scatter_shaders.find(key);
        if(shader_it == _three_way_scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(_device,
                         shader,
                         [select_first, select_second](BufferVar<T>    input_buffer,
                                                       BufferVar<T>    first_output_buffer,
                                                       BufferVar<T>    second_output_buffer,
                                                       BufferVar<T>    unselected_output_buffer,
                                                       BufferVar<uint> first_positions_buffer,
                                                       BufferVar<uint> second_positions_buffer,
                                                       BufferVar<uint> count_output,
                                                       UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index           = dispatch_id().x;
                             UInt last_index      = num_items_var - 1u;
                             auto value           = input_buffer.read(index);
                             auto last_value      = input_buffer.read(last_index);
                             Bool first_selected  = select_first(value);
                             Bool second_selected = !first_selected & select_second(value);
                             Bool last_first      = select_first(last_value);
                             Bool last_second     = !last_first & select_second(last_value);
                             UInt first_count =
                                 first_positions_buffer.read(last_index) + select(0u, 1u, last_first);
                             UInt second_count =
                                 second_positions_buffer.read(last_index) + select(0u, 1u, last_second);
                             UInt first_before  = first_positions_buffer.read(index);
                             UInt second_before = second_positions_buffer.read(index);
                             $if(first_selected)
                             {
                                 first_output_buffer.write(first_before, value);
                             }
                             $else
                             {
                                 $if(second_selected)
                                 {
                                     second_output_buffer.write(second_before, value);
                                 }
                                 $else
                                 {
                                     UInt unselected_before = index - first_before - second_before;
                                     UInt unselected_count = num_items_var - first_count - second_count;
                                     unselected_output_buffer.write(unselected_count - 1u - unselected_before, value);
                                 };
                             };
                             $if(index == last_index)
                             {
                                 count_output.write(0u, first_count);
                                 count_output.write(1u, second_count);
                             };
                         });
            _three_way_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = _three_way_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input,
                             first_output,
                             second_output,
                             unselected_output,
                             first_positions,
                             second_positions,
                             num_selected_out,
                             static_cast<uint>(num_items))
                       .dispatch(num_items);
    }
};
}  // namespace luisa::parallel_primitive
