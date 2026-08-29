/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <limits>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
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

    Device _device;
    ScanT  _scan;
    bool   _created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _flag_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _predicate_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> _scatter_shaders;
    luisa::shared_ptr<luisa::compute::Resource> _zero_count_shader;

  public:
    DevicePartition()  = default;
    ~DevicePartition() = default;

    void create(Device& device)
    {
        _device  = device;
        _created = true;
        _scan.create(device);
    }

    [[nodiscard]] static size_t GetTempStorageBytes(size_t num_items)
    {
        return 2u * num_items * sizeof(uint) + ScanT::template GetTempStorageBytes<uint>(num_items);
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

        auto [flags, positions, scan_temp] = _temp_views(temp_storage, num_items);
        _dispatch_flags(cmdlist, selection_flags, flags, num_items);
        _scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        _dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
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

        auto [flags, positions, scan_temp] = _temp_views(temp_storage, num_items);
        _dispatch_predicate(cmdlist, input, flags, num_items, predicate);
        _scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        _dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
    }

  private:
    struct TempViews
    {
        BufferView<uint> flags;
        BufferView<uint> positions;
        BufferView<uint> scan_temp;
    };

    [[nodiscard]] static TempViews _temp_views(BufferView<uint> temp_storage, size_t num_items)
    {
        size_t scan_uints = bytes_to_uint_count(ScanT::template GetTempStorageBytes<uint>(num_items));
        return {temp_storage.subview(0u, num_items),
                temp_storage.subview(num_items, num_items),
                temp_storage.subview(2u * num_items, scan_uints)};
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
};
}  // namespace luisa::parallel_primitive
