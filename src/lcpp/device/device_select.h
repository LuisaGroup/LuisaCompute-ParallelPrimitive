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
class DeviceSelect : public LuisaModule
{
  private:
    using ScanT = DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;

    Device m_device;
    ScanT  m_scan;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_flag_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_predicate_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_unique_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_scatter_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_pair_scatter_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_in_place_shaders;
    luisa::shared_ptr<luisa::compute::Resource> m_zero_count_shader;

  public:
    DeviceSelect()  = default;
    ~DeviceSelect() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
        m_scan.create(device);
    }

    [[nodiscard]] static size_t GetTempStorageBytes(size_t num_items)
    {
        if(num_items == 0u)
        {
            return 0u;
        }
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
        validate_views(temp_storage, input, output, num_selected_out, num_items);
        LUISA_ASSERT(selection_flags.size() >= num_items, "Selection flag view is too small.");
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        auto [flags, positions, scan_temp] = temp_views(temp_storage, num_items);
        dispatch_flags(cmdlist, selection_flags, flags, num_items);
        m_scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
    }

    template <NumericT T, NumericT FlagT>
    void Flagged(CommandList&      cmdlist,
                 BufferView<uint>  temp_storage,
                 BufferView<T>     data,
                 BufferView<FlagT> selection_flags,
                 BufferView<uint>  num_selected_out,
                 size_t            num_items)
    {
        Callable<bool(FlagT)> predicate = [](Var<FlagT> flag) noexcept { return flag != FlagT{}; };
        dispatch_in_place_flagged(
            cmdlist, temp_storage, data, selection_flags, num_selected_out, num_items, predicate, "flagged");
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
        validate_views(temp_storage, input, output, num_selected_out, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        auto [flags, positions, scan_temp] = temp_views(temp_storage, num_items);
        dispatch_predicate(cmdlist, input, flags, num_items, predicate);
        m_scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
    }

    template <NumericT T>
    void If(CommandList&             cmdlist,
            BufferView<uint>         temp_storage,
            BufferView<T>            data,
            BufferView<uint>         num_selected_out,
            size_t                   num_items,
            const Callable<bool(T)>& predicate)
    {
        validate_in_place_views(temp_storage, data, num_selected_out, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }
        dispatch_in_place_if(cmdlist, data, num_selected_out, num_items, predicate);
    }

    template <NumericT T, NumericT FlagT>
    void FlaggedIf(CommandList&                 cmdlist,
                   BufferView<uint>             temp_storage,
                   BufferView<T>                input,
                   BufferView<FlagT>            selection_flags,
                   BufferView<T>                output,
                   BufferView<uint>             num_selected_out,
                   size_t                       num_items,
                   const Callable<bool(FlagT)>& predicate)
    {
        validate_views(temp_storage, input, output, num_selected_out, num_items);
        LUISA_ASSERT(selection_flags.size() >= num_items, "Selection flag view is too small.");
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        auto [flags, positions, scan_temp] = temp_views(temp_storage, num_items);
        dispatch_flag_predicate(cmdlist, selection_flags, flags, num_items, predicate);
        m_scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
    }

    template <NumericT T, NumericT FlagT>
    void FlaggedIf(CommandList&                 cmdlist,
                   BufferView<uint>             temp_storage,
                   BufferView<T>                data,
                   BufferView<FlagT>            selection_flags,
                   BufferView<uint>             num_selected_out,
                   size_t                       num_items,
                   const Callable<bool(FlagT)>& predicate)
    {
        dispatch_in_place_flagged(
            cmdlist, temp_storage, data, selection_flags, num_selected_out, num_items, predicate, "flagged_if");
    }

    template <NumericT T>
    void Unique(CommandList&     cmdlist,
                BufferView<uint> temp_storage,
                BufferView<T>    input,
                BufferView<T>    output,
                BufferView<uint> num_selected_out,
                size_t           num_items)
    {
        Callable<bool(T, T)> equality = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs == rhs; };
        Unique(cmdlist, temp_storage, input, output, num_selected_out, num_items, equality);
    }

    template <NumericT T>
    void Unique(CommandList& cmdlist, BufferView<uint> temp_storage, BufferView<T> data, BufferView<uint> num_selected_out, size_t num_items)
    {
        Callable<bool(T, T)> equality = [](Var<T> lhs, Var<T> rhs) noexcept { return lhs == rhs; };
        Unique(cmdlist, temp_storage, data, num_selected_out, num_items, equality);
    }

    template <NumericT T>
    void Unique(CommandList&                cmdlist,
                BufferView<uint>            temp_storage,
                BufferView<T>               data,
                BufferView<uint>            num_selected_out,
                size_t                      num_items,
                const Callable<bool(T, T)>& equality)
    {
        validate_in_place_views(temp_storage, data, num_selected_out, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }
        dispatch_in_place_unique(cmdlist, data, num_selected_out, num_items, equality);
    }

    template <NumericT T>
    void Unique(CommandList&                cmdlist,
                BufferView<uint>            temp_storage,
                BufferView<T>               input,
                BufferView<T>               output,
                BufferView<uint>            num_selected_out,
                size_t                      num_items,
                const Callable<bool(T, T)>& equality)
    {
        validate_views(temp_storage, input, output, num_selected_out, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        auto [flags, positions, scan_temp] = temp_views(temp_storage, num_items);
        dispatch_unique_flags(cmdlist, input, flags, num_items, equality);
        m_scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        dispatch_scatter(cmdlist, input, output, flags, positions, num_selected_out, num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    void UniqueByKey(CommandList&       cmdlist,
                     BufferView<uint>   temp_storage,
                     BufferView<KeyT>   keys_input,
                     BufferView<ValueT> values_input,
                     BufferView<KeyT>   keys_output,
                     BufferView<ValueT> values_output,
                     BufferView<uint>   num_selected_out,
                     size_t             num_items)
    {
        Callable<bool(KeyT, KeyT)> equality = [](Var<KeyT> lhs, Var<KeyT> rhs) noexcept
        { return lhs == rhs; };
        UniqueByKey(cmdlist, temp_storage, keys_input, values_input, keys_output, values_output, num_selected_out, num_items, equality);
    }

    template <NumericT KeyT, NumericT ValueT>
    void UniqueByKey(CommandList&                      cmdlist,
                     BufferView<uint>                  temp_storage,
                     BufferView<KeyT>                  keys_input,
                     BufferView<ValueT>                values_input,
                     BufferView<KeyT>                  keys_output,
                     BufferView<ValueT>                values_output,
                     BufferView<uint>                  num_selected_out,
                     size_t                            num_items,
                     const Callable<bool(KeyT, KeyT)>& equality)
    {
        validate_views(temp_storage, keys_input, keys_output, num_selected_out, num_items);
        LUISA_ASSERT(values_input.size() >= num_items, "Input value view is too small.");
        LUISA_ASSERT(values_output.size() >= num_items, "Output value view is too small.");
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        auto [flags, positions, scan_temp] = temp_views(temp_storage, num_items);
        dispatch_unique_flags(cmdlist, keys_input, flags, num_items, equality);
        m_scan.ExclusiveSum(cmdlist, scan_temp, flags, positions, num_items);
        dispatch_pair_scatter(
            cmdlist, keys_input, values_input, keys_output, values_output, flags, positions, num_selected_out, num_items);
    }

  private:
    struct TempViews
    {
        BufferView<uint> flags;
        BufferView<uint> positions;
        BufferView<uint> scan_temp;
    };

    [[nodiscard]] static TempViews temp_views(BufferView<uint> temp_storage, size_t num_items)
    {
        size_t scan_uints = bytes_to_uint_count(ScanT::template GetTempStorageBytes<uint>(num_items));
        return {temp_storage.subview(0u, num_items),
                temp_storage.subview(num_items, num_items),
                temp_storage.subview(2u * num_items, scan_uints)};
    }

    template <NumericT T>
    void validate_views(BufferView<uint> temp_storage,
                        BufferView<T>    input,
                        BufferView<T>    output,
                        BufferView<uint> num_selected_out,
                        size_t           num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceSelect::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceSelect.");
        LUISA_ASSERT(input.size() >= num_items, "Input view is too small.");
        LUISA_ASSERT(output.size() >= num_items, "Output view is too small.");
        LUISA_ASSERT(num_selected_out.size() >= 1u, "Selected-count output is empty.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetTempStorageBytes(num_items)),
                     "DeviceSelect temporary storage is too small.");
    }

    template <NumericT T>
    void validate_in_place_views(BufferView<uint> temp_storage,
                                 BufferView<T>    data,
                                 BufferView<uint> num_selected_out,
                                 size_t           num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceSelect::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceSelect.");
        LUISA_ASSERT(data.size() >= num_items, "In-place data view is too small.");
        LUISA_ASSERT(num_selected_out.size() >= 1u, "Selected-count output is empty.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(GetTempStorageBytes(num_items)),
                     "DeviceSelect temporary storage is too small.");
    }

    template <typename... T>
    [[nodiscard]] static luisa::string shader_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    void write_zero_count(CommandList& cmdlist, BufferView<uint> num_selected_out)
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
        cmdlist << (*shader)(num_selected_out).dispatch(1u);
    }

    template <NumericT FlagT>
    void dispatch_flags(CommandList& cmdlist, BufferView<FlagT> selection_flags, BufferView<uint> flags, size_t num_items)
    {
        using FlagShader = Shader<1, Buffer<FlagT>, Buffer<uint>>;
        auto key         = shader_key<FlagT>("select_flags");
        auto shader_it   = m_flag_shaders.find(key);
        if(shader_it == m_flag_shaders.end())
        {
            U<FlagShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<FlagT> input, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             output.write(index, select(0u, 1u, input.read(index) != FlagT{}));
                         });
            m_flag_shaders.try_emplace(key, std::move(shader));
            shader_it = m_flag_shaders.find(key);
        }
        auto shader = reinterpret_cast<FlagShader*>(&(*shader_it->second));
        cmdlist << (*shader)(selection_flags, flags).dispatch(num_items);
    }

    template <NumericT T>
    void dispatch_predicate(CommandList&             cmdlist,
                            BufferView<T>            input,
                            BufferView<uint>         flags,
                            size_t                   num_items,
                            const Callable<bool(T)>& predicate)
    {
        using PredicateShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key              = shader_key<T>("select_predicate");
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = m_predicate_shaders.find(key);
        if(shader_it == m_predicate_shaders.end())
        {
            U<PredicateShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate](BufferVar<T> input_buffer, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             output.write(index, select(0u, 1u, predicate(input_buffer.read(index))));
                         });
            m_predicate_shaders.try_emplace(key, std::move(shader));
            shader_it = m_predicate_shaders.find(key);
        }
        auto shader = reinterpret_cast<PredicateShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, flags).dispatch(num_items);
    }

    template <NumericT FlagT>
    void dispatch_flag_predicate(CommandList&                 cmdlist,
                                 BufferView<FlagT>            selection_flags,
                                 BufferView<uint>             flags,
                                 size_t                       num_items,
                                 const Callable<bool(FlagT)>& predicate)
    {
        using PredicateShader = Shader<1, Buffer<FlagT>, Buffer<uint>>;
        auto key              = shader_key<FlagT>("select_flag_predicate");
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = m_predicate_shaders.find(key);
        if(shader_it == m_predicate_shaders.end())
        {
            U<PredicateShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate](BufferVar<FlagT> input, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             output.write(index, select(0u, 1u, predicate(input.read(index))));
                         });
            m_predicate_shaders.try_emplace(key, std::move(shader));
            shader_it = m_predicate_shaders.find(key);
        }
        auto shader = reinterpret_cast<PredicateShader*>(&(*shader_it->second));
        cmdlist << (*shader)(selection_flags, flags).dispatch(num_items);
    }

    template <NumericT T, NumericT FlagT>
    void dispatch_in_place_flagged(CommandList&                 cmdlist,
                                   BufferView<uint>             temp_storage,
                                   BufferView<T>                data,
                                   BufferView<FlagT>            selection_flags,
                                   BufferView<uint>             num_selected_out,
                                   size_t                       num_items,
                                   const Callable<bool(FlagT)>& predicate,
                                   luisa::string_view           algorithm)
    {
        validate_in_place_views(temp_storage, data, num_selected_out, num_items);
        LUISA_ASSERT(selection_flags.size() >= num_items, "Selection flag view is too small.");
        validate_in_place_flag_overlap(data, selection_flags, num_items);
        if(num_items == 0u)
        {
            write_zero_count(cmdlist, num_selected_out);
            return;
        }

        using SelectShader = Shader<1, Buffer<T>, Buffer<FlagT>, Buffer<uint>, uint>;
        auto key           = shader_key<T, FlagT>(algorithm);
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = m_in_place_shaders.find(key);
        if(shader_it == m_in_place_shaders.end())
        {
            U<SelectShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate](BufferVar<T>     data_buffer,
                                     BufferVar<FlagT> flags_buffer,
                                     BufferVar<uint>  count_output,
                                     UInt             num_items_var) noexcept
                         {
                             set_block_size(details::WARP_SIZE);
                             $if(dispatch_id().x == 0u)
                             {
                                 UInt write_index = 0u;
                                 $for(index, 0u, num_items_var, 1u)
                                 {
                                     Var<T>     value = data_buffer.read(index);
                                     Var<FlagT> flag  = flags_buffer.read(index);
                                     $if(predicate(flag))
                                     {
                                         data_buffer.write(write_index, value);
                                         write_index += 1u;
                                     };
                                 };
                                 count_output.write(0u, write_index);
                             };
                         });
            m_in_place_shaders.try_emplace(key, std::move(shader));
            shader_it = m_in_place_shaders.find(key);
        }
        auto shader = reinterpret_cast<SelectShader*>(&(*shader_it->second));
        cmdlist << (*shader)(data, selection_flags, num_selected_out, static_cast<uint>(num_items)).dispatch(1u);
    }

    template <NumericT T, NumericT FlagT>
    static void validate_in_place_flag_overlap(BufferView<T> data, BufferView<FlagT> flags, size_t num_items)
    {
        if(num_items == 0u || data.handle() != flags.handle())
        {
            return;
        }
        auto data_begin  = data.offset_bytes();
        auto data_end    = data_begin + num_items * data.stride();
        auto flags_begin = flags.offset_bytes();
        auto flags_end   = flags_begin + num_items * flags.stride();
        auto exact_alias = data_begin == flags_begin && data.stride() == flags.stride();
        LUISA_ASSERT(exact_alias || data_end <= flags_begin || flags_end <= data_begin,
                     "In-place DeviceSelect data and flag ranges may only overlap by exact aliasing.");
    }

    template <NumericT T>
    void dispatch_in_place_if(CommandList&             cmdlist,
                              BufferView<T>            data,
                              BufferView<uint>         num_selected_out,
                              size_t                   num_items,
                              const Callable<bool(T)>& predicate)
    {
        using SelectShader = Shader<1, Buffer<T>, Buffer<uint>, uint>;
        auto key           = shader_key<T>("select_if_in_place");
        key += luisa::format("+{}", predicate.function().hash());
        auto shader_it = m_in_place_shaders.find(key);
        if(shader_it == m_in_place_shaders.end())
        {
            U<SelectShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate](BufferVar<T> data_buffer, BufferVar<uint> count_output, UInt num_items_var) noexcept
                         {
                             set_block_size(details::WARP_SIZE);
                             $if(dispatch_id().x == 0u)
                             {
                                 UInt write_index = 0u;
                                 $for(index, 0u, num_items_var, 1u)
                                 {
                                     Var<T> value = data_buffer.read(index);
                                     $if(predicate(value))
                                     {
                                         data_buffer.write(write_index, value);
                                         write_index += 1u;
                                     };
                                 };
                                 count_output.write(0u, write_index);
                             };
                         });
            m_in_place_shaders.try_emplace(key, std::move(shader));
            shader_it = m_in_place_shaders.find(key);
        }
        auto shader = reinterpret_cast<SelectShader*>(&(*shader_it->second));
        cmdlist << (*shader)(data, num_selected_out, static_cast<uint>(num_items)).dispatch(1u);
    }

    template <NumericT T>
    void dispatch_in_place_unique(CommandList&                cmdlist,
                                  BufferView<T>               data,
                                  BufferView<uint>            num_selected_out,
                                  size_t                      num_items,
                                  const Callable<bool(T, T)>& equality)
    {
        using UniqueShader = Shader<1, Buffer<T>, Buffer<uint>, uint>;
        auto key           = shader_key<T>("select_unique_in_place");
        key += luisa::format("+{}", equality.function().hash());
        auto shader_it = m_in_place_shaders.find(key);
        if(shader_it == m_in_place_shaders.end())
        {
            U<UniqueShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [equality](BufferVar<T> data_buffer, BufferVar<uint> count_output, UInt num_items_var) noexcept
                         {
                             set_block_size(details::WARP_SIZE);
                             $if(dispatch_id().x == 0u)
                             {
                                 Var<T> previous    = data_buffer.read(0u);
                                 UInt   write_index = 1u;
                                 $for(index, 1u, num_items_var, 1u)
                                 {
                                     Var<T> value = data_buffer.read(index);
                                     $if(!equality(previous, value))
                                     {
                                         data_buffer.write(write_index, value);
                                         write_index += 1u;
                                     };
                                     previous = value;
                                 };
                                 count_output.write(0u, write_index);
                             };
                         });
            m_in_place_shaders.try_emplace(key, std::move(shader));
            shader_it = m_in_place_shaders.find(key);
        }
        auto shader = reinterpret_cast<UniqueShader*>(&(*shader_it->second));
        cmdlist << (*shader)(data, num_selected_out, static_cast<uint>(num_items)).dispatch(1u);
    }

    template <NumericT T>
    void dispatch_unique_flags(CommandList&                cmdlist,
                               BufferView<T>               input,
                               BufferView<uint>            flags,
                               size_t                      num_items,
                               const Callable<bool(T, T)>& equality)
    {
        using UniqueShader = Shader<1, Buffer<T>, Buffer<uint>>;
        auto key           = shader_key<T>("select_unique");
        key += luisa::format("+{}", equality.function().hash());
        auto shader_it = m_unique_shaders.find(key);
        if(shader_it == m_unique_shaders.end())
        {
            U<UniqueShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [equality](BufferVar<T> input_buffer, BufferVar<uint> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             Bool selected = index == 0u;
                             $if(index > 0u)
                             {
                                 selected = !equality(input_buffer.read(index - 1u), input_buffer.read(index));
                             };
                             output.write(index, select(0u, 1u, selected));
                         });
            m_unique_shaders.try_emplace(key, std::move(shader));
            shader_it = m_unique_shaders.find(key);
        }
        auto shader = reinterpret_cast<UniqueShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, flags).dispatch(num_items);
    }

    template <NumericT T>
    void dispatch_scatter(CommandList&     cmdlist,
                          BufferView<T>    input,
                          BufferView<T>    output,
                          BufferView<uint> flags,
                          BufferView<uint> positions,
                          BufferView<uint> num_selected_out,
                          size_t           num_items)
    {
        using ScatterShader = Shader<1, Buffer<T>, Buffer<T>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        auto key       = shader_key<T>("select_scatter");
        auto shader_it = m_scatter_shaders.find(key);
        if(shader_it == m_scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<T>    input_buffer,
                            BufferVar<T>    output_buffer,
                            BufferVar<uint> flags_buffer,
                            BufferVar<uint> positions_buffer,
                            BufferVar<uint> count_output,
                            UInt            num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             UInt selected = flags_buffer.read(index);
                             UInt position = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 output_buffer.write(position, input_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output.write(0u, position + selected);
                             };
                         });
            m_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = m_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist << (*shader)(input, output, flags, positions, num_selected_out, static_cast<uint>(num_items))
                       .dispatch(num_items);
    }

    template <NumericT KeyT, NumericT ValueT>
    void dispatch_pair_scatter(CommandList&       cmdlist,
                               BufferView<KeyT>   keys_input,
                               BufferView<ValueT> values_input,
                               BufferView<KeyT>   keys_output,
                               BufferView<ValueT> values_output,
                               BufferView<uint>   flags,
                               BufferView<uint>   positions,
                               BufferView<uint>   num_selected_out,
                               size_t             num_items)
    {
        using ScatterShader =
            Shader<1, Buffer<KeyT>, Buffer<ValueT>, Buffer<KeyT>, Buffer<ValueT>, Buffer<uint>, Buffer<uint>, Buffer<uint>, uint>;
        auto key       = shader_key<KeyT, ValueT>("select_pair_scatter");
        auto shader_it = m_pair_scatter_shaders.find(key);
        if(shader_it == m_pair_scatter_shaders.end())
        {
            U<ScatterShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<KeyT>   keys_input_buffer,
                            BufferVar<ValueT> values_input_buffer,
                            BufferVar<KeyT>   keys_output_buffer,
                            BufferVar<ValueT> values_output_buffer,
                            BufferVar<uint>   flags_buffer,
                            BufferVar<uint>   positions_buffer,
                            BufferVar<uint>   count_output,
                            UInt              num_items_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index    = dispatch_id().x;
                             UInt selected = flags_buffer.read(index);
                             UInt position = positions_buffer.read(index);
                             $if(selected != 0u)
                             {
                                 keys_output_buffer.write(position, keys_input_buffer.read(index));
                                 values_output_buffer.write(position, values_input_buffer.read(index));
                             };
                             $if(index + 1u == num_items_var)
                             {
                                 count_output.write(0u, position + selected);
                             };
                         });
            m_pair_scatter_shaders.try_emplace(key, std::move(shader));
            shader_it = m_pair_scatter_shaders.find(key);
        }
        auto shader = reinterpret_cast<ScatterShader*>(&(*shader_it->second));
        cmdlist << (*shader)(keys_input,
                             values_input,
                             keys_output,
                             values_output,
                             flags,
                             positions,
                             num_selected_out,
                             static_cast<uint>(num_items))
                       .dispatch(num_items);
    }
};
}  // namespace luisa::parallel_primitive
