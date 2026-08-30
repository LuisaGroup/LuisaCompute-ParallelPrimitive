/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <lcpp/runtime/core.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceTransform : public LuisaModule
{
  private:
    static_assert(BLOCK_SIZE > 0u, "DeviceTransform block size must be positive.");

    template <typename... T>
    struct TypeList
    {
    };

    template <typename... A, typename... B>
    static auto concatenate(TypeList<A...>, TypeList<B...>) -> TypeList<A..., B...>;

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_fill_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_generate_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_transform_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_transform_if_shaders;

  public:
    DeviceTransform()  = default;
    ~DeviceTransform() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    template <typename T>
    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept
    {
        return 0u;
    }

    template <typename OutputT, typename ValueT>
    void Fill(CommandList& cmdlist, BufferView<uint>, BufferView<OutputT> output, size_t num_items, ValueT value)
    {
        validate_item_count(num_items);
        if(num_items == 0u)
        {
            return;
        }
        validate_view(output, num_items, "output");

        using FillShader = Shader<1, Buffer<OutputT>, ValueT>;
        auto key         = type_sequence_key<OutputT, ValueT>("device_transform_fill");
        auto shader_it   = m_fill_shaders.find(key);
        if(shader_it == m_fill_shaders.end())
        {
            U<FillShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<OutputT> output_buffer, Var<ValueT> value_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             if constexpr(std::is_same_v<OutputT, ValueT>)
                             {
                                 output_buffer.write(dispatch_id().x, value_var);
                             }
                             else
                             {
                                 output_buffer.write(dispatch_id().x, cast<OutputT>(value_var));
                             }
                         });
            m_fill_shaders.try_emplace(key, std::move(shader));
            shader_it = m_fill_shaders.find(key);
        }

        auto shader = reinterpret_cast<FillShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTransform::Fill shader.");
        cmdlist << (*shader)(output, value).dispatch(num_items);
    }

    template <typename OutputT, typename Generator>
    void Generate(CommandList& cmdlist, BufferView<uint>, BufferView<OutputT> output, size_t num_items, Generator generator)
    {
        validate_item_count(num_items);
        if(num_items == 0u)
        {
            return;
        }
        validate_view(output, num_items, "output");

        Callable generator_callable = [generator]() noexcept { return generator(); };
        using GenerateShader        = Shader<1, Buffer<OutputT>>;
        auto key                    = luisa::format("{}+{}",
                                 type_sequence_key<OutputT>("device_transform_generate"),
                                 generator_callable.function().hash());
        auto shader_it              = m_generate_shaders.find(key);
        if(shader_it == m_generate_shaders.end())
        {
            U<GenerateShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [generator](BufferVar<OutputT> output_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             auto generated = generator();
                             output_buffer.write(dispatch_id().x, assignable_value<OutputT>(generated));
                         });
            m_generate_shaders.try_emplace(key, std::move(shader));
            shader_it = m_generate_shaders.find(key);
        }

        auto shader = reinterpret_cast<GenerateShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTransform::Generate shader.");
        cmdlist << (*shader)(output).dispatch(num_items);
    }

    template <typename InputT, typename OutputT, typename TransformOp>
    void Transform(CommandList&        cmdlist,
                   BufferView<uint>    temp_storage,
                   BufferView<InputT>  input,
                   BufferView<OutputT> output,
                   size_t              num_items,
                   TransformOp         transform_op)
    {
        Transform(cmdlist, temp_storage, std::make_tuple(input), output, num_items, std::move(transform_op));
    }

    template <typename... InputT, typename OutputT, typename TransformOp>
    void Transform(CommandList& cmdlist,
                   BufferView<uint>,
                   std::tuple<BufferView<InputT>...> inputs,
                   BufferView<OutputT>               output,
                   size_t                            num_items,
                   TransformOp                       transform_op)
    {
        validate_item_count(num_items);
        if(num_items == 0u)
        {
            return;
        }
        validate_views(inputs, num_items, "input");
        validate_view(output, num_items, "output");
        dispatch_transform(cmdlist, inputs, output, num_items, std::move(transform_op));
    }

    template <typename... InputT, typename... OutputT, typename TransformOp>
    void Transform(CommandList& cmdlist,
                   BufferView<uint>,
                   std::tuple<BufferView<InputT>...>  inputs,
                   std::tuple<BufferView<OutputT>...> outputs,
                   size_t                             num_items,
                   TransformOp                        transform_op)
    {
        static_assert(sizeof...(OutputT) > 0u, "DeviceTransform requires at least one output.");
        validate_item_count(num_items);
        if(num_items == 0u)
        {
            return;
        }
        validate_views(inputs, num_items, "input");
        validate_views(outputs, num_items, "output");
        using AllTypes = decltype(concatenate(TypeList<OutputT...>{}, TypeList<InputT...>{}));
        dispatch_multi_transform(cmdlist, inputs, outputs, num_items, std::move(transform_op), AllTypes{});
    }

    template <typename InputT, typename OutputT, typename Predicate, typename TransformOp>
    void TransformIf(CommandList&        cmdlist,
                     BufferView<uint>    temp_storage,
                     BufferView<InputT>  input,
                     BufferView<OutputT> output,
                     size_t              num_items,
                     Predicate           predicate,
                     TransformOp         transform_op)
    {
        TransformIf(cmdlist, temp_storage, std::make_tuple(input), output, num_items, std::move(predicate), std::move(transform_op));
    }

    template <typename... InputT, typename OutputT, typename Predicate, typename TransformOp>
    void TransformIf(CommandList& cmdlist,
                     BufferView<uint>,
                     std::tuple<BufferView<InputT>...> inputs,
                     BufferView<OutputT>               output,
                     size_t                            num_items,
                     Predicate                         predicate,
                     TransformOp                       transform_op)
    {
        validate_item_count(num_items);
        if(num_items == 0u)
        {
            return;
        }
        validate_views(inputs, num_items, "input");
        validate_view(output, num_items, "output");
        dispatch_transform_if(cmdlist, inputs, output, num_items, std::move(predicate), std::move(transform_op));
    }

    /// LuisaCompute does not expose addressable references to typed buffer
    /// elements. Argument identity is therefore unobservable in its DSL, and
    /// this entry point has the same value semantics as Transform.
    template <typename InputT, typename OutputT, typename TransformOp>
    void TransformStableArgumentAddresses(CommandList&        cmdlist,
                                          BufferView<uint>    temp_storage,
                                          BufferView<InputT>  input,
                                          BufferView<OutputT> output,
                                          size_t              num_items,
                                          TransformOp         transform_op)
    {
        Transform(cmdlist, temp_storage, input, output, num_items, std::move(transform_op));
    }

    template <typename... InputT, typename OutputT, typename TransformOp>
    void TransformStableArgumentAddresses(CommandList&                      cmdlist,
                                          BufferView<uint>                  temp_storage,
                                          std::tuple<BufferView<InputT>...> inputs,
                                          BufferView<OutputT>               output,
                                          size_t                            num_items,
                                          TransformOp                       transform_op)
    {
        Transform(cmdlist, temp_storage, std::move(inputs), output, num_items, std::move(transform_op));
    }

  private:
    void validate_item_count(size_t num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceTransform::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(),
                     "DeviceTransform item count exceeds the shader index range.");
    }

    template <typename T>
    static void validate_view(BufferView<T> view, size_t num_items, luisa::string_view name)
    {
        LUISA_ASSERT(view.size() >= num_items, "DeviceTransform {} view is too small.", name);
    }

    template <typename... T, size_t... I>
    static void validate_views_impl(const std::tuple<BufferView<T>...>& views,
                                    size_t                              num_items,
                                    luisa::string_view                  name,
                                    std::index_sequence<I...>)
    {
        (validate_view(std::get<I>(views), num_items, name), ...);
    }

    template <typename... T>
    static void validate_views(const std::tuple<BufferView<T>...>& views, size_t num_items, luisa::string_view name)
    {
        validate_views_impl(views, num_items, name, std::index_sequence_for<T...>{});
    }

    template <typename... T>
    [[nodiscard]] static luisa::string type_sequence_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    template <typename... InputT, typename OutputT, typename TransformOp>
    void dispatch_transform(CommandList&                             cmdlist,
                            const std::tuple<BufferView<InputT>...>& inputs,
                            BufferView<OutputT>                      output,
                            size_t                                   num_items,
                            TransformOp                              transform_op)
    {
        Callable callable = [transform_op](Var<InputT>... values) noexcept
        { return transform_op(values...); };
        using TransformShader = Shader<1, Buffer<OutputT>, Buffer<InputT>...>;
        auto key              = luisa::format("{}+{}",
                                 type_sequence_key<OutputT, InputT...>("device_transform"),
                                 callable.function().hash());
        auto shader_it        = m_transform_shaders.find(key);
        if(shader_it == m_transform_shaders.end())
        {
            U<TransformShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [transform_op](BufferVar<OutputT> output_buffer, BufferVar<InputT>... input_buffers) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             auto input_values =
                                 std::tuple<Var<InputT>...>{Var<InputT>{input_buffers.read(index)}...};
                             auto result = std::apply(transform_op, input_values);
                             output_buffer.write(index, assignable_value<OutputT>(result));
                         });
            m_transform_shaders.try_emplace(key, std::move(shader));
            shader_it = m_transform_shaders.find(key);
        }

        auto shader = reinterpret_cast<TransformShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTransform::Transform shader.");
        std::apply([&](auto... input_views)
                   { cmdlist << (*shader)(output, input_views...).dispatch(num_items); },
                   inputs);
    }

    template <typename... InputT, typename OutputT, typename Predicate, typename TransformOp>
    void dispatch_transform_if(CommandList&                             cmdlist,
                               const std::tuple<BufferView<InputT>...>& inputs,
                               BufferView<OutputT>                      output,
                               size_t                                   num_items,
                               Predicate                                predicate,
                               TransformOp                              transform_op)
    {
        Callable predicate_callable = [predicate](Var<InputT>... values) noexcept
        { return predicate(values...); };
        Callable transform_callable = [transform_op](Var<InputT>... values) noexcept
        { return transform_op(values...); };
        using TransformIfShader = Shader<1, Buffer<OutputT>, Buffer<InputT>...>;
        auto key                = luisa::format("{}+{}+{}",
                                 type_sequence_key<OutputT, InputT...>("device_transform_if"),
                                 predicate_callable.function().hash(),
                                 transform_callable.function().hash());
        auto shader_it          = m_transform_if_shaders.find(key);
        if(shader_it == m_transform_if_shaders.end())
        {
            U<TransformIfShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [predicate, transform_op](BufferVar<OutputT> output_buffer, BufferVar<InputT>... input_buffers) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index = dispatch_id().x;
                             auto input_values =
                                 std::tuple<Var<InputT>...>{Var<InputT>{input_buffers.read(index)}...};
                             $if(std::apply(predicate, input_values))
                             {
                                 auto result = std::apply(transform_op, input_values);
                                 output_buffer.write(index, assignable_value<OutputT>(result));
                             };
                         });
            m_transform_if_shaders.try_emplace(key, std::move(shader));
            shader_it = m_transform_if_shaders.find(key);
        }

        auto shader = reinterpret_cast<TransformIfShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceTransform::TransformIf shader.");
        std::apply([&](auto... input_views)
                   { cmdlist << (*shader)(output, input_views...).dispatch(num_items); },
                   inputs);
    }

    template <typename Result>
    [[nodiscard]] static auto compose_multi_result(Result&& result)
    {
        if constexpr(requires { result.template get<0u>(); })
        {
            return std::forward<Result>(result);
        }
        else
        {
            return std::apply([](auto&&... components)
                              { return compose(std::forward<decltype(components)>(components)...); },
                              std::forward<Result>(result));
        }
    }

    template <size_t I, typename Result>
    [[nodiscard]] static auto result_component(Result& result)
    {
        if constexpr(requires { result.template get<I>(); })
        {
            return result.template get<I>();
        }
        else
        {
            return std::get<I>(result);
        }
    }

    template <typename OutputT, typename Value>
    [[nodiscard]] static auto assignable_value(Value&& value)
    {
        using ValueT = expr_value_t<Value>;
        if constexpr(std::is_same_v<OutputT, ValueT>)
        {
            return std::forward<Value>(value);
        }
        else
        {
            return cast<OutputT>(std::forward<Value>(value));
        }
    }

    template <size_t INPUT_OFFSET, typename... InputT, typename BufferTuple, size_t... I>
    [[nodiscard]] static auto read_inputs(BufferTuple& buffers, UInt index, TypeList<InputT...>, std::index_sequence<I...>)
    {
        return std::tuple<Var<InputT>...>{Var<InputT>{std::get<INPUT_OFFSET + I>(buffers).read(index)}...};
    }

    template <typename... OutputT, typename Result, size_t... I>
    [[nodiscard]] static auto materialize_results(Result& result, TypeList<OutputT...>, std::index_sequence<I...>)
    {
        return std::tuple<Var<OutputT>...>{
            Var<OutputT>{assignable_value<OutputT>(result_component<I>(result))}...};
    }

    template <typename BufferTuple, typename ResultTuple, size_t... I>
    static void write_results(BufferTuple& buffers, ResultTuple& results, UInt index, std::index_sequence<I...>)
    {
        (std::get<I>(buffers).write(index, std::get<I>(results)), ...);
    }

    template <typename... InputT, typename... OutputT, typename TransformOp, typename... AllT>
    void dispatch_multi_transform(CommandList&                              cmdlist,
                                  const std::tuple<BufferView<InputT>...>&  inputs,
                                  const std::tuple<BufferView<OutputT>...>& outputs,
                                  size_t                                    num_items,
                                  TransformOp                               transform_op,
                                  TypeList<AllT...>)
    {
        Callable callable = [transform_op](Var<InputT>... values) noexcept
        { return compose_multi_result(transform_op(values...)); };
        using TransformShader = Shader<1, Buffer<AllT>...>;
        auto key              = luisa::format("{}+{}",
                                 type_sequence_key<AllT...>("device_transform_multi_output"),
                                 callable.function().hash());
        auto shader_it        = m_transform_shaders.find(key);
        if(shader_it == m_transform_shaders.end())
        {
            U<TransformShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [transform_op](BufferVar<AllT>... all_buffers) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt index        = dispatch_id().x;
                             auto buffers      = std::forward_as_tuple(all_buffers...);
                             auto input_values = read_inputs<sizeof...(OutputT)>(
                                 buffers, index, TypeList<InputT...>{}, std::index_sequence_for<InputT...>{});
                             auto raw_result = std::apply(transform_op, input_values);
                             auto results    = materialize_results(raw_result,
                                                                TypeList<OutputT...>{},
                                                                std::index_sequence_for<OutputT...>{});
                             write_results(buffers, results, index, std::index_sequence_for<OutputT...>{});
                         });
            m_transform_shaders.try_emplace(key, std::move(shader));
            shader_it = m_transform_shaders.find(key);
        }

        auto shader = reinterpret_cast<TransformShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile multi-output DeviceTransform shader.");
        auto views = std::tuple_cat(outputs, inputs);
        std::apply([&](auto... buffer_views)
                   { cmdlist << (*shader)(buffer_views...).dispatch(num_items); },
                   views);
    }
};
}  // namespace luisa::parallel_primitive
