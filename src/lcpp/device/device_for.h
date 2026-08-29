/*
 * @Author: Ligo
 * @Date: 2025-09-19 23:06:17
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-08-30
 */
#pragma once

#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

class DeviceFor : public LuisaModule
{
  private:
    static constexpr uint BLOCK_SIZE = details::BLOCK_SIZE;

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_for_each_shaders;

  public:
    DeviceFor()  = default;
    ~DeviceFor() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept { return 0u; }

    template <NumericT T>
    void ForEachN(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        LUISA_ASSERT(m_created, "DeviceFor::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= data.size(),
                     "DeviceFor item count {} exceeds buffer size {}.",
                     num_items,
                     data.size());
        if(num_items == 0u)
        {
            return;
        }

        using ForEachShader = Shader<1, Buffer<T>>;
        luisa::string key{Type::of<T>()->description()};
        key += "+";
        key += luisa::format("{}", op.function().hash());

        auto shader_it = m_for_each_shaders.find(key);
        if(shader_it == m_for_each_shaders.end())
        {
            U<ForEachShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [op](BufferVar<T> buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt   index = dispatch_id().x;
                             Var<T> value = buffer.read(index);
                             op(value);
                             buffer.write(index, value);
                         });
            m_for_each_shaders.try_emplace(key, std::move(shader));
            shader_it = m_for_each_shaders.find(key);
        }

        auto shader = reinterpret_cast<ForEachShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFor shader.");
        cmdlist << (*shader)(data).dispatch(num_items);
    }

    template <NumericT T>
    void ForEach(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <NumericT T>
    void ForEach(CommandList& cmdlist, BufferView<T> data, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, data.size(), op);
    }

    template <NumericT T>
    void for_each(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <NumericT T>
    void for_each_n(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <NumericT T>
    void for_each_extents(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }
};
}  // namespace luisa::parallel_primitive
