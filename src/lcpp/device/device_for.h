/*
 * @Author: Ligo
 * @Date: 2025-09-19 23:06:17
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-08-30
 */
#pragma once

#include <array>
#include <limits>
#include <utility>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

enum class DeviceForLayout
{
    ROW_MAJOR,
    COLUMN_MAJOR
};

class DeviceFor : public LuisaModule
{
  private:
    static constexpr uint BLOCK_SIZE = details::BLOCK_SIZE;

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_for_each_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_for_each_copy_shaders;
    luisa::vector<U<Shader<1>>> m_index_shaders;

  public:
    DeviceFor()  = default;
    ~DeviceFor() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t) noexcept { return 0u; }

    template <typename T>
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

    template <typename T>
    void ForEach(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <typename T>
    void ForEach(CommandList& cmdlist, BufferView<T> data, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, data.size(), op);
    }

    template <typename T>
    void for_each(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <typename T>
    void for_each_n(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <typename T>
    void for_each_extents(CommandList& cmdlist, BufferView<T> data, size_t num_items, const Callable<void(T&)>& op)
    {
        ForEachN(cmdlist, data, num_items, op);
    }

    template <typename Op>
    void Bulk(CommandList& cmdlist, size_t shape, Op op)
    {
        LUISA_ASSERT(m_created, "DeviceFor::create must be called before dispatch.");
        LUISA_ASSERT(shape <= std::numeric_limits<uint>::max(),
                     "DeviceFor bulk shape exceeds the shader index range.");
        if(shape == 0u)
        {
            return;
        }

        U<Shader<1>> shader = nullptr;
        lazy_compile(m_device,
                     shader,
                     [op]() noexcept
                     {
                         set_block_size(BLOCK_SIZE);
                         op(dispatch_id().x);
                     });
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFor bulk shader.");
        cmdlist << (*shader)().dispatch(shape);
        m_index_shaders.emplace_back(std::move(shader));
    }

    template <typename T, typename Op>
    void ForEachCopyN(CommandList& cmdlist, BufferView<T> data, size_t num_items, Op op)
    {
        LUISA_ASSERT(m_created, "DeviceFor::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= data.size(),
                     "DeviceFor copy item count {} exceeds buffer size {}.",
                     num_items,
                     data.size());
        if(num_items == 0u)
        {
            return;
        }

        using ForEachCopyShader = Shader<1, Buffer<T>>;
        auto key = luisa::format("copy+{}+{}", Type::of<T>()->description(), op.function().hash());
        auto shader_it = m_for_each_copy_shaders.find(key);
        if(shader_it == m_for_each_copy_shaders.end())
        {
            U<ForEachCopyShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [op](BufferVar<T> buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             Var<T> value = buffer.read(dispatch_id().x);
                             op(value);
                         });
            m_for_each_copy_shaders.try_emplace(key, std::move(shader));
            shader_it = m_for_each_copy_shaders.find(key);
        }
        auto shader = reinterpret_cast<ForEachCopyShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFor copy shader.");
        cmdlist << (*shader)(data).dispatch(num_items);
    }

    template <typename T, typename Op>
    void ForEachCopy(CommandList& cmdlist, BufferView<T> data, Op op)
    {
        ForEachCopyN(cmdlist, data, data.size(), std::move(op));
    }

    template <size_t RANK, typename Op>
    void ForEachInExtents(CommandList& cmdlist, const std::array<size_t, RANK>& extents, Op op)
    {
        ForEachInLayout(cmdlist, extents, DeviceForLayout::ROW_MAJOR, std::move(op));
    }

    template <size_t RANK, typename Op>
    void ForEachInLayout(CommandList& cmdlist, const std::array<size_t, RANK>& extents, DeviceForLayout layout, Op op)
    {
        LUISA_ASSERT(m_created, "DeviceFor::create must be called before dispatch.");
        auto num_items = extent_size(extents);
        if(num_items == 0u)
        {
            return;
        }

        U<Shader<1>> shader = nullptr;
        lazy_compile(m_device,
                     shader,
                     [extents, layout, op]() noexcept
                     {
                         set_block_size(BLOCK_SIZE);
                         UInt                   linear    = dispatch_id().x;
                         UInt                   remaining = linear;
                         std::array<UInt, RANK> coordinates;
                         if(layout == DeviceForLayout::ROW_MAJOR)
                         {
                             for(size_t dimension = RANK; dimension > 0u; --dimension)
                             {
                                 auto index         = dimension - 1u;
                                 coordinates[index] = remaining % static_cast<uint>(extents[index]);
                                 remaining /= static_cast<uint>(extents[index]);
                             }
                         }
                         else
                         {
                             for(size_t dimension = 0u; dimension < RANK; ++dimension)
                             {
                                 coordinates[dimension] = remaining % static_cast<uint>(extents[dimension]);
                                 remaining /= static_cast<uint>(extents[dimension]);
                             }
                         }
                         invoke_extent_op(op, linear, coordinates, std::make_index_sequence<RANK>{});
                     });
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceFor extents shader.");
        cmdlist << (*shader)().dispatch(num_items);
        m_index_shaders.emplace_back(std::move(shader));
    }

  private:
    template <size_t RANK>
    [[nodiscard]] static size_t extent_size(const std::array<size_t, RANK>& extents)
    {
        size_t size = 1u;
        for(auto extent : extents)
        {
            LUISA_ASSERT(extent <= std::numeric_limits<uint>::max(),
                         "DeviceFor extent exceeds the shader index range.");
            if(extent == 0u)
            {
                return 0u;
            }
            LUISA_ASSERT(size <= std::numeric_limits<uint>::max() / extent,
                         "DeviceFor extents exceed the shader index range.");
            size *= extent;
        }
        return size;
    }

    template <typename Op, size_t RANK, size_t... I>
    static void invoke_extent_op(Op&                           op,
                                 const UInt&                   linear,
                                 const std::array<UInt, RANK>& coordinates,
                                 std::index_sequence<I...>)
    {
        op(linear, coordinates[I]...);
    }
};
}  // namespace luisa::parallel_primitive
