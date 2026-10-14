#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <array>
#include <cstddef>

namespace luisa::parallel_primitive
{

template <typename T, size_t BLOCK_DIM_X, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockShuffle : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static_assert(BLOCK_THREADS > 0u);

  public:
    BlockShuffle()
        : m_shared_mem(new SmemType<T>{BLOCK_THREADS})
    {
    }

    explicit BlockShuffle(SmemTypePtr<T> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    void Offset(const compute::Var<T>& input, compute::Var<T>& output, compute::Int distance = 1) const
    {
        compute::UInt tid    = linear_thread_id();
        (*m_shared_mem)[tid] = input;
        compute::sync_block();
        compute::Int source = compute::cast<int>(tid) + distance;
        $if(source >= 0 & source < compute::Int(BLOCK_THREADS))
        {
            output = (*m_shared_mem)[compute::cast<uint>(source)];
        };
        compute::sync_block();
    }

    void Rotate(const compute::Var<T>& input, compute::Var<T>& output, compute::UInt distance = 1u) const
    {
        compute::UInt tid    = linear_thread_id();
        (*m_shared_mem)[tid] = input;
        compute::sync_block();
        output = (*m_shared_mem)[(tid + distance) % compute::UInt(BLOCK_THREADS)];
        compute::sync_block();
    }

    template <size_t ITEMS_PER_THREAD>
    void Up(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
            compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output) const
    {
        compute::Var<T> block_suffix;
        up(input, output, block_suffix, false);
    }

    template <size_t ITEMS_PER_THREAD>
    void Up(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
            compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output,
            compute::Var<T>&                                     block_suffix) const
    {
        up(input, output, block_suffix, true);
    }

    template <size_t ITEMS_PER_THREAD>
    void Down(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
              compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output) const
    {
        compute::Var<T> block_prefix;
        down(input, output, block_prefix, false);
    }

    template <size_t ITEMS_PER_THREAD>
    void Down(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
              compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output,
              compute::Var<T>&                                     block_prefix) const
    {
        down(input, output, block_prefix, true);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    template <size_t ITEMS_PER_THREAD>
    void up(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
            compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output,
            compute::Var<T>&                                     block_suffix,
            bool                                                 return_suffix) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source_items;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            source_items[item] = input[item];
        }
        compute::UInt tid    = linear_thread_id();
        (*m_shared_mem)[tid] = source_items[ITEMS_PER_THREAD - 1u];
        compute::sync_block();
        for(size_t item = 1u; item < ITEMS_PER_THREAD; ++item)
        {
            output[item] = source_items[item - 1u];
        }
        $if(tid > 0u)
        {
            output[0] = (*m_shared_mem)[tid - 1u];
        };
        if(return_suffix)
        {
            block_suffix = (*m_shared_mem)[BLOCK_THREADS - 1u];
        }
        compute::sync_block();
    }

    template <size_t ITEMS_PER_THREAD>
    void down(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
              compute::Var<std::array<T, ITEMS_PER_THREAD>>&       output,
              compute::Var<T>&                                     block_prefix,
              bool                                                 return_prefix) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source_items;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            source_items[item] = input[item];
        }
        compute::UInt tid    = linear_thread_id();
        (*m_shared_mem)[tid] = source_items[0];
        compute::sync_block();
        for(size_t item = 0u; item + 1u < ITEMS_PER_THREAD; ++item)
        {
            output[item] = source_items[item + 1u];
        }
        $if(tid + 1u < compute::UInt(BLOCK_THREADS))
        {
            output[ITEMS_PER_THREAD - 1u] = (*m_shared_mem)[tid + 1u];
        };
        if(return_prefix)
        {
            block_prefix = (*m_shared_mem)[0];
        }
        compute::sync_block();
    }

    SmemTypePtr<T> m_shared_mem;
};

}  // namespace luisa::parallel_primitive
