/*
 * @Author: Ligo 
 * @Date: 2025-10-14 14:01:20 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 13:58:39
 */
#pragma once

#include <cstddef>
#include <luisa/dsl/var.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/resource.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_exchange.h>

namespace luisa::parallel_primitive
{

enum class BlockLoadAlgorithm
{
    BLOCK_LOAD_DIRECT,
    BLOCK_LOAD_STRIPED,
    BLOCK_LOAD_VECTORIZE,
    BLOCK_LOAD_TRANSPOSE,
    BLOCK_LOAD_WARP_TRANSPOSE,
    BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
};

using namespace luisa::compute;

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::BufferVar<T>&          block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        dst_items[i] = block_src_it.read(tile_offset + linear_tid + i * compute::UInt(BlockThreads));
    }
}

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::ByteBufferVar&         block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = tile_offset + linear_tid + i * compute::UInt(BlockThreads);
        dst_items[i] = block_src_it.read<T>(src_pos * (uint)sizeof(T));
    }
}

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::BufferVar<T>&          block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items,
                       compute::UInt                         block_item_end)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = linear_tid + i * compute::UInt(BlockThreads);
        $if(src_pos < block_item_end)
        {
            src_pos += tile_offset;
            dst_items[i] = block_src_it.read(src_pos);
        };
    }
}

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::ByteBufferVar&         block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items,
                       compute::UInt                         block_item_end)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = linear_tid + i * compute::UInt(BlockThreads);
        $if(src_pos < block_item_end)
        {
            src_pos += tile_offset;
            dst_items[i] = block_src_it.read<T>(src_pos * (uint)sizeof(T));
        };
    }
}

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::BufferVar<T>&          block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items,
                       compute::UInt                         block_item_end,
                       compute::Var<T>                       default_value)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        dst_items[i] = default_value;
    }
    LoadDirectStriped<BlockThreads, T, ItemsPerThread>(linear_tid, block_src_it, tile_offset, dst_items, block_item_end);
}

template <uint BlockThreads, typename T, size_t ItemsPerThread>
void LoadDirectStriped(compute::UInt                         linear_tid,
                       const compute::ByteBufferVar&         block_src_it,
                       compute::UInt                         tile_offset,
                       compute::ArrayVar<T, ItemsPerThread>& dst_items,
                       compute::UInt                         block_item_end,
                       compute::Var<T>                       default_value)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        dst_items[i] = default_value;
    }
    LoadDirectStriped<BlockThreads, T, ItemsPerThread>(linear_tid, block_src_it, tile_offset, dst_items, block_item_end);
}


template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::BufferVar<T>&          block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items)
{
    compute::UInt tid;
    if constexpr(WARP_SIZE == details::WARP_SIZE) { tid = compute::warp_lane_id(); }
    else { tid = linear_tid % compute::UInt(WARP_SIZE); }
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    // Load directly in warp-striped order
    for(int i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = tile_offset + warp_offset + tid + (i * compute::UInt(WARP_SIZE));
        dst_items[i] = block_src_it.read(src_pos);
    }
}


template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::ByteBufferVar&         block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items)
{
    compute::UInt tid;
    if constexpr(WARP_SIZE == details::WARP_SIZE) { tid = compute::warp_lane_id(); }
    else { tid = linear_tid % compute::UInt(WARP_SIZE); }
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    // Load directly in warp-striped order
    for(int i = 0; i < ItemsPerThread; ++i)
    {
        UInt src_pos = tile_offset + warp_offset + tid + (i * compute::UInt(WARP_SIZE));
        dst_items[i] = block_src_it.read<T>(src_pos * (uint)sizeof(T));
    }
}

template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::BufferVar<T>&          block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items,
                           compute::UInt                         block_item_end)
{
    compute::UInt tid;
    if constexpr(WARP_SIZE == details::WARP_SIZE) { tid = compute::warp_lane_id(); }
    else { tid = linear_tid % compute::UInt(WARP_SIZE); }
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);


    for(auto i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = warp_offset + tid + (i * compute::UInt(WARP_SIZE));
        $if(src_pos < block_item_end)
        {
            src_pos += tile_offset;
            dst_items[i] = block_src_it.read(src_pos);
        };
    }
}

template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::ByteBufferVar&         block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items,
                           compute::UInt                         block_item_end)
{
    compute::UInt tid;
    if constexpr(WARP_SIZE == details::WARP_SIZE) { tid = compute::warp_lane_id(); }
    else { tid = linear_tid % compute::UInt(WARP_SIZE); }
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    for(auto i = 0; i < ItemsPerThread; i++)
    {
        UInt src_pos = warp_offset + tid + (i * compute::UInt(WARP_SIZE));
        $if(src_pos < block_item_end)
        {
            src_pos += tile_offset;
            dst_items[i] = block_src_it.read<T>(src_pos * (uint)sizeof(T));
        };
    }
}

template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::BufferVar<T>&          block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items,
                           compute::UInt                         block_item_end,
                           compute::Var<T>                       default_value)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        dst_items[i] = default_value;
    }
    LoadDirectWarpStriped<T, ItemsPerThread, WARP_SIZE>(linear_tid, block_src_it, tile_offset, dst_items, block_item_end);
}

template <typename T, size_t ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void LoadDirectWarpStriped(compute::UInt                         linear_tid,
                           const compute::ByteBufferVar&         block_src_it,
                           compute::UInt                         tile_offset,
                           compute::ArrayVar<T, ItemsPerThread>& dst_items,
                           compute::UInt                         block_item_end,
                           compute::Var<T>                       default_value)
{
    for(auto i = 0; i < ItemsPerThread; i++)
    {
        dst_items[i] = default_value;
    }
    LoadDirectWarpStriped<T, ItemsPerThread, WARP_SIZE>(linear_tid, block_src_it, tile_offset, dst_items, block_item_end);
}


template <typename Type4Byte,
          size_t BlockSize = details::BLOCK_SIZE,
          size_t ITEMS_PER_THREAD = 2,
          BlockLoadAlgorithm DefaultLoadAlgorithm = BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
          bool WARP_SYNC = false>
class BlockLoad : public LuisaModule
{
  public:
    explicit BlockLoad(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    BlockLoad()
    {
        if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE
                     || DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE
                     || DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize * ITEMS_PER_THREAD};
        }
    }

    ~BlockLoad() = default;

  public:
    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start)
    {
        if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE
                     || DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
        {
            static_assert(BlockSize % details::WARP_SIZE == 0u);
            compute::set_block_size(BlockSize);
            compute::Var<compute::uint> tid = compute::thread_id().x;
            compute::Var<compute::uint> base = block_item_start +
                tid / compute::UInt(details::WARP_SIZE) * compute::UInt(details::WARP_SIZE * ITEMS_PER_THREAD) +
                compute::warp_lane_id();
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> warp_striped_items;
            for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
            {
                warp_striped_items[item] = d_in.read(base + compute::UInt(item * details::WARP_SIZE));
            }
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD, 1u, 1u, false, WARP_SYNC>(m_shared_mem).WarpStripedToBlocked(
                warp_striped_items, thread_data);
        }
        else
        {
            Load(d_in, thread_data, block_item_start, compute::UInt(BlockSize * ITEMS_PER_THREAD), Type4Byte{});
        }
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start,
              compute::UInt                                   block_item_end)
    {
        compute::set_block_size(BlockSize);
        compute::UInt tid = compute::thread_id().x;

        if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_DIRECT
                     || DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE)
        {
            load_blocked_guarded(d_in, thread_data, block_item_start, block_item_end, tid);
        }
        else if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_STRIPED)
        {
            load_striped_guarded(d_in, thread_data, block_item_start, block_item_end, tid);
        }
        else if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE)
        {
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> striped_items;
            for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
            {
                striped_items[item] = thread_data[item];
            }
            load_striped_guarded(d_in, striped_items, block_item_start, block_item_end, tid);
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD, 1u, 1u, false, WARP_SYNC>(m_shared_mem).StripedToBlocked(
                striped_items, thread_data);
        }
        else
        {
            static_assert(BlockSize % details::WARP_SIZE == 0u);
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> warp_striped_items;
            for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
            {
                warp_striped_items[item] = thread_data[item];
            }
            load_warp_striped_guarded(d_in, warp_striped_items, block_item_start, block_item_end, tid);
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD, 1u, 1u, false, WARP_SYNC>(m_shared_mem).WarpStripedToBlocked(
                warp_striped_items, thread_data);
        }
    }

    void Load(const compute::BufferVar<Type4Byte>&            d_in,
              compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
              compute::UInt                                   block_item_start,
              compute::UInt                                   block_item_end,
              Var<Type4Byte>                                  default_value)
    {
        compute::set_block_size(BlockSize);
        compute::UInt tid = compute::thread_id().x;

        if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_DIRECT
                     || DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE)
        {
            load_blocked_default(d_in, thread_data, block_item_start, block_item_end, default_value, tid);
        }
        else if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_STRIPED)
        {
            load_striped_default(d_in, thread_data, block_item_start, block_item_end, default_value, tid);
        }
        else if constexpr(DefaultLoadAlgorithm == BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE)
        {
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> striped_items;
            load_striped_default(d_in, striped_items, block_item_start, block_item_end, default_value, tid);
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD, 1u, 1u, false, WARP_SYNC>(m_shared_mem).StripedToBlocked(
                striped_items, thread_data);
        }
        else
        {
            static_assert(BlockSize % details::WARP_SIZE == 0u);
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> warp_striped_items;
            load_warp_striped_default(d_in, warp_striped_items, block_item_start, block_item_end, default_value, tid);
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD, 1u, 1u, false, WARP_SYNC>(m_shared_mem).WarpStripedToBlocked(
                warp_striped_items, thread_data);
        }
    }


  private:
    static void load_blocked_guarded(const compute::BufferVar<Type4Byte>&            input,
                                     compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                     compute::UInt                                   tile_start,
                                     compute::UInt                                   valid_items,
                                     compute::UInt                                   tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            $if(index < valid_items)
            {
                items[item] = input.read(tile_start + index);
            };
        }
    }

    static void load_striped_guarded(const compute::BufferVar<Type4Byte>&            input,
                                     compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                     compute::UInt                                   tile_start,
                                     compute::UInt                                   valid_items,
                                     compute::UInt                                   tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid + compute::UInt(item * BlockSize);
            $if(index < valid_items)
            {
                items[item] = input.read(tile_start + index);
            };
        }
    }

    static void load_warp_striped_guarded(const compute::BufferVar<Type4Byte>&            input,
                                          compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                          compute::UInt tile_start,
                                          compute::UInt valid_items,
                                          compute::UInt tid)
    {
        compute::UInt lane        = compute::warp_lane_id();
        compute::UInt warp_offset = tid / compute::UInt(details::WARP_SIZE)
                                    * compute::UInt(details::WARP_SIZE * ITEMS_PER_THREAD);
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = warp_offset + lane + compute::UInt(item * details::WARP_SIZE);
            $if(index < valid_items)
            {
                items[item] = input.read(tile_start + index);
            };
        }
    }

    static void load_blocked_default(const compute::BufferVar<Type4Byte>&            input,
                                     compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                     compute::UInt                                   tile_start,
                                     compute::UInt                                   valid_items,
                                     const compute::Var<Type4Byte>&                  default_value,
                                     compute::UInt                                   tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            items[item] = default_value;
        }
        load_blocked_guarded(input, items, tile_start, valid_items, tid);
    }

    static void load_striped_default(const compute::BufferVar<Type4Byte>&            input,
                                     compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                     compute::UInt                                   tile_start,
                                     compute::UInt                                   valid_items,
                                     const compute::Var<Type4Byte>&                  default_value,
                                     compute::UInt                                   tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            items[item] = default_value;
        }
        load_striped_guarded(input, items, tile_start, valid_items, tid);
    }

    static void load_warp_striped_default(const compute::BufferVar<Type4Byte>&            input,
                                          compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                          compute::UInt                  tile_start,
                                          compute::UInt                  valid_items,
                                          const compute::Var<Type4Byte>& default_value,
                                          compute::UInt                  tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            items[item] = default_value;
        }
        load_warp_striped_guarded(input, items, tile_start, valid_items, tid);
    }

    SmemTypePtr<Type4Byte> m_shared_mem = nullptr;
};
}  // namespace luisa::parallel_primitive
