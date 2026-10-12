/*
 * @Author: Ligo 
 * @Date: 2025-10-14 16:49:47 
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 14:18:26
 */


#pragma once
#include <luisa/dsl/var.h>
#include <luisa/dsl/sugar.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_exchange.h>

namespace luisa::parallel_primitive
{

enum class BlockStoreAlgorithm
{
    BLOCK_STORE_DIRECT,
    BLOCK_STORE_STRIPED,
    BLOCK_STORE_VECTORIZE,
    BLOCK_STORE_TRANSPOSE,
    BLOCK_STORE_WARP_TRANSPOSE,
    BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
};


template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::BufferVar<T>&                      block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items)
{
    compute::UInt tid         = linear_tid % compute::UInt(WARP_SIZE);
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        block_itr.write(thread_offset + (item * compute::UInt(WARP_SIZE)), items[item]);
    }
}

template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::ByteBufferVar&                     block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items)
{
    compute::UInt tid         = linear_tid % compute::UInt(WARP_SIZE);
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        block_itr.write((thread_offset + (item * compute::UInt(WARP_SIZE))) * compute::UInt(sizeof(T)),
                        items[item]);
    }
}

template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::BufferVar<T>&                      block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items,
                            compute::UInt                               valid_item)
{
    compute::UInt tid         = linear_tid % compute::UInt(WARP_SIZE);
    compute::UInt wid         = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset = wid * compute::UInt(WARP_SIZE * ItemsPerThread);

    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        $if(warp_offset + tid + (item * compute::UInt(WARP_SIZE)) < valid_item)
        {
            block_itr.write(thread_offset + (item * compute::UInt(WARP_SIZE)), items[item]);
        };
    }
}

template <typename T, int ItemsPerThread, size_t WARP_SIZE = details::WARP_SIZE>
void StoreDirectWarpStriped(compute::UInt                               linear_tid,
                            compute::ByteBufferVar&                     block_itr,
                            compute::UInt                               global_offset,
                            const compute::ArrayVar<T, ItemsPerThread>& items,
                            compute::UInt                               valid_item)
{
    compute::UInt tid           = linear_tid % compute::UInt(WARP_SIZE);
    compute::UInt wid           = linear_tid / compute::UInt(WARP_SIZE);
    compute::UInt warp_offset   = wid * compute::UInt(WARP_SIZE * ItemsPerThread);
    compute::UInt thread_offset = global_offset + warp_offset + tid;
    // Store directly in warp-striped order
    for(auto item = 0; item < ItemsPerThread; item++)
    {
        compute::UInt pos = warp_offset + tid + (item * compute::UInt(WARP_SIZE));
        $if(pos < valid_item)
        {
            block_itr.write((thread_offset + (item * compute::UInt(WARP_SIZE))) * compute::UInt(sizeof(T)),
                            items[item]);
        };
    }
}


template <typename Type4Byte, size_t BlockSize = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = 2, BlockStoreAlgorithm DefaultStoreAlgorithm = BlockStoreAlgorithm::BLOCK_STORE_DIRECT>
class BlockStore : public LuisaModule
{
  public:
    explicit BlockStore(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    BlockStore()
    {
        if constexpr(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE
                     || DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE
                     || DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED)
        {
            m_shared_mem = new SmemType<Type4Byte>{BlockSize * ITEMS_PER_THREAD};
        }
    }

    ~BlockStore() = default;

  public:
    void Store(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
               const compute::BufferVar<Type4Byte>&                  d_out,
               compute::UInt                                         block_item_start)
    {
        Store(thread_data, d_out, block_item_start, compute::UInt(BlockSize * ITEMS_PER_THREAD));
    }

    void Store(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& thread_data,
               const compute::BufferVar<Type4Byte>&                  d_out,
               compute::UInt                                         block_item_start,
               compute::UInt                                         block_item_end)
    {
        compute::set_block_size(BlockSize);
        compute::UInt tid = compute::thread_id().x;
        if constexpr(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_DIRECT
                     || DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE)
        {
            store_blocked(thread_data, d_out, block_item_start, block_item_end, tid);
        }
        else if constexpr(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_STRIPED)
        {
            store_striped(thread_data, d_out, block_item_start, block_item_end, tid);
        }
        else if constexpr(DefaultStoreAlgorithm == BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE)
        {
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> striped_items;
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD>(m_shared_mem).BlockedToStriped(thread_data, striped_items);
            store_striped(striped_items, d_out, block_item_start, block_item_end, tid);
        }
        else
        {
            static_assert(BlockSize % details::WARP_SIZE == 0u);
            compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> warp_striped_items;
            BlockExchange<Type4Byte, BlockSize, ITEMS_PER_THREAD>(m_shared_mem).BlockedToWarpStriped(thread_data, warp_striped_items);
            store_warp_striped(warp_striped_items, d_out, block_item_start, block_item_end, tid);
        }
    }

  private:
    static void store_blocked(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                              const compute::BufferVar<Type4Byte>&                  output,
                              compute::UInt                                         tile_start,
                              compute::UInt                                         valid_items,
                              compute::UInt                                         tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            $if(index < valid_items)
            {
                output.write(tile_start + index, items[item]);
            };
        }
    }

    static void store_striped(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                              const compute::BufferVar<Type4Byte>&                  output,
                              compute::UInt                                         tile_start,
                              compute::UInt                                         valid_items,
                              compute::UInt                                         tid)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid + compute::UInt(item * BlockSize);
            $if(index < valid_items)
            {
                output.write(tile_start + index, items[item]);
            };
        }
    }

    static void store_warp_striped(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& items,
                                   const compute::BufferVar<Type4Byte>&                  output,
                                   compute::UInt                                         tile_start,
                                   compute::UInt valid_items,
                                   compute::UInt tid)
    {
        compute::UInt lane        = tid % compute::UInt(details::WARP_SIZE);
        compute::UInt warp_offset = tid / compute::UInt(details::WARP_SIZE)
                                    * compute::UInt(details::WARP_SIZE * ITEMS_PER_THREAD);
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = warp_offset + lane + compute::UInt(item * details::WARP_SIZE);
            $if(index < valid_items)
            {
                output.write(tile_start + index, items[item]);
            };
        }
    }

    SmemTypePtr<Type4Byte> m_shared_mem = nullptr;
};
}  // namespace luisa::parallel_primitive
