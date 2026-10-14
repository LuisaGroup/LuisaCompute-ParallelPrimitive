#pragma once

#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_exchange.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

enum class WarpStoreAlgorithm
{
    WARP_STORE_DIRECT,
    WARP_STORE_STRIPED,
    WARP_STORE_VECTORIZE,
    WARP_STORE_TRANSPOSE,
};

template <typename T, size_t ITEMS_PER_THREAD, WarpStoreAlgorithm ALGORITHM = WarpStoreAlgorithm::WARP_STORE_DIRECT, size_t LOGICAL_WARP_THREADS = details::WARP_SIZE>
class WarpStore : public LuisaModule
{
    static_assert(LOGICAL_WARP_THREADS > 0u);
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1u)) == 0u);
    static_assert(LOGICAL_WARP_THREADS <= details::WARP_SIZE);

  public:
    void Store(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
               const compute::BufferVar<T>&                  output,
               compute::UInt                                 warp_item_start) const
    {
        Store(items, output, warp_item_start, compute::UInt(LOGICAL_WARP_THREADS * ITEMS_PER_THREAD));
    }

    void Store(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
               const compute::BufferVar<T>&                  output,
               compute::UInt                                 warp_item_start,
               compute::UInt                                 valid_items) const
    {
        if constexpr(ALGORITHM == WarpStoreAlgorithm::WARP_STORE_STRIPED)
        {
            store_striped(items, output, warp_item_start, valid_items);
        }
        else if constexpr(ALGORITHM == WarpStoreAlgorithm::WARP_STORE_TRANSPOSE)
        {
            compute::ArrayVar<T, ITEMS_PER_THREAD> striped_items;
            WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE>()
                .BlockedToStriped(items, striped_items);
            store_striped(striped_items, output, warp_item_start, valid_items);
        }
        else
        {
            // LuisaCompute lets each backend choose vectorized memory instructions. The
            // WARP_STORE_VECTORIZE policy therefore shares the blocked access semantics here.
            store_blocked(items, output, warp_item_start, valid_items);
        }
    }

  private:
    static void store_blocked(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                              const compute::BufferVar<T>&                  output,
                              compute::UInt                                 warp_item_start,
                              compute::UInt                                 valid_items)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(i);
            $if(item < valid_items)
            {
                output.write(warp_item_start + item, items[i]);
            };
        }
    }

    static void store_striped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                              const compute::BufferVar<T>&                  output,
                              compute::UInt                                 warp_item_start,
                              compute::UInt                                 valid_items)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane + compute::UInt(i * LOGICAL_WARP_THREADS);
            $if(item < valid_items)
            {
                output.write(warp_item_start + item, items[i]);
            };
        }
    }
};

}  // namespace luisa::parallel_primitive
