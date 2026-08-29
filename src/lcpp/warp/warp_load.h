#pragma once

#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_exchange.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

enum class WarpLoadAlgorithm
{
    WARP_LOAD_DIRECT,
    WARP_LOAD_STRIPED,
    WARP_LOAD_VECTORIZE,
    WARP_LOAD_TRANSPOSE,
};

template <typename T, size_t ITEMS_PER_THREAD, WarpLoadAlgorithm ALGORITHM = WarpLoadAlgorithm::WARP_LOAD_DIRECT, size_t LOGICAL_WARP_THREADS = details::WARP_SIZE>
class WarpLoad : public LuisaModule
{
    static_assert(LOGICAL_WARP_THREADS > 0u);
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1u)) == 0u);
    static_assert(LOGICAL_WARP_THREADS <= details::WARP_SIZE);

  public:
    void Load(const compute::BufferVar<T>&            input,
              compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
              compute::UInt                           warp_item_start) const
    {
        Load(input, items, warp_item_start, compute::UInt(LOGICAL_WARP_THREADS * ITEMS_PER_THREAD), T{});
    }

    void Load(const compute::BufferVar<T>&            input,
              compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
              compute::UInt                           warp_item_start,
              compute::UInt                           valid_items) const
    {
        if constexpr(ALGORITHM == WarpLoadAlgorithm::WARP_LOAD_STRIPED)
        {
            load_striped_guarded(input, items, warp_item_start, valid_items);
        }
        else if constexpr(ALGORITHM == WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE)
        {
            compute::ArrayVar<T, ITEMS_PER_THREAD> striped_items;
            for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                striped_items[i] = items[i];
            }
            load_striped_guarded(input, striped_items, warp_item_start, valid_items);
            WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE>()
                .StripedToBlocked(striped_items, items);
        }
        else
        {
            load_blocked_guarded(input, items, warp_item_start, valid_items);
        }
    }

    void Load(const compute::BufferVar<T>&            input,
              compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
              compute::UInt                           warp_item_start,
              compute::UInt                           valid_items,
              const compute::Var<T>&                  out_of_bounds_default) const
    {
        if constexpr(ALGORITHM == WarpLoadAlgorithm::WARP_LOAD_STRIPED)
        {
            load_striped(input, items, warp_item_start, valid_items, out_of_bounds_default);
        }
        else if constexpr(ALGORITHM == WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE)
        {
            compute::ArrayVar<T, ITEMS_PER_THREAD> striped_items;
            load_striped(input, striped_items, warp_item_start, valid_items, out_of_bounds_default);
            WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE>()
                .StripedToBlocked(striped_items, items);
        }
        else
        {
            // LuisaCompute lets each backend choose vectorized memory instructions. The
            // WARP_LOAD_VECTORIZE policy therefore shares the blocked access semantics here.
            load_blocked(input, items, warp_item_start, valid_items, out_of_bounds_default);
        }
    }

  private:
    static void load_blocked_guarded(const compute::BufferVar<T>&            input,
                                     compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                                     compute::UInt                           warp_item_start,
                                     compute::UInt                           valid_items)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(i);
            $if(item < valid_items)
            {
                items[i] = input.read(warp_item_start + item);
            };
        }
    }

    static void load_striped_guarded(const compute::BufferVar<T>&            input,
                                     compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                                     compute::UInt                           warp_item_start,
                                     compute::UInt                           valid_items)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane + compute::UInt(i * LOGICAL_WARP_THREADS);
            $if(item < valid_items)
            {
                items[i] = input.read(warp_item_start + item);
            };
        }
    }

    static void load_blocked(const compute::BufferVar<T>&            input,
                             compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                             compute::UInt                           warp_item_start,
                             compute::UInt                           valid_items,
                             const compute::Var<T>&                  out_of_bounds_default)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(i);
            $if(item < valid_items)
            {
                items[i] = input.read(warp_item_start + item);
            }
            $else
            {
                items[i] = out_of_bounds_default;
            };
        }
    }

    static void load_striped(const compute::BufferVar<T>&            input,
                             compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                             compute::UInt                           warp_item_start,
                             compute::UInt                           valid_items,
                             const compute::Var<T>&                  out_of_bounds_default)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t i = 0u; i < ITEMS_PER_THREAD; ++i)
        {
            compute::UInt item = lane + compute::UInt(i * LOGICAL_WARP_THREADS);
            $if(item < valid_items)
            {
                items[i] = input.read(warp_item_start + item);
            }
            $else
            {
                items[i] = out_of_bounds_default;
            };
        }
    }
};

}  // namespace luisa::parallel_primitive
