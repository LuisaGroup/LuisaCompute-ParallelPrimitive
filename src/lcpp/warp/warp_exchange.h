#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

enum class WarpExchangeAlgorithm
{
    WARP_EXCHANGE_SMEM,
    WARP_EXCHANGE_SHUFFLE,
};

template <typename T, size_t ITEMS_PER_THREAD, size_t LOGICAL_WARP_THREADS = details::WARP_SIZE, WarpExchangeAlgorithm ALGORITHM = WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE>
class WarpExchange : public LuisaModule
{
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(LOGICAL_WARP_THREADS > 0u);
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1u)) == 0u);
    static_assert(LOGICAL_WARP_THREADS <= details::WARP_SIZE);
    static_assert(ALGORITHM == WarpExchangeAlgorithm::WARP_EXCHANGE_SMEM
                  || ALGORITHM == WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE);

  public:
    template <typename OutputArray>
    void BlockedToStriped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source_items;
        copy_items(input_items, source_items);

        compute::UInt physical_lane = compute::warp_lane_id();
        compute::UInt warp_base     = logical_warp_base(physical_lane);
        compute::UInt lane          = physical_lane - warp_base;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt striped_index = lane + compute::UInt(item * LOGICAL_WARP_THREADS);
            compute::UInt source_lane   = striped_index / compute::UInt(ITEMS_PER_THREAD);
            compute::UInt source_item   = striped_index % compute::UInt(ITEMS_PER_THREAD);
            output_items[item] = read_selected_item(source_items, warp_base, source_lane, source_item);
        }
    }

    template <typename OutputArray>
    void StripedToBlocked(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source_items;
        copy_items(input_items, source_items);

        compute::UInt physical_lane = compute::warp_lane_id();
        compute::UInt warp_base     = logical_warp_base(physical_lane);
        compute::UInt lane          = physical_lane - warp_base;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt blocked_index = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            compute::UInt source_lane = blocked_index % compute::UInt(LOGICAL_WARP_THREADS);
            compute::UInt source_item = blocked_index / compute::UInt(LOGICAL_WARP_THREADS);
            output_items[item] = read_selected_item(source_items, warp_base, source_lane, source_item);
        }
    }

    template <typename RankArray>
    void ScatterToStriped(compute::ArrayVar<T, ITEMS_PER_THREAD>& items, const RankArray& ranks) const
    {
        ScatterToStriped(items, items, ranks);
    }

    template <typename OutputArray, typename RankArray>
    void ScatterToStriped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                          OutputArray&                                  output_items,
                          const RankArray&                              ranks) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source_items;
        copy_items(input_items, source_items);
        using RankT = compute::expr_value_t<decltype(ranks[0])>;

        compute::UInt physical_lane = compute::warp_lane_id();
        compute::UInt warp_base     = logical_warp_base(physical_lane);
        compute::UInt lane          = physical_lane - warp_base;
        for(size_t output_item = 0u; output_item < ITEMS_PER_THREAD; ++output_item)
        {
            compute::UInt   output_rank = lane + compute::UInt(output_item * LOGICAL_WARP_THREADS);
            compute::Var<T> value       = compute::warp_read_lane(source_items[0], warp_base);

            for(size_t source_item = 0u; source_item < ITEMS_PER_THREAD; ++source_item)
            {
                $for(source_lane, compute::UInt(LOGICAL_WARP_THREADS))
                {
                    compute::UInt   physical_source = warp_base + source_lane;
                    compute::Var<T> candidate =
                        compute::warp_read_lane(source_items[source_item], physical_source);
                    compute::Var<RankT> candidate_rank =
                        compute::warp_read_lane(ranks[source_item], physical_source);
                    value = compute::select(value, candidate, candidate_rank == compute::Var<RankT>(output_rank));
                };
            }
            output_items[output_item] = value;
        }
    }

  private:
    static compute::UInt logical_warp_base(compute::UInt physical_lane)
    {
        return physical_lane / compute::UInt(LOGICAL_WARP_THREADS) * compute::UInt(LOGICAL_WARP_THREADS);
    }

    template <typename InputArray, typename OutputArray>
    static void copy_items(const InputArray& input_items, OutputArray& output_items)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            output_items[item] = input_items[item];
        }
    }

    static compute::Var<T> read_selected_item(const compute::ArrayVar<T, ITEMS_PER_THREAD>& source_items,
                                              compute::UInt warp_base,
                                              compute::UInt source_lane,
                                              compute::UInt source_item)
    {
        compute::Var<T> value = compute::warp_read_lane(source_items[0], warp_base + source_lane);
        for(size_t item = 1u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::Var<T> candidate = compute::warp_read_lane(source_items[item], warp_base + source_lane);
            value = compute::select(value, candidate, source_item == compute::UInt(item));
        }
        return value;
    }
};
}  // namespace luisa::parallel_primitive
