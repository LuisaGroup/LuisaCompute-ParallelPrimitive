#pragma once

#include <lcpp/common/null_type.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_exchange.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

template <typename KeyT, size_t ITEMS_PER_THREAD, size_t LOGICAL_WARP_THREADS = details::WARP_SIZE, typename ValueT = NullType>
class WarpMergeSort : public LuisaModule
{
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(LOGICAL_WARP_THREADS > 0u);
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1u)) == 0u);
    static_assert(LOGICAL_WARP_THREADS <= details::WARP_SIZE);

  public:
    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        sort_keys(keys, compare_op);
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
              CompareOp                                  compare_op,
              compute::UInt                              valid_items,
              const compute::Var<KeyT>&                  out_of_bounds_default) const
    {
        fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        sort_keys(keys, compare_op);
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              CompareOp                                    compare_op) const
    {
        sort_pairs(keys, values, compare_op);
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              CompareOp                                    compare_op,
              compute::UInt                                valid_items,
              const compute::Var<KeyT>&                    out_of_bounds_default) const
    {
        fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        sort_pairs(keys, values, compare_op);
    }

    template <typename CompareOp>
    void StableSort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        Sort(keys, compare_op);
    }

    template <typename CompareOp>
    void StableSort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                    CompareOp                                  compare_op,
                    compute::UInt                              valid_items,
                    const compute::Var<KeyT>&                  out_of_bounds_default) const
    {
        Sort(keys, compare_op, valid_items, out_of_bounds_default);
    }

    template <typename CompareOp>
    void StableSort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    CompareOp                                    compare_op) const
    {
        Sort(keys, values, compare_op);
    }

    template <typename CompareOp>
    void StableSort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    CompareOp                                    compare_op,
                    compute::UInt                                valid_items,
                    const compute::Var<KeyT>&                    out_of_bounds_default) const
    {
        Sort(keys, values, compare_op, valid_items, out_of_bounds_default);
    }

  private:
    static compute::UInt logical_warp_base(compute::UInt physical_lane)
    {
        return physical_lane / compute::UInt(LOGICAL_WARP_THREADS) * compute::UInt(LOGICAL_WARP_THREADS);
    }

    static void fill_invalid_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                                  compute::UInt                              valid_items,
                                  const compute::Var<KeyT>&                  out_of_bounds_default)
    {
        compute::UInt lane = compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt linear_item = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            keys[item] = compute::select(keys[item], out_of_bounds_default, linear_item >= valid_items);
        }
    }

    template <typename CompareOp>
    static void compute_ranks(const compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              compute::ArrayVar<uint, ITEMS_PER_THREAD>&       ranks,
                              CompareOp                                        compare_op)
    {
        compute::UInt physical_lane = compute::warp_lane_id();
        compute::UInt warp_base     = logical_warp_base(physical_lane);
        compute::UInt lane          = physical_lane - warp_base;

        for(size_t target_item = 0u; target_item < ITEMS_PER_THREAD; ++target_item)
        {
            compute::UInt target_index = lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(target_item);
            compute::UInt rank = 0u;

            for(size_t source_item = 0u; source_item < ITEMS_PER_THREAD; ++source_item)
            {
                $for(source_lane, compute::UInt(LOGICAL_WARP_THREADS))
                {
                    compute::Var<KeyT> candidate =
                        compute::warp_read_lane(keys[source_item], warp_base + source_lane);
                    compute::Bool candidate_before = compare_op(candidate, keys[target_item]);
                    compute::Bool target_before    = compare_op(keys[target_item], candidate);
                    compute::Bool equivalent       = !(candidate_before | target_before);
                    compute::UInt source_index =
                        source_lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(source_item);
                    compute::Bool precedes = candidate_before | (equivalent & (source_index < target_index));
                    rank += compute::select(compute::UInt(0u), compute::UInt(1u), precedes);
                };
            }
            ranks[target_item] = rank;
        }
    }

    template <typename ItemT>
    static void scatter_ranks_to_blocked(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input_items,
                                         compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& output_items,
                                         const compute::ArrayVar<uint, ITEMS_PER_THREAD>& ranks)
    {
        compute::ArrayVar<ItemT, ITEMS_PER_THREAD> striped_items;
        WarpExchange<ItemT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE> exchange;
        exchange.ScatterToStriped(input_items, striped_items, ranks);
        exchange.StripedToBlocked(striped_items, output_items);
    }

    template <typename CompareOp>
    static void sort_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op)
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD> ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD> sorted_keys;
        compute_ranks(keys, ranks, compare_op);
        scatter_ranks_to_blocked<KeyT>(keys, sorted_keys, ranks);
        copy_items<KeyT>(sorted_keys, keys);
    }

    template <typename CompareOp>
    static void sort_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                           compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                           CompareOp                                    compare_op)
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD>   ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD>   sorted_keys;
        compute::ArrayVar<ValueT, ITEMS_PER_THREAD> sorted_values;
        compute_ranks(keys, ranks, compare_op);
        scatter_ranks_to_blocked<KeyT>(keys, sorted_keys, ranks);
        scatter_ranks_to_blocked<ValueT>(values, sorted_values, ranks);
        copy_items<KeyT>(sorted_keys, keys);
        copy_items<ValueT>(sorted_values, values);
    }

    template <typename ItemT>
    static void copy_items(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input_items,
                           compute::ArrayVar<ItemT, ITEMS_PER_THREAD>&       output_items)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            output_items[item] = input_items[item];
        }
    }
};

}  // namespace luisa::parallel_primitive
