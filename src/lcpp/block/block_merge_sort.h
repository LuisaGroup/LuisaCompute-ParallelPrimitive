#pragma once

#include <lcpp/block/block_exchange.h>
#include <lcpp/common/null_type.h>
#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

template <typename KeyT, size_t BLOCK_DIM_X, size_t ITEMS_PER_THREAD, typename ValueT = NullType, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockMergeSort : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<KeyT> keys;
    };
    using TempStorage = tempStorage;

    BlockMergeSort()
        : m_shared_keys(new SmemType<KeyT>{BLOCK_ITEMS})
    {
    }

    explicit BlockMergeSort(tempStorage storage)
        : m_shared_keys(storage.keys)
    {
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        sort_keys<false>(keys, compare_op);
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
              CompareOp                                  compare_op,
              compute::UInt                              valid_items,
              const compute::Var<KeyT>&                  out_of_bounds_default) const
    {
        fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        sort_keys<false>(keys, compare_op);
    }

    template <typename CompareOp>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              CompareOp                                    compare_op) const
    {
        sort_pairs<false>(keys, values, compare_op);
    }

    template <typename CompareOp, bool IS_LAST_TILE = true>
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              CompareOp                                    compare_op,
              compute::UInt                                valid_items,
              const compute::Var<KeyT>&                    out_of_bounds_default) const
    {
        if constexpr(IS_LAST_TILE)
        {
            fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        }
        sort_pairs<false>(keys, values, compare_op);
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

    template <typename CompareOp, bool IS_LAST_TILE = true>
    void StableSort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    CompareOp                                    compare_op,
                    compute::UInt                                valid_items,
                    const compute::Var<KeyT>&                    out_of_bounds_default) const
    {
        Sort<CompareOp, IS_LAST_TILE>(keys, values, compare_op, valid_items, out_of_bounds_default);
    }

    template <typename CompareOp>
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        sort_keys<true>(keys, compare_op);
    }

    template <typename CompareOp>
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              CompareOp                                  compare_op,
                              compute::UInt                              valid_items,
                              const compute::Var<KeyT>& out_of_bounds_default) const
    {
        fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        sort_keys<true>(keys, compare_op);
    }

    template <typename CompareOp>
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                              CompareOp                                    compare_op) const
    {
        sort_pairs<true>(keys, values, compare_op);
    }

    template <typename CompareOp, bool IS_LAST_TILE = true>
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                              CompareOp                                    compare_op,
                              compute::UInt                                valid_items,
                              const compute::Var<KeyT>& out_of_bounds_default) const
    {
        if constexpr(IS_LAST_TILE)
        {
            fill_invalid_keys(keys, valid_items, out_of_bounds_default);
        }
        sort_pairs<true>(keys, values, compare_op);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    static void fill_invalid_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                                  compute::UInt                              valid_items,
                                  const compute::Var<KeyT>&                  out_of_bounds_default)
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            keys[item] = compute::select(keys[item], out_of_bounds_default, index >= valid_items);
        }
    }

    template <typename CompareOp>
    void compute_ranks(const compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                       compute::ArrayVar<uint, ITEMS_PER_THREAD>&       ranks,
                       CompareOp                                        compare_op) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            m_shared_keys->write(tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item), keys[item]);
        }
        compute::sync_block();

        for(size_t target_item = 0u; target_item < ITEMS_PER_THREAD; ++target_item)
        {
            compute::UInt target_index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(target_item);
            compute::UInt rank = 0u;
            $for(source_index, compute::UInt(BLOCK_ITEMS))
            {
                compute::Var<KeyT> candidate        = m_shared_keys->read(source_index);
                compute::Bool      candidate_before = compare_op(candidate, keys[target_item]);
                compute::Bool      target_before    = compare_op(keys[target_item], candidate);
                compute::Bool      equivalent       = !(candidate_before | target_before);
                compute::Bool precedes = candidate_before | (equivalent & (source_index < target_index));
                rank += compute::select(0u, 1u, precedes);
            };
            ranks[target_item] = rank;
        }
        compute::sync_block();
    }

    template <bool STRIPED, typename CompareOp>
    void sort_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD> ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD> sorted_keys;
        compute_ranks(keys, ranks, compare_op);
        BlockExchange<KeyT, BLOCK_DIM_X, ITEMS_PER_THREAD, BLOCK_DIM_Y, BLOCK_DIM_Z> exchange;
        if constexpr(STRIPED)
        {
            exchange.ScatterToStriped(keys, sorted_keys, ranks);
        }
        else
        {
            exchange.ScatterToBlocked(keys, sorted_keys, ranks);
        }
        copy_items<KeyT>(sorted_keys, keys);
    }

    template <bool STRIPED, typename CompareOp>
    void sort_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    CompareOp                                    compare_op) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD>   ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD>   sorted_keys;
        compute::ArrayVar<ValueT, ITEMS_PER_THREAD> sorted_values;
        compute_ranks(keys, ranks, compare_op);

        BlockExchange<KeyT, BLOCK_DIM_X, ITEMS_PER_THREAD, BLOCK_DIM_Y, BLOCK_DIM_Z> key_exchange;
        if constexpr(STRIPED)
        {
            key_exchange.ScatterToStriped(keys, sorted_keys, ranks);
        }
        else
        {
            key_exchange.ScatterToBlocked(keys, sorted_keys, ranks);
        }
        BlockExchange<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, BLOCK_DIM_Y, BLOCK_DIM_Z> value_exchange;
        if constexpr(STRIPED)
        {
            value_exchange.ScatterToStriped(values, sorted_values, ranks);
        }
        else
        {
            value_exchange.ScatterToBlocked(values, sorted_values, ranks);
        }
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

  private:
    SmemTypePtr<KeyT> m_shared_keys;
};

}  // namespace luisa::parallel_primitive
