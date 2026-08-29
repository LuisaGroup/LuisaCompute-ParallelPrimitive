#pragma once

#include <lcpp/agent/radix_rank_sort_operations.h>
#include <lcpp/block/block_exchange.h>
#include <lcpp/common/null_type.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

template <NumericT KeyT, size_t BLOCK_DIM_X, size_t ITEMS_PER_THREAD, typename ValueT = NullType, size_t RADIX_BITS = 4u, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockRadixSort : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD;
    using UnsignedBits                    = typename Traits<KeyT>::UnsignedBits;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(RADIX_BITS > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<UnsignedBits> ordered_keys;
    };
    using TempStorage = tempStorage;

    BlockRadixSort()
        : m_ordered_keys(new SmemType<UnsignedBits>{BLOCK_ITEMS})
    {
    }

    explicit BlockRadixSort(tempStorage storage)
        : m_ordered_keys(storage.ordered_keys)
    {
    }

    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
              compute::UInt                              begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_keys<false, false>(keys, begin_bit, end_bit);
    }

    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              compute::UInt                                begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_pairs<false, false>(keys, values, begin_bit, end_bit);
    }

    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                        compute::UInt                              begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_keys<true, false>(keys, begin_bit, end_bit);
    }

    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                        compute::UInt                                begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_pairs<true, false>(keys, values, begin_bit, end_bit);
    }

    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              compute::UInt                              begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_keys<false, true>(keys, begin_bit, end_bit);
    }

    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                              compute::UInt                                begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_pairs<false, true>(keys, values, begin_bit, end_bit);
    }

    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                                        compute::UInt                              begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_keys<true, true>(keys, begin_bit, end_bit);
    }

    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                                        compute::UInt                                begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
    {
        sort_pairs<true, true>(keys, values, begin_bit, end_bit);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    static compute::Var<UnsignedBits> ordered_bits(const compute::Var<KeyT>& key)
    {
        return Traits<KeyT>::TwiddleIn(compute::as<UnsignedBits>(key));
    }

    static compute::Var<UnsignedBits> selected_bits(const compute::Var<KeyT>& key,
                                                    compute::UInt             begin_bit,
                                                    compute::UInt             end_bit)
    {
        constexpr auto             BIT_WIDTH = sizeof(KeyT) * 8u;
        compute::Var<UnsignedBits> selected  = UnsignedBits{0};
        $if(end_bit > begin_bit)
        {
            compute::UInt width = compute::min(end_bit - begin_bit, compute::UInt(BIT_WIDTH));
            compute::Var<UnsignedBits> mask =
                UnsignedBits(-1) >> compute::cast<UnsignedBits>(compute::UInt(BIT_WIDTH) - width);
            selected = (ordered_bits(key) >> compute::cast<UnsignedBits>(begin_bit)) & mask;
        };
        return selected;
    }

    template <bool DESCENDING>
    void compute_ranks(const compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                       compute::ArrayVar<uint, ITEMS_PER_THREAD>&       ranks,
                       compute::UInt                                    begin_bit,
                       compute::UInt                                    end_bit) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            m_ordered_keys->write(tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item),
                                  selected_bits(keys[item], begin_bit, end_bit));
        }
        compute::sync_block();

        for(size_t target_item = 0u; target_item < ITEMS_PER_THREAD; ++target_item)
        {
            compute::UInt target_index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(target_item);
            compute::Var<UnsignedBits> target = m_ordered_keys->read(target_index);
            compute::UInt              rank   = 0u;
            $for(source_index, compute::UInt(BLOCK_ITEMS))
            {
                compute::Var<UnsignedBits> candidate = m_ordered_keys->read(source_index);
                compute::Bool candidate_before = DESCENDING ? candidate > target : candidate < target;
                compute::Bool precedes =
                    candidate_before | ((candidate == target) & (source_index < target_index));
                rank += compute::select(0u, 1u, precedes);
            };
            ranks[target_item] = rank;
        }
        compute::sync_block();
    }

    template <bool STRIPED, typename ItemT>
    static void scatter(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input,
                        compute::ArrayVar<ItemT, ITEMS_PER_THREAD>&       output,
                        const compute::ArrayVar<uint, ITEMS_PER_THREAD>&  ranks)
    {
        BlockExchange<ItemT, BLOCK_DIM_X, ITEMS_PER_THREAD, BLOCK_DIM_Y, BLOCK_DIM_Z> exchange;
        if constexpr(STRIPED)
        {
            exchange.ScatterToStriped(input, output, ranks);
        }
        else
        {
            exchange.ScatterToBlocked(input, output, ranks);
        }
    }

    template <bool DESCENDING, bool STRIPED>
    void sort_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, compute::UInt begin_bit, compute::UInt end_bit) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD> ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD> sorted_keys;
        compute_ranks<DESCENDING>(keys, ranks, begin_bit, end_bit);
        scatter<STRIPED, KeyT>(keys, sorted_keys, ranks);
        copy_items<KeyT>(sorted_keys, keys);
    }

    template <bool DESCENDING, bool STRIPED>
    void sort_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    compute::UInt                                begin_bit,
                    compute::UInt                                end_bit) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD>   ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD>   sorted_keys;
        compute::ArrayVar<ValueT, ITEMS_PER_THREAD> sorted_values;
        compute_ranks<DESCENDING>(keys, ranks, begin_bit, end_bit);
        scatter<STRIPED, KeyT>(keys, sorted_keys, ranks);
        scatter<STRIPED, ValueT>(values, sorted_values, ranks);
        copy_items<KeyT>(sorted_keys, keys);
        copy_items<ValueT>(sorted_values, values);
    }

    template <typename ItemT>
    static void copy_items(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input,
                           compute::ArrayVar<ItemT, ITEMS_PER_THREAD>&       output)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            output[item] = input[item];
        }
    }

  private:
    SmemTypePtr<UnsignedBits> m_ordered_keys;
};

}  // namespace luisa::parallel_primitive
