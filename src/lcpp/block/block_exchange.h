#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

template <typename T, size_t BLOCK_DIM_X, size_t ITEMS_PER_THREAD, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u, bool WARP_TIME_SLICING = false>
class BlockExchange : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);

  public:
    BlockExchange()
        : m_shared_mem(new SmemType<T>{BLOCK_ITEMS})
    {
    }

    explicit BlockExchange(SmemTypePtr<T> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    ~BlockExchange() = default;

    template <typename OutputArray>
    void BlockedToStriped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        exchange(input_items, output_items, true, false);
    }

    void BlockedToStriped(compute::ArrayVar<T, ITEMS_PER_THREAD>& items) const
    {
        BlockedToStriped(items, items);
    }

    template <typename OutputArray>
    void StripedToBlocked(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        exchange(input_items, output_items, false, true);
    }

    void StripedToBlocked(compute::ArrayVar<T, ITEMS_PER_THREAD>& items) const
    {
        StripedToBlocked(items, items);
    }

    template <typename OutputArray>
    void BlockedToWarpStriped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        warp_exchange(input_items, output_items, true);
    }

    void BlockedToWarpStriped(compute::ArrayVar<T, ITEMS_PER_THREAD>& items) const
    {
        BlockedToWarpStriped(items, items);
    }

    template <typename OutputArray>
    void WarpStripedToBlocked(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items, OutputArray& output_items) const
    {
        warp_exchange(input_items, output_items, false);
    }

    void WarpStripedToBlocked(compute::ArrayVar<T, ITEMS_PER_THREAD>& items) const
    {
        WarpStripedToBlocked(items, items);
    }

    template <typename OutputArray, typename RankArray>
    void ScatterToBlocked(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                          OutputArray&                                  output_items,
                          const RankArray&                              ranks) const
    {
        scatter(input_items, output_items, ranks, true, false);
    }

    template <typename RankArray>
    void ScatterToBlocked(compute::ArrayVar<T, ITEMS_PER_THREAD>& items, const RankArray& ranks) const
    {
        ScatterToBlocked(items, items, ranks);
    }

    template <typename OutputArray, typename RankArray>
    void ScatterToStriped(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                          OutputArray&                                  output_items,
                          const RankArray&                              ranks) const
    {
        scatter(input_items, output_items, ranks, false, false);
    }

    template <typename RankArray>
    void ScatterToStriped(compute::ArrayVar<T, ITEMS_PER_THREAD>& items, const RankArray& ranks) const
    {
        ScatterToStriped(items, items, ranks);
    }

    template <typename OutputArray, typename RankArray>
    void ScatterToStripedGuarded(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                                 OutputArray&                                  output_items,
                                 const RankArray&                              ranks) const
    {
        scatter(input_items, output_items, ranks, false, true);
    }

    template <typename RankArray>
    void ScatterToStripedGuarded(compute::ArrayVar<T, ITEMS_PER_THREAD>& items, const RankArray& ranks) const
    {
        ScatterToStripedGuarded(items, items, ranks);
    }

    template <typename OutputArray, typename RankArray, typename FlagArray>
    void ScatterToStripedFlagged(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                                 OutputArray&                                  output_items,
                                 const RankArray&                              ranks,
                                 const FlagArray&                              is_valid) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            $if(is_valid[item])
            {
                (*m_shared_mem)[ranks[item]] = input_items[item];
            };
        }
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            output_items[item] = (*m_shared_mem)[tid + compute::UInt(item * BLOCK_THREADS)];
        }
        compute::sync_block();
    }

    template <typename RankArray, typename FlagArray>
    void ScatterToStripedFlagged(compute::ArrayVar<T, ITEMS_PER_THREAD>& items,
                                 const RankArray&                        ranks,
                                 const FlagArray&                        is_valid) const
    {
        ScatterToStripedFlagged(items, items, ranks, is_valid);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    template <typename OutputArray>
    void exchange(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                  OutputArray&                                  output_items,
                  bool                                          input_blocked,
                  bool                                          output_blocked) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index    = input_blocked ?
                                         tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item) :
                                         tid + compute::UInt(item * BLOCK_THREADS);
            (*m_shared_mem)[index] = input_items[item];
        }
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = output_blocked ?
                                      tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item) :
                                      tid + compute::UInt(item * BLOCK_THREADS);
            output_items[item]  = (*m_shared_mem)[index];
        }
        compute::sync_block();
    }

    template <typename OutputArray>
    void warp_exchange(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                       OutputArray&                                  output_items,
                       bool                                          input_blocked) const
    {
        constexpr size_t WARP_THREADS = details::WARP_SIZE;
        compute::UInt    tid          = linear_thread_id();
        compute::UInt    lane         = tid % compute::UInt(WARP_THREADS);
        compute::UInt    warp_offset =
            tid / compute::UInt(WARP_THREADS) * compute::UInt(WARP_THREADS * ITEMS_PER_THREAD);
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index =
                input_blocked ? warp_offset + lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item) :
                                warp_offset + lane + compute::UInt(item * WARP_THREADS);
            (*m_shared_mem)[index] = input_items[item];
        }
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index =
                input_blocked ? warp_offset + lane + compute::UInt(item * WARP_THREADS) :
                                warp_offset + lane * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            output_items[item] = (*m_shared_mem)[index];
        }
        compute::sync_block();
    }

    template <typename OutputArray, typename RankArray>
    void scatter(const compute::ArrayVar<T, ITEMS_PER_THREAD>& input_items,
                 OutputArray&                                  output_items,
                 const RankArray&                              ranks,
                 bool                                          output_blocked,
                 bool                                          guarded) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            if(guarded)
            {
                $if(ranks[item] < compute::UInt(BLOCK_ITEMS))
                {
                    (*m_shared_mem)[ranks[item]] = input_items[item];
                };
            }
            else
            {
                (*m_shared_mem)[ranks[item]] = input_items[item];
            }
        }
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = output_blocked ?
                                      tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item) :
                                      tid + compute::UInt(item * BLOCK_THREADS);
            output_items[item]  = (*m_shared_mem)[index];
        }
        compute::sync_block();
    }

    SmemTypePtr<T> m_shared_mem;
};
}  // namespace luisa::parallel_primitive
