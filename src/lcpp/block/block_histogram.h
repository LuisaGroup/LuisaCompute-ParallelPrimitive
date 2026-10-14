#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

enum class BlockHistogramAlgorithm
{
    BLOCK_HISTO_SORT,
    BLOCK_HISTO_ATOMIC,
};

inline constexpr auto BLOCK_HISTO_SORT   = BlockHistogramAlgorithm::BLOCK_HISTO_SORT;
inline constexpr auto BLOCK_HISTO_ATOMIC = BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC;

template <typename T, size_t BLOCK_DIM_X, size_t ITEMS_PER_THREAD, size_t BINS, BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_SORT, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockHistogram : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(BINS > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<T> items;
    };
    using TempStorage = tempStorage;

    BlockHistogram()
        : m_shared_items(new SmemType<T>{ALGORITHM == BLOCK_HISTO_SORT ? BLOCK_ITEMS : 1u})
    {
    }

    explicit BlockHistogram(tempStorage storage)
        : m_shared_items(storage.items)
    {
    }

    template <typename CounterT>
    void InitHistogram(SmemTypePtr<CounterT> histogram) const
    {
        compute::UInt tid = linear_thread_id();
        $for(bin, tid, compute::UInt(BINS), compute::UInt(BLOCK_THREADS))
        {
            histogram->write(bin, CounterT{});
        };
    }

    template <typename CounterT>
    void Init(SmemTypePtr<CounterT> histogram) const
    {
        InitHistogram(histogram);
    }

    template <typename CounterT>
    void Histogram(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items, SmemTypePtr<CounterT> histogram) const
    {
        InitHistogram(histogram);
        Composite(items, histogram);
    }

    template <typename CounterT>
    void Composite(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items, SmemTypePtr<CounterT> histogram) const
    {
        // InitHistogram and caller-provided initial counts are block-wide writes.
        compute::sync_block();
        if constexpr(ALGORITHM == BLOCK_HISTO_ATOMIC)
        {
            composite_atomic(items, histogram);
        }
        else
        {
            composite_sorted(items, histogram);
        }
        // Make the completed histogram safe to consume immediately in the block.
        compute::sync_block();
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    template <typename CounterT>
    void composite_atomic(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items, SmemTypePtr<CounterT> histogram) const
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt bin = compute::cast<uint>(items[item]);
            histogram->atomic(bin).fetch_add(compute::cast<CounterT>(1u));
        }
    }

    template <typename CounterT>
    void composite_sorted(const compute::ArrayVar<T, ITEMS_PER_THREAD>& items, SmemTypePtr<CounterT> histogram) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            m_shared_items->write(tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item), items[item]);
        }
        compute::sync_block();

        $for(bin, tid, compute::UInt(BINS), compute::UInt(BLOCK_THREADS))
        {
            compute::Var<CounterT> count = histogram->read(bin);
            $for(item, compute::UInt(BLOCK_ITEMS))
            {
                count += compute::cast<CounterT>(m_shared_items->read(item) == compute::cast<T>(bin));
            };
            histogram->write(bin, count);
        };
    }

  private:
    SmemTypePtr<T> m_shared_items;
};

}  // namespace luisa::parallel_primitive
