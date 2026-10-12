#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>
#include <cstdint>

namespace luisa::parallel_primitive
{

template <typename ItemT, size_t BLOCK_DIM_X, size_t RUNS_PER_THREAD, size_t DECODED_ITEMS_PER_THREAD, typename DecodedOffsetT = uint32_t, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockRunLengthDecode : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_RUNS    = BLOCK_THREADS * RUNS_PER_THREAD;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(RUNS_PER_THREAD > 0u);
    static_assert(DECODED_ITEMS_PER_THREAD > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<ItemT>          run_values;
        SmemTypePtr<DecodedOffsetT> run_offsets;
        SmemTypePtr<DecodedOffsetT> total_decoded_size;
    };
    using TempStorage = tempStorage;

    template <typename RunLengthArray, typename TotalDecodedSizeVar>
    BlockRunLengthDecode(const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values,
                         const RunLengthArray&                            run_lengths,
                         TotalDecodedSizeVar&                             total_decoded_size)
        : BlockRunLengthDecode()
    {
        InitWithRunLengths(run_values, run_lengths, total_decoded_size);
    }

    template <typename RunOffsetArray>
    BlockRunLengthDecode(const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values, const RunOffsetArray& run_offsets)
        : BlockRunLengthDecode()
    {
        InitWithRunOffsets(run_values, run_offsets);
    }

    template <typename RunLengthArray, typename TotalDecodedSizeVar>
    BlockRunLengthDecode(tempStorage                                      storage,
                         const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values,
                         const RunLengthArray&                            run_lengths,
                         TotalDecodedSizeVar&                             total_decoded_size)
        : BlockRunLengthDecode(storage)
    {
        InitWithRunLengths(run_values, run_lengths, total_decoded_size);
    }

    template <typename RunOffsetArray>
    BlockRunLengthDecode(tempStorage                                      storage,
                         const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values,
                         const RunOffsetArray&                            run_offsets)
        : BlockRunLengthDecode(storage)
    {
        InitWithRunOffsets(run_values, run_offsets);
    }

    template <typename RunLengthArray, typename TotalDecodedSizeVar>
    void InitWithRunLengths(const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values,
                            const RunLengthArray&                            run_lengths,
                            TotalDecodedSizeVar& total_decoded_size) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < RUNS_PER_THREAD; ++item)
        {
            compute::UInt run = tid * compute::UInt(RUNS_PER_THREAD) + compute::UInt(item);
            m_run_values->write(run, run_values[item]);
            m_run_offsets->write(run, compute::cast<DecodedOffsetT>(run_lengths[item]));
        }
        compute::sync_block();

        $if(tid == 0u)
        {
            compute::Var<DecodedOffsetT> running = DecodedOffsetT{};
            $for(run, compute::UInt(BLOCK_RUNS))
            {
                compute::Var<DecodedOffsetT> length = m_run_offsets->read(run);
                m_run_offsets->write(run, running);
                running += length;
            };
            m_total_decoded_size->write(0u, running);
        };
        compute::sync_block();
        total_decoded_size = m_total_decoded_size->read(0u);
    }

    template <typename RunOffsetArray>
    void InitWithRunOffsets(const compute::ArrayVar<ItemT, RUNS_PER_THREAD>& run_values,
                            const RunOffsetArray&                            run_offsets) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < RUNS_PER_THREAD; ++item)
        {
            compute::UInt run = tid * compute::UInt(RUNS_PER_THREAD) + compute::UInt(item);
            m_run_values->write(run, run_values[item]);
            m_run_offsets->write(run, compute::cast<DecodedOffsetT>(run_offsets[item]));
        }
        compute::sync_block();
    }

    template <typename RelativeOffsetArray>
        requires requires(RelativeOffsetArray& offsets) { offsets[0u]; }
    void RunLengthDecode(compute::ArrayVar<ItemT, DECODED_ITEMS_PER_THREAD>& decoded_items,
                         RelativeOffsetArray&                                item_offsets,
                         compute::Var<DecodedOffsetT> from_decoded_offset) const
    {
        compute::Var<DecodedOffsetT> thread_decoded_offset =
            from_decoded_offset
            + compute::cast<DecodedOffsetT>(linear_thread_id())
                  * compute::cast<DecodedOffsetT>(DECODED_ITEMS_PER_THREAD);

        for(size_t item = 0u; item < DECODED_ITEMS_PER_THREAD; ++item)
        {
            compute::Var<DecodedOffsetT> decoded_offset =
                thread_decoded_offset + compute::cast<DecodedOffsetT>(item);
            compute::UInt assigned_run = 0u;
            $for(run, compute::UInt(1u), compute::UInt(BLOCK_RUNS))
            {
                compute::Bool begins_before_or_at = m_run_offsets->read(run) <= decoded_offset;
                assigned_run = compute::select(assigned_run, run, begins_before_or_at);
            };

            compute::Var<DecodedOffsetT> run_begin = m_run_offsets->read(assigned_run);
            decoded_items[item]                    = m_run_values->read(assigned_run);
            item_offsets[item]                     = decoded_offset - run_begin;
        }
    }

    template <typename RelativeOffsetArray>
        requires requires(RelativeOffsetArray& offsets) { offsets[0u]; }
    void RunLengthDecode(compute::ArrayVar<ItemT, DECODED_ITEMS_PER_THREAD>& decoded_items,
                         RelativeOffsetArray&                                item_offsets) const
    {
        RunLengthDecode(decoded_items, item_offsets, compute::Var<DecodedOffsetT>{DecodedOffsetT{}});
    }

    void RunLengthDecode(compute::ArrayVar<ItemT, DECODED_ITEMS_PER_THREAD>& decoded_items,
                         compute::Var<DecodedOffsetT> from_decoded_offset) const
    {
        compute::ArrayVar<DecodedOffsetT, DECODED_ITEMS_PER_THREAD> item_offsets;
        RunLengthDecode(decoded_items, item_offsets, from_decoded_offset);
    }

    void RunLengthDecode(compute::ArrayVar<ItemT, DECODED_ITEMS_PER_THREAD>& decoded_items) const
    {
        RunLengthDecode(decoded_items, compute::Var<DecodedOffsetT>{DecodedOffsetT{}});
    }

  private:
    BlockRunLengthDecode()
        : m_run_values(new SmemType<ItemT>{BLOCK_RUNS})
        , m_run_offsets(new SmemType<DecodedOffsetT>{BLOCK_RUNS})
        , m_total_decoded_size(new SmemType<DecodedOffsetT>{1u})
    {
    }

    explicit BlockRunLengthDecode(tempStorage storage)
        : m_run_values(storage.run_values)
        , m_run_offsets(storage.run_offsets)
        , m_total_decoded_size(storage.total_decoded_size)
    {
    }

    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

  private:
    SmemTypePtr<ItemT>          m_run_values;
    SmemTypePtr<DecodedOffsetT> m_run_offsets;
    SmemTypePtr<DecodedOffsetT> m_total_decoded_size;
};

}  // namespace luisa::parallel_primitive
