#pragma once

#include <lcpp/block/detail/block_scan_mem.h>
#include <lcpp/block/detail/block_scan_warp.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/thread/thread_scan.h>

#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive
{

enum class BlockScanAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

template <typename Type4Byte,
          size_t             BLOCK_SIZE          = details::BLOCK_SIZE,
          size_t             ITEMS_PER_THREAD    = details::ITEMS_PER_THREAD,
          size_t             WARP_SIZE           = details::WARP_SIZE,
          BlockScanAlgorithm DEFALUTE_ALGORITHNM = BlockScanAlgorithm::WARP_SHUFFLE>
class BlockScan : public LuisaModule
{
    static_assert(BLOCK_SIZE > 0u);
    static_assert(BLOCK_SIZE % details::WARP_SIZE == 0u,
                  "LuisaCompute block sizes must be a multiple of the physical warp size.");
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(WARP_SIZE > 0u);
    static_assert((WARP_SIZE & (WARP_SIZE - 1u)) == 0u);
    static_assert(WARP_SIZE <= details::WARP_SIZE);
    static constexpr bool USE_WARP_SHUFFLE = DEFALUTE_ALGORITHNM == BlockScanAlgorithm::WARP_SHUFFLE;

  public:
    BlockScan()
    {
        if constexpr(!USE_WARP_SHUFFLE)
        {
            m_shared_mem = new SmemType<Type4Byte>{BLOCK_SIZE};
        }
        else
        {
            m_shared_mem = new SmemType<Type4Byte>{(BLOCK_SIZE + WARP_SIZE - 1u) / WARP_SIZE};
        }
    }

    explicit BlockScan(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    ~BlockScan() = default;

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, ScanOp scan_op)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(input, output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, Var<Type4Byte>& block_aggregate, ScanOp scan_op)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, ScanOp scan_op, const Var<Type4Byte>& initial_value)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(input, output, block_aggregate, scan_op, initial_value);
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& input,
                       Var<Type4Byte>&       output,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op, initial_value);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op, initial_value);
        }
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void ExclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, ScanOp scan_op, BlockPrefixCallbackOp prefix_op)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, scan_op, prefix_op);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().ExclusiveScan(
                m_shared_mem, input, output, scan_op, prefix_op);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       ScanOp                                                scan_op)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(input, output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        Var<Type4Byte> block_aggregate;
        ExclusiveScan(input, output, block_aggregate, scan_op, initial_value);
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            ExclusiveScan(input[0], output[0], block_aggregate, scan_op);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, block_aggregate, scan_op);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(
                input, output, scan_op, thread_prefix, compute::thread_x() != 0u);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            ExclusiveScan(input[0], output[0], block_aggregate, scan_op, initial_value);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, block_aggregate, scan_op, initial_value);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(input, output, scan_op, thread_prefix);
        }
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void ExclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       ScanOp                                                scan_op,
                       BlockPrefixCallbackOp                                 prefix_op)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            ExclusiveScan(input[0], output[0], scan_op, prefix_op);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, scan_op, prefix_op);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanExclusive(input, output, scan_op, thread_prefix);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, ScanOp scan_op)
    {
        Var<Type4Byte> block_aggregate;
        InclusiveScan(input, output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, Var<Type4Byte>& block_aggregate, ScanOp scan_op)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& input,
                       Var<Type4Byte>&       output,
                       Var<Type4Byte>&       block_aggregate,
                       ScanOp                scan_op,
                       Var<Type4Byte>        initial_value)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op, initial_value);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, block_aggregate, scan_op, initial_value);
        }
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void InclusiveScan(const Var<Type4Byte>& input, Var<Type4Byte>& output, ScanOp scan_op, BlockPrefixCallbackOp prefix_op)
    {
        if constexpr(USE_WARP_SHUFFLE)
        {
            details::BlockScanShfl<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, scan_op, prefix_op);
        }
        else
        {
            details::BlockScanMem<Type4Byte, BLOCK_SIZE, WARP_SIZE>().InclusiveScan(
                m_shared_mem, input, output, scan_op, prefix_op);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       ScanOp                                                scan_op)
    {
        Var<Type4Byte> block_aggregate;
        InclusiveScan(input, output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            InclusiveScan(input[0], output[0], block_aggregate, scan_op);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, block_aggregate, scan_op);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(
                input, output, scan_op, thread_prefix, compute::thread_x() != 0u);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       Var<Type4Byte>&                                       block_aggregate,
                       ScanOp                                                scan_op,
                       Var<Type4Byte>                                        initial_value)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            InclusiveScan(input[0], output[0], block_aggregate, scan_op, initial_value);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, block_aggregate, scan_op, initial_value);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(input, output, scan_op, thread_prefix);
        }
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void InclusiveScan(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                       compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                       ScanOp                                                scan_op,
                       BlockPrefixCallbackOp                                 prefix_op)
    {
        if constexpr(ITEMS_PER_THREAD == 1u)
        {
            InclusiveScan(input[0], output[0], scan_op, prefix_op);
        }
        else
        {
            Var<Type4Byte> thread_aggregate =
                ThreadReduce<Type4Byte, ITEMS_PER_THREAD>().Reduce(input, scan_op);
            Var<Type4Byte> thread_prefix;
            ExclusiveScan(thread_aggregate, thread_prefix, scan_op, prefix_op);
            ThreadScan<Type4Byte, ITEMS_PER_THREAD>().ThreadScanInclusive(input, output, scan_op, thread_prefix);
        }
    }

    void ExclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output)
    {
        ExclusiveScan(input, output, SumOp(), Var<Type4Byte>{Type4Byte{}});
    }

    void ExclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output, Var<Type4Byte>& block_aggregate)
    {
        ExclusiveScan(input, output, block_aggregate, SumOp(), Var<Type4Byte>{Type4Byte{}});
    }

    template <typename BlockPrefixCallbackOp>
    void ExclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output, BlockPrefixCallbackOp prefix_op)
    {
        ExclusiveScan(input, output, SumOp(), prefix_op);
    }

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output)
    {
        ExclusiveScan(input, output, SumOp(), Var<Type4Byte>{Type4Byte{}});
    }

    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                      Var<Type4Byte>&                                       block_aggregate)
    {
        ExclusiveScan(input, output, block_aggregate, SumOp(), Var<Type4Byte>{Type4Byte{}});
    }

    template <typename BlockPrefixCallbackOp>
    void ExclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                      BlockPrefixCallbackOp                                 prefix_op)
    {
        ExclusiveScan(input, output, SumOp(), prefix_op);
    }

    void InclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output)
    {
        InclusiveScan(input, output, SumOp());
    }

    void InclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output, Var<Type4Byte>& block_aggregate)
    {
        InclusiveScan(input, output, block_aggregate, SumOp());
    }

    template <typename BlockPrefixCallbackOp>
    void InclusiveSum(const Var<Type4Byte>& input, Var<Type4Byte>& output, BlockPrefixCallbackOp prefix_op)
    {
        InclusiveScan(input, output, SumOp(), prefix_op);
    }

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output)
    {
        InclusiveScan(input, output, SumOp());
    }

    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                      Var<Type4Byte>&                                       block_aggregate)
    {
        InclusiveScan(input, output, block_aggregate, SumOp());
    }

    template <typename BlockPrefixCallbackOp>
    void InclusiveSum(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                      compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>&       output,
                      BlockPrefixCallbackOp                                 prefix_op)
    {
        InclusiveScan(input, output, SumOp(), prefix_op);
    }

  private:
    SmemTypePtr<Type4Byte> m_shared_mem;
};

}  // namespace luisa::parallel_primitive
