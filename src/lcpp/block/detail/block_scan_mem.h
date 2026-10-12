#pragma once

#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive::details
{

template <typename T>
using BlockScanSmemPtr = luisa::compute::Shared<T>*;

template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE>
struct BlockScanMem
{
    template <typename ScanOp>
    void InclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       compute::Var<Type4Byte>&       block_aggregate,
                       ScanOp                         scan_op) const
    {
        inclusive_scan(shared_mem, input, output, block_aggregate, scan_op);
    }

    template <typename ScanOp>
    void InclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       compute::Var<Type4Byte>&       block_aggregate,
                       ScanOp                         scan_op,
                       const compute::Var<Type4Byte>& initial_value) const
    {
        compute::Var<Type4Byte> raw_inclusive;
        inclusive_scan(shared_mem, input, raw_inclusive, block_aggregate, scan_op);
        output = scan_op(initial_value, raw_inclusive);
    }

    template <typename ScanOp>
    void ExclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       compute::Var<Type4Byte>&       block_aggregate,
                       ScanOp                         scan_op) const
    {
        compute::Var<Type4Byte> raw_inclusive;
        inclusive_scan(shared_mem, input, raw_inclusive, block_aggregate, scan_op);
        compute::UInt tid = compute::thread_id().x;
        output            = input;
        $if(tid > 0u)
        {
            output = (*shared_mem)[tid - 1u];
        };
    }

    template <typename ScanOp>
    void ExclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       compute::Var<Type4Byte>&       block_aggregate,
                       ScanOp                         scan_op,
                       const compute::Var<Type4Byte>& initial_value) const
    {
        compute::Var<Type4Byte> raw_inclusive;
        inclusive_scan(shared_mem, input, raw_inclusive, block_aggregate, scan_op);
        compute::UInt tid = compute::thread_id().x;
        output            = initial_value;
        $if(tid > 0u)
        {
            output = scan_op(initial_value, (*shared_mem)[tid - 1u]);
        };
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void ExclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       ScanOp                         scan_op,
                       BlockPrefixCallbackOp          prefix_op) const
    {
        compute::Var<Type4Byte> block_aggregate;
        ExclusiveScan(shared_mem, input, output, block_aggregate, scan_op);
        apply_prefix(shared_mem, output, block_aggregate, scan_op, prefix_op, true);
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    void InclusiveScan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                       const compute::Var<Type4Byte>& input,
                       compute::Var<Type4Byte>&       output,
                       ScanOp                         scan_op,
                       BlockPrefixCallbackOp          prefix_op) const
    {
        compute::Var<Type4Byte> block_aggregate;
        InclusiveScan(shared_mem, input, output, block_aggregate, scan_op);
        apply_prefix(shared_mem, output, block_aggregate, scan_op, prefix_op, false);
    }

  private:
    template <typename ScanOp>
    static void inclusive_scan(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                               const compute::Var<Type4Byte>& input,
                               compute::Var<Type4Byte>&       output,
                               compute::Var<Type4Byte>&       block_aggregate,
                               ScanOp                         scan_op)
    {
        compute::UInt tid  = compute::thread_id().x;
        (*shared_mem)[tid] = input;
        compute::sync_block();

        for(size_t offset = 1u; offset < BLOCK_SIZE; offset <<= 1u)
        {
            compute::Var<Type4Byte> value = (*shared_mem)[tid];
            $if(tid >= compute::UInt(offset))
            {
                value = scan_op((*shared_mem)[tid - compute::UInt(offset)], value);
            };
            compute::sync_block();
            (*shared_mem)[tid] = value;
            compute::sync_block();
        }

        output          = (*shared_mem)[tid];
        block_aggregate = (*shared_mem)[compute::UInt(BLOCK_SIZE - 1u)];
    }

    template <typename ScanOp, typename BlockPrefixCallbackOp>
    static void apply_prefix(BlockScanSmemPtr<Type4Byte>&   shared_mem,
                             compute::Var<Type4Byte>&       output,
                             const compute::Var<Type4Byte>& block_aggregate,
                             ScanOp                         scan_op,
                             BlockPrefixCallbackOp          prefix_op,
                             bool                           exclusive)
    {
        compute::UInt tid = compute::thread_id().x;
        $if(tid == 0u)
        {
            (*shared_mem)[0u] = prefix_op(block_aggregate);
        };
        compute::sync_block();
        compute::Var<Type4Byte> prefix = (*shared_mem)[0u];
        if(exclusive)
        {
            output = Select(prefix, scan_op(prefix, output), tid > 0u);
        }
        else
        {
            output = scan_op(prefix, output);
        }
        compute::sync_block();
    }
};

}  // namespace luisa::parallel_primitive::details
