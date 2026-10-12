/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 16:49:59
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{
namespace details
{
    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    using namespace luisa::compute;
    template <typename Type4Byte, size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_SIZE = details::WARP_SIZE>

    struct BlockReduceMem
    {
        template <typename ReduceOp>
        Var<Type4Byte> Reduce(SmemTypePtr<Type4Byte>& m_shared_mem,
                              const Var<Type4Byte>&   thread_data,
                              ReduceOp                reduce_op,
                              UInt                    valid_item = BLOCK_SIZE)
        {
            UInt           tid    = thread_id().x;
            Var<Type4Byte> result = thread_data;
            (*m_shared_mem)[tid]  = thread_data;
            sync_block();

            // Preserve input order and support partial/non-power-of-two blocks.
            // As with CUB, only thread 0's result is specified.
            $if(tid == 0u)
            {
                result = (*m_shared_mem)[0u];
                $for(item, compute::UInt(1u), compute::UInt(BLOCK_SIZE))
                {
                    $if(item < valid_item)
                    {
                        result = reduce_op(result, (*m_shared_mem)[item]);
                    };
                };
            };
            return result;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive
