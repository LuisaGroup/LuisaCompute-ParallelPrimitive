/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:33:13 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 17:04:22
 */

#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/common/thread_operators.h>
#include <type_traits>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename T>
    using SmemTypePtr = luisa::compute::Shared<T>*;

    template <typename Type4Byte,
              size_t BLOCK_SIZE = details::BLOCK_SIZE,
              size_t WARP_SIZE = details::WARP_SIZE,
              bool   CUDA_WARP_REDUX = false>
    struct BlockReduceShfl
    {
        template <bool IS_FULL_TILE, typename ReduceOp>
        Var<Type4Byte> Reduce(SmemTypePtr<Type4Byte>& m_shared_mem,
                              const Var<Type4Byte>&   thread_data,
                              ReduceOp                reduce_op,
                              UInt                    valid_item)
        {
            UInt           warp_id     = thread_id().x / compute::UInt(WARP_SIZE);
            UInt           warp_offset = warp_id * compute::UInt(WARP_SIZE);
            constexpr uint warp_nums   = (BLOCK_SIZE + WARP_SIZE - 1u) / WARP_SIZE;
            UInt           lane_id     = warp_lane_id() % compute::UInt(WARP_SIZE);
            UInt           thid        = thread_id().x;

            UInt tile_valid = IS_FULL_TILE ? compute::UInt(BLOCK_SIZE) :
                                             compute::min(valid_item, compute::UInt(BLOCK_SIZE));
            UInt remaining = compute::select(0u, tile_valid - warp_offset, tile_valid > warp_offset);
            UInt warp_valid_num = compute::min(compute::UInt(WARP_SIZE), remaining);

            Var<Type4Byte> warp_aggregate;
            if constexpr(CUDA_WARP_REDUX && IS_FULL_TILE &&
                         std::is_same_v<std::remove_cvref_t<ReduceOp>, SumOp> &&
                         (std::is_same_v<Type4Byte, int> || std::is_same_v<Type4Byte, uint>))
            {
                warp_aggregate = details::WarpReduceShfl<Type4Byte, WARP_SIZE>{}.ReduceAddSync(thread_data);
            }
            else
            {
                warp_aggregate =
                    WarpReduce<Type4Byte, WARP_SIZE>().Reduce(thread_data, reduce_op, warp_valid_num);
            }

            // write warp result to shared memory
            $if(lane_id == 0u & warp_valid_num > 0u)
            {
                (*m_shared_mem)[warp_id] = warp_aggregate;
            };

            sync_block();

            $if(thid == 0)
            {
                $for(warp_idx, compute::UInt(1u), compute::UInt(warp_nums))
                {
                    $if(IS_FULL_TILE | warp_idx * compute::UInt(WARP_SIZE) < tile_valid)
                    {
                        Var<Type4Byte> val = (*m_shared_mem)[warp_idx];
                        warp_aggregate     = reduce_op(warp_aggregate, val);
                    };
                };
            };

            return warp_aggregate;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive
