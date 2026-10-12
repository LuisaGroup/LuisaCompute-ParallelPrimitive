/*
 * @Author: Ligo 
 * @Date: 2025-10-17 16:22:56 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:57:01
 */


#pragma once
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;
    template <typename Type4Byte, size_t LOGIC_WARP_SIZE = details::WARP_SIZE>
    struct WarpReduceShfl
    {
        constexpr static bool IS_ARCH_WARP = (LOGIC_WARP_SIZE == details::WARP_SIZE);
        compute::UInt         physical_lane;
        compute::UInt         warp_base;
        compute::UInt         lane_id;

        WarpReduceShfl()
            : physical_lane(warp_lane_id())
            , warp_base(physical_lane / compute::UInt(LOGIC_WARP_SIZE) * compute::UInt(LOGIC_WARP_SIZE))
            , lane_id(physical_lane - warp_base)
        {
        }

        template <typename ReduceOp>
        Var<Type4Byte> Reduce(const Var<Type4Byte>& input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE)
        {
            Var<Type4Byte> result = input;
            $for(source_lane, compute::UInt(LOGIC_WARP_SIZE))
            {
                Var<Type4Byte> candidate = WarpReadLane(input, warp_base + source_lane);
                Var<Type4Byte> combined  = op(result, candidate);
                Bool           include   = source_lane > lane_id & source_lane < valid_item;
                result                   = Select(result, combined, include);
            };
            return result;
        }

        template <bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
        Var<Type4Byte> SegmentReduce(const Var<Type4Byte>& input,
                                     const Var<FlagT>&     flag,
                                     ReduceOp              reduce_op,
                                     UInt                  valid_item = LOGIC_WARP_SIZE)
        {
            UInt segment_end    = compute::select(UInt(0u), valid_item - 1u, valid_item > 0u);
            Bool found_boundary = false;
            $for(source_lane, compute::UInt(LOGIC_WARP_SIZE))
            {
                UInt       candidate_lane = source_lane;
                Var<FlagT> candidate_flag = WarpReadLane(flag, warp_base + candidate_lane);
                Bool       is_boundary    = candidate_flag != Var<FlagT>(0);
                if constexpr(HEAD_SEGMENT)
                {
                    is_boundary = is_boundary & (candidate_lane > lane_id);
                }
                else
                {
                    is_boundary = is_boundary & (candidate_lane >= lane_id);
                }
                is_boundary        = is_boundary & (candidate_lane < valid_item) & !found_boundary;
                UInt candidate_end = HEAD_SEGMENT ? candidate_lane - 1u : candidate_lane;
                segment_end        = compute::select(segment_end, candidate_end, is_boundary);
                found_boundary     = found_boundary | is_boundary;
            };

            Var<Type4Byte> result = input;
            $for(source_lane, compute::UInt(LOGIC_WARP_SIZE))
            {
                UInt           candidate_lane = source_lane;
                Var<Type4Byte> candidate      = WarpReadLane(input, warp_base + candidate_lane);
                Var<Type4Byte> combined       = reduce_op(result, candidate);
                Bool           include = candidate_lane > lane_id & candidate_lane <= segment_end
                               & candidate_lane < valid_item;
                result = Select(result, combined, include);
            };
            return result;
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive
