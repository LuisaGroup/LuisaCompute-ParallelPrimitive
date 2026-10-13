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
#include <type_traits>
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
        Var<Type4Byte> Reduce(const Var<Type4Byte>& input, ReduceOp op)
        {
            Var<Type4Byte> result = input;
            for(size_t offset = 1u; offset < LOGIC_WARP_SIZE; offset <<= 1u)
            {
                UInt source_lane = lane_id + UInt(offset);
                source_lane = compute::select(lane_id, source_lane, source_lane < UInt(LOGIC_WARP_SIZE));
                Var<Type4Byte> candidate = WarpReadLane(result, warp_base + source_lane);
                result = Select(result, op(result, candidate), lane_id + UInt(offset) < UInt(LOGIC_WARP_SIZE));
            }
            return result;
        }

        template <typename ReduceOp>
        Var<Type4Byte> Reduce(const Var<Type4Byte>& input, ReduceOp op, UInt valid_item)
        {
            UInt clamped_valid = compute::min(valid_item, UInt(LOGIC_WARP_SIZE));
            Var<Type4Byte> result = input;
            for(size_t offset = 1u; offset < LOGIC_WARP_SIZE; offset <<= 1u)
            {
                UInt source_lane = lane_id + UInt(offset);
                source_lane = compute::select(lane_id, source_lane, source_lane < UInt(LOGIC_WARP_SIZE));
                Var<Type4Byte> candidate = WarpReadLane(result, warp_base + source_lane);
                Var<Type4Byte> combined  = op(result, candidate);
                Bool           include   = lane_id + UInt(offset) < clamped_valid;
                result                   = Select(result, combined, include);
            };
            return result;
        }

        // CUDA exposes a native warp reduction for 32-bit integer addition.
        // The device-level reducer selects this variant only for CUDA.
        Var<Type4Byte> ReduceAddSync(const Var<Type4Byte>& input)
        {
            static_assert(LOGIC_WARP_SIZE == details::WARP_SIZE);
            static_assert(std::is_same_v<Type4Byte, int> || std::is_same_v<Type4Byte, uint>);
            static compute::ExternalCallable<Type4Byte(uint, Type4Byte)> reduce_add_sync{"__reduce_add_sync"};
            return reduce_add_sync(UInt(0xffffffffu), input);
        }

        template <typename FlagT, typename ReduceOp>
        Var<Type4Byte> SegmentReduceFull(const Var<Type4Byte>& input,
                                         const Var<FlagT>&     flag,
                                         ReduceOp              reduce_op)
        {
            constexpr unsigned logical_mask = LOGIC_WARP_SIZE == details::WARP_SIZE
                                                  ? 0xffffffffu
                                                  : ((1u << LOGIC_WARP_SIZE) - 1u);
            UInt local_mask = compute::warp_active_bit_mask(flag != Var<FlagT>(0)).x >> warp_base;
            local_mask      = (local_mask | UInt(1u << (LOGIC_WARP_SIZE - 1u))) & UInt(logical_mask);
            UInt segment_end = compute::ctz(local_mask & (UInt(0xffffffffu) << lane_id));

            Var<Type4Byte> result = input;
            for(size_t offset = 1u; offset < LOGIC_WARP_SIZE; offset <<= 1u)
            {
                UInt source_lane = lane_id + UInt(offset);
                source_lane = compute::select(lane_id, source_lane, source_lane < UInt(LOGIC_WARP_SIZE));
                Var<Type4Byte> candidate = WarpReadLane(result, warp_base + source_lane);
                result = Select(result, reduce_op(result, candidate), lane_id + UInt(offset) <= segment_end);
            }
            return result;
        }

        template <bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
        Var<Type4Byte> SegmentReduce(const Var<Type4Byte>& input,
                                     const Var<FlagT>&     flag,
                                     ReduceOp              reduce_op,
                                     UInt                  valid_item = LOGIC_WARP_SIZE)
        {
            if constexpr(!HEAD_SEGMENT)
            {
                // Tail flags identify the last item in a segment.  CUB finds
                // the next tail with one ballot and then performs the segment
                // reduction with the five shuffle-down stages of a warp
                // reduction.  The previous implementation scanned all 32
                // lanes twice for every look-back window.
                UInt local_mask    = compute::warp_active_bit_mask(flag != Var<FlagT>(0)).x;
                local_mask         = local_mask >> warp_base;
                UInt clamped_valid = compute::min(valid_item, UInt(LOGIC_WARP_SIZE));
                constexpr unsigned logical_mask = LOGIC_WARP_SIZE == details::WARP_SIZE
                                                      ? 0xffffffffu
                                                      : ((1u << LOGIC_WARP_SIZE) - 1u);
                UInt valid_mask = UInt(logical_mask);
                $if(clamped_valid < UInt(LOGIC_WARP_SIZE))
                {
                    valid_mask = (UInt(1u) << clamped_valid) - UInt(1u);
                };
                local_mask = local_mask & valid_mask;
                UInt valid_last = compute::select(UInt(0u), clamped_valid - UInt(1u), clamped_valid > UInt(0u));
                local_mask     = local_mask | (UInt(1u) << valid_last);
                UInt lane_mask  = UInt(0xffffffffu) << lane_id;
                UInt segment_end = compute::ctz(local_mask & lane_mask);
                segment_end = compute::select(lane_id, segment_end, lane_id < clamped_valid);

                Var<Type4Byte> result = input;
                for(size_t offset = 1u; offset < LOGIC_WARP_SIZE; offset <<= 1u)
                {
                    UInt source_lane = lane_id + UInt(offset);
                    source_lane = compute::select(lane_id, source_lane, source_lane < UInt(LOGIC_WARP_SIZE));
                    Var<Type4Byte> candidate = WarpReadLane(result, warp_base + source_lane);
                    Bool include = lane_id + UInt(offset) <= segment_end;
                    result      = Select(result, reduce_op(result, candidate), include);
                }
                return result;
            }

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
