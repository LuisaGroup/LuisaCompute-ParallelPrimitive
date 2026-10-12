/*
 * @Author: Ligo 
 * @Date: 2025-10-17 15:00:46 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-10-22 23:57:08
 */


#pragma once

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <typename Type4Byte, size_t LOGIC_WARP_SIZE = details::WARP_SIZE>
    struct WarpScanShfl
    {
        template <typename ScanOp>
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> output = inclusive_result(thread_input, scan_op);
            inclusive_output      = scan_op(initial_value, output);
        }

        template <typename ScanOp>
        void InclusiveScan(const Var<Type4Byte>& thread_input, Var<Type4Byte>& inclusive_output, ScanOp scan_op)
        {
            inclusive_output = inclusive_result(thread_input, scan_op);
        }

        template <typename ScanOp>
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op)
        {
            Var<Type4Byte> output = inclusive_result(thread_input, scan_op);
            inclusive_output      = output;
            warp_aggregate        = aggregate(output);
        }

        template <typename ScanOp>
        void InclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       inclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> output = inclusive_result(thread_input, scan_op);
            inclusive_output      = scan_op(initial_value, output);
            warp_aggregate        = aggregate(output);
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input, Var<Type4Byte>& exclusive_output, ScanOp scan_op)
        {
            Var<Type4Byte> inclusive_output = inclusive_result(thread_input, scan_op);
            exclusive_output = exclusive_from_inclusive(inclusive_output, thread_input);
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> inclusive_output = inclusive_result(thread_input, scan_op);
            Var<Type4Byte> previous = exclusive_from_inclusive(inclusive_output, thread_input);
            UInt           lane     = logical_lane();
            Var<Type4Byte> prefixed = scan_op(initial_value, previous);
            exclusive_output        = Select(initial_value, prefixed, lane > 0u);
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op)
        {
            Var<Type4Byte> inclusive_output = inclusive_result(thread_input, scan_op);
            exclusive_output = exclusive_from_inclusive(inclusive_output, thread_input);
            warp_aggregate   = aggregate(inclusive_output);
        }

        template <typename ScanOp>
        void ExclusiveScan(const Var<Type4Byte>& thread_input,
                           Var<Type4Byte>&       exclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op,
                           const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> inclusive_output = inclusive_result(thread_input, scan_op);
            Var<Type4Byte> previous = exclusive_from_inclusive(inclusive_output, thread_input);
            UInt           lane     = logical_lane();
            Var<Type4Byte> prefixed = scan_op(initial_value, previous);
            exclusive_output        = Select(initial_value, prefixed, lane > 0u);
            warp_aggregate          = aggregate(inclusive_output);
        }

        template <typename ScanOp>
        void Scan(const Var<Type4Byte>& thread_input,
                  Var<Type4Byte>&       inclusive_output,
                  Var<Type4Byte>&       exclusive_output,
                  ScanOp                scan_op)
        {
            Var<Type4Byte> output = inclusive_result(thread_input, scan_op);
            inclusive_output      = output;
            exclusive_output      = exclusive_from_inclusive(output, thread_input);
        }

        template <typename ScanOp>
        void Scan(const Var<Type4Byte>& thread_input,
                  Var<Type4Byte>&       inclusive_output,
                  Var<Type4Byte>&       exclusive_output,
                  ScanOp                scan_op,
                  const Var<Type4Byte>& initial_value)
        {
            Var<Type4Byte> output   = inclusive_result(thread_input, scan_op);
            inclusive_output        = scan_op(initial_value, output);
            Var<Type4Byte> previous = exclusive_from_inclusive(output, thread_input);
            UInt           lane     = logical_lane();
            Var<Type4Byte> prefixed = scan_op(initial_value, previous);
            exclusive_output        = Select(initial_value, prefixed, lane > 0u);
        }

        template <typename ScanOp>
        void InclusiveScanPartial(const Var<Type4Byte>& thread_input,
                                  Var<Type4Byte>&       inclusive_output,
                                  Var<Type4Byte>&       warp_aggregate,
                                  ScanOp                scan_op,
                                  UInt                  valid_items)
        {
            inclusive_output = inclusive_partial_result(thread_input, scan_op, valid_items);
            UInt last_lane = compute::min(compute::max(valid_items, 1u), UInt(LOGIC_WARP_SIZE)) - 1u;
            warp_aggregate = WarpReadLane(inclusive_output, logical_warp_base() + last_lane);
        }

      private:
        static UInt logical_warp_base()
        {
            UInt physical_lane = compute::warp_lane_id();
            return physical_lane / UInt(LOGIC_WARP_SIZE) * UInt(LOGIC_WARP_SIZE);
        }

        static UInt logical_lane() { return compute::warp_lane_id() - logical_warp_base(); }

        template <typename ScanOp>
        static Var<Type4Byte> inclusive_result(const Var<Type4Byte>& thread_input, ScanOp scan_op)
        {
            UInt           physical_lane = compute::warp_lane_id();
            UInt           warp_base     = logical_warp_base();
            UInt           lane          = physical_lane - warp_base;
            Var<Type4Byte> result        = thread_input;
            for(size_t offset = 1u; offset < LOGIC_WARP_SIZE; offset <<= 1u)
            {
                UInt source_lane =
                    compute::select(physical_lane, physical_lane - UInt(offset), lane >= UInt(offset));
                Var<Type4Byte> candidate = WarpReadLane(result, source_lane);
                Var<Type4Byte> combined  = scan_op(candidate, result);
                result                   = Select(result, combined, lane >= UInt(offset));
            }
            return result;
        }

        template <typename ScanOp>
        static Var<Type4Byte> inclusive_partial_result(const Var<Type4Byte>& thread_input, ScanOp scan_op, UInt valid_items)
        {
            UInt           warp_base = logical_warp_base();
            UInt           lane      = logical_lane();
            Var<Type4Byte> result    = WarpReadLane(thread_input, warp_base);
            for(size_t source = 1u; source < LOGIC_WARP_SIZE; ++source)
            {
                Var<Type4Byte> candidate = WarpReadLane(thread_input, warp_base + UInt(source));
                $if(lane < valid_items & UInt(source) <= lane)
                {
                    result = scan_op(result, candidate);
                };
            }
            return result;
        }

        static Var<Type4Byte> exclusive_from_inclusive(const Var<Type4Byte>& inclusive_output,
                                                       const Var<Type4Byte>& lane_zero_fallback)
        {
            UInt           physical_lane = compute::warp_lane_id();
            UInt           warp_base     = logical_warp_base();
            UInt           lane          = physical_lane - warp_base;
            UInt           source_lane = compute::select(warp_base, physical_lane - 1u, lane > 0u);
            Var<Type4Byte> previous    = WarpReadLane(inclusive_output, source_lane);
            return Select(lane_zero_fallback, previous, lane > 0u);
        }

        static Var<Type4Byte> aggregate(const Var<Type4Byte>& inclusive_output)
        {
            return WarpReadLane(inclusive_output, logical_warp_base() + UInt(LOGIC_WARP_SIZE - 1u));
        }
    };
}  // namespace details
}  // namespace luisa::parallel_primitive
