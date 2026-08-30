#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive::details
{

template <typename T, size_t LOGICAL_WARP_SIZE = details::WARP_SIZE>
class WarpScanSmem : public LuisaModule
{
  public:
    explicit WarpScanSmem(SmemTypePtr<T> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::Var<T>& input, compute::Var<T>& output, ScanOp scan_op) const
    {
        prepare(input);
        output = inclusive(scan_op);
        finish();
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::Var<T>& input,
                       compute::Var<T>&       output,
                       ScanOp                 scan_op,
                       const compute::Var<T>& initial_value) const
    {
        prepare(input);
        output = inclusive(scan_op, initial_value);
        finish();
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::Var<T>& input, compute::Var<T>& output, compute::Var<T>& warp_aggregate, ScanOp scan_op) const
    {
        prepare(input);
        output         = inclusive(scan_op);
        warp_aggregate = aggregate(scan_op);
        finish();
    }

    template <typename ScanOp>
    void InclusiveScan(const compute::Var<T>& input,
                       compute::Var<T>&       output,
                       compute::Var<T>&       warp_aggregate,
                       ScanOp                 scan_op,
                       const compute::Var<T>& initial_value) const
    {
        prepare(input);
        output         = inclusive(scan_op, initial_value);
        warp_aggregate = aggregate(scan_op);
        finish();
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::Var<T>& input, compute::Var<T>& output, ScanOp scan_op) const
    {
        prepare(input);
        output = exclusive(scan_op, input);
        finish();
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::Var<T>& input,
                       compute::Var<T>&       output,
                       ScanOp                 scan_op,
                       const compute::Var<T>& initial_value) const
    {
        prepare(input);
        output = exclusive(scan_op, initial_value, initial_value);
        finish();
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::Var<T>& input, compute::Var<T>& output, compute::Var<T>& warp_aggregate, ScanOp scan_op) const
    {
        prepare(input);
        output         = exclusive(scan_op, input);
        warp_aggregate = aggregate(scan_op);
        finish();
    }

    template <typename ScanOp>
    void ExclusiveScan(const compute::Var<T>& input,
                       compute::Var<T>&       output,
                       compute::Var<T>&       warp_aggregate,
                       ScanOp                 scan_op,
                       const compute::Var<T>& initial_value) const
    {
        prepare(input);
        output         = exclusive(scan_op, initial_value, initial_value);
        warp_aggregate = aggregate(scan_op);
        finish();
    }

    template <typename ScanOp>
    void Scan(const compute::Var<T>& input, compute::Var<T>& inclusive_output, compute::Var<T>& exclusive_output, ScanOp scan_op) const
    {
        prepare(input);
        inclusive_output = inclusive(scan_op);
        exclusive_output = exclusive(scan_op, input);
        finish();
    }

    template <typename ScanOp>
    void Scan(const compute::Var<T>& input,
              compute::Var<T>&       inclusive_output,
              compute::Var<T>&       exclusive_output,
              ScanOp                 scan_op,
              const compute::Var<T>& initial_value) const
    {
        prepare(input);
        inclusive_output = inclusive(scan_op, initial_value);
        exclusive_output = exclusive(scan_op, initial_value, initial_value);
        finish();
    }

    template <typename ScanOp>
    void InclusiveScanPartial(const compute::Var<T>& input,
                              compute::Var<T>&       inclusive_output,
                              compute::Var<T>&       warp_aggregate,
                              ScanOp                 scan_op,
                              compute::UInt          valid_items) const
    {
        prepare(input);
        inclusive_output = inclusive_partial(scan_op, valid_items);
        warp_aggregate   = aggregate_partial(scan_op, valid_items);
        finish();
    }

    compute::Var<T> Broadcast(const compute::Var<T>& input, compute::UInt source_lane) const
    {
        prepare(input);
        compute::Var<T> result = m_shared_mem->read(logical_warp_base() + source_lane);
        finish();
        return result;
    }

  private:
    void prepare(const compute::Var<T>& input) const
    {
        m_shared_mem->write(linear_thread_id(), input);
        compute::sync_block();
    }

    static void finish() { compute::sync_block(); }

    template <typename ScanOp>
    compute::Var<T> inclusive(ScanOp scan_op) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::UInt   lane   = logical_lane();
        compute::Var<T> result = m_shared_mem->read(base);
        $for(source_lane, compute::UInt(1u), lane + 1u)
        {
            result = scan_op(result, m_shared_mem->read(base + source_lane));
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> inclusive(ScanOp scan_op, const compute::Var<T>& initial_value) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::UInt   lane   = logical_lane();
        compute::Var<T> result = initial_value;
        $for(source_lane, compute::UInt(0u), lane + 1u)
        {
            result = scan_op(result, m_shared_mem->read(base + source_lane));
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> inclusive_partial(ScanOp scan_op, compute::UInt valid_items) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::UInt   lane   = logical_lane();
        compute::Var<T> result = m_shared_mem->read(base);
        $if(lane < valid_items)
        {
            $for(source_lane, compute::UInt(1u), lane + 1u)
            {
                result = scan_op(result, m_shared_mem->read(base + source_lane));
            };
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> exclusive(ScanOp scan_op, const compute::Var<T>& lane_zero_fallback) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::UInt   lane   = logical_lane();
        compute::Var<T> result = lane_zero_fallback;
        $if(lane > 0u)
        {
            result = m_shared_mem->read(base);
            $for(source_lane, compute::UInt(1u), lane)
            {
                result = scan_op(result, m_shared_mem->read(base + source_lane));
            };
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> exclusive(ScanOp                 scan_op,
                              const compute::Var<T>& initial_value,
                              const compute::Var<T>& lane_zero_fallback) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::UInt   lane   = logical_lane();
        compute::Var<T> result = lane_zero_fallback;
        $if(lane > 0u)
        {
            result = initial_value;
            $for(source_lane, compute::UInt(0u), lane)
            {
                result = scan_op(result, m_shared_mem->read(base + source_lane));
            };
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> aggregate(ScanOp scan_op) const
    {
        compute::UInt   base   = logical_warp_base();
        compute::Var<T> result = m_shared_mem->read(base);
        $for(source_lane, compute::UInt(1u), compute::UInt(LOGICAL_WARP_SIZE))
        {
            result = scan_op(result, m_shared_mem->read(base + source_lane));
        };
        return result;
    }

    template <typename ScanOp>
    compute::Var<T> aggregate_partial(ScanOp scan_op, compute::UInt valid_items) const
    {
        compute::UInt   base          = logical_warp_base();
        compute::Var<T> result        = m_shared_mem->read(base);
        compute::UInt   clamped_items = compute::min(valid_items, compute::UInt(LOGICAL_WARP_SIZE));
        $for(source_lane, compute::UInt(1u), clamped_items)
        {
            result = scan_op(result, m_shared_mem->read(base + source_lane));
        };
        return result;
    }

    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid  = compute::thread_id();
        compute::UInt3 size = compute::block_size();
        return tid.x + size.x * (tid.y + size.y * tid.z);
    }

    static compute::UInt logical_lane()
    {
        return linear_thread_id() % compute::UInt(LOGICAL_WARP_SIZE);
    }

    static compute::UInt logical_warp_base() { return linear_thread_id() - logical_lane(); }

  private:
    SmemTypePtr<T> m_shared_mem;
};

}  // namespace luisa::parallel_primitive::details
