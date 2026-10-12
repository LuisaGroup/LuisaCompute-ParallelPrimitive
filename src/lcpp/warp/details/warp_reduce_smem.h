#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <cstddef>

namespace luisa::parallel_primitive::details
{

template <typename T, size_t LOGICAL_WARP_SIZE = details::WARP_SIZE>
class WarpReduceSmem : public LuisaModule
{
  public:
    explicit WarpReduceSmem(SmemTypePtr<T> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }

    template <typename ReduceOp>
    compute::Var<T> Reduce(const compute::Var<T>& input, ReduceOp reduce_op, compute::UInt valid_items = LOGICAL_WARP_SIZE) const
    {
        compute::UInt tid         = linear_thread_id();
        compute::UInt lane        = tid % compute::UInt(LOGICAL_WARP_SIZE);
        compute::UInt warp_base   = tid - lane;
        compute::UInt logical_end = compute::min(valid_items, compute::UInt(LOGICAL_WARP_SIZE));

        m_shared_mem->write(tid, input);
        compute::sync_block();

        compute::Var<T> result = input;
        $for(source_lane, lane + 1u, logical_end)
        {
            result = reduce_op(result, m_shared_mem->read(warp_base + source_lane));
        };
        compute::sync_block();
        return result;
    }

    template <bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
    compute::Var<T> SegmentReduce(const compute::Var<T>&     input,
                                  const compute::Var<FlagT>& flag,
                                  ReduceOp                   reduce_op,
                                  compute::UInt              valid_items = LOGICAL_WARP_SIZE) const
    {
        constexpr size_t MAX_BLOCK_THREADS = 1024u;
        auto             flags             = new SmemType<FlagT>{MAX_BLOCK_THREADS};
        compute::UInt    tid               = linear_thread_id();
        compute::UInt    lane              = tid % compute::UInt(LOGICAL_WARP_SIZE);
        compute::UInt    warp_base         = tid - lane;
        compute::UInt    logical_end = compute::min(valid_items, compute::UInt(LOGICAL_WARP_SIZE));

        m_shared_mem->write(tid, input);
        flags->write(tid, flag);
        compute::sync_block();

        compute::UInt segment_end = compute::select(compute::UInt(0u), logical_end - 1u, logical_end > 0u);
        compute::Bool found_boundary = false;
        $for(candidate_lane, compute::UInt(0u), logical_end)
        {
            compute::Bool is_boundary =
                flags->read(warp_base + candidate_lane) != compute::Var<FlagT>{FlagT{}};
            if constexpr(HEAD_SEGMENT)
            {
                is_boundary = is_boundary & (candidate_lane > lane);
            }
            else
            {
                is_boundary = is_boundary & (candidate_lane >= lane);
            }
            is_boundary                 = is_boundary & !found_boundary;
            compute::UInt candidate_end = HEAD_SEGMENT ? candidate_lane - 1u : candidate_lane;
            segment_end                 = compute::select(segment_end, candidate_end, is_boundary);
            found_boundary              = found_boundary | is_boundary;
        };

        compute::Var<T> result = input;
        $for(source_lane, lane + 1u, segment_end + 1u)
        {
            result = reduce_op(result, m_shared_mem->read(warp_base + source_lane));
        };
        compute::sync_block();
        return result;
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid  = compute::thread_id();
        compute::UInt3 size = compute::block_size();
        return tid.x + size.x * (tid.y + size.y * tid.z);
    }

  private:
    SmemTypePtr<T> m_shared_mem;
};

}  // namespace luisa::parallel_primitive::details
