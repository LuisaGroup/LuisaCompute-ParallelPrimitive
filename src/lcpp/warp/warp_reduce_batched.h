// Copyright 2026 LuisaCompute Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>

#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

namespace luisa::parallel_primitive
{

template <typename T, size_t BATCHES, size_t LOGICAL_WARP_THREADS = details::WARP_SIZE, bool SYNC_PHYSICAL_WARP = false>
class WarpReduceBatched : public LuisaModule
{
    static_assert(LOGICAL_WARP_THREADS > 0u && LOGICAL_WARP_THREADS <= details::WARP_SIZE);
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1u)) == 0u);

  public:
    static constexpr size_t MAX_OUTPUTS_PER_THREAD = (BATCHES + LOGICAL_WARP_THREADS - 1u) / LOGICAL_WARP_THREADS;

    WarpReduceBatched() = default;

    template <typename ReductionOp>
    [[nodiscard]] Var<T> Reduce(const compute::Var<std::array<T, BATCHES>>& inputs, ReductionOp reduction_op)
    {
        static_assert(BATCHES > 0u && BATCHES <= LOGICAL_WARP_THREADS,
                      "Use ReduceToStriped or ReduceToBlocked when batches exceed the logical warp size.");
        Var<T> result = inputs[0];
        for(size_t batch = 0u; batch < BATCHES; ++batch)
        {
            Var<T> aggregate = reduce_batch(inputs, batch, reduction_op);
            $if(logical_lane() == compute::UInt(batch))
            {
                result = aggregate;
            };
        }
        return result;
    }

    template <typename ReductionOp>
    void ReduceToStriped(const compute::Var<std::array<T, BATCHES>>&          inputs,
                         compute::Var<std::array<T, MAX_OUTPUTS_PER_THREAD>>& outputs,
                         ReductionOp                                          reduction_op)
    {
        reduce_to_layout<false>(inputs, outputs, reduction_op);
    }

    template <typename ReductionOp>
    void ReduceToBlocked(const compute::Var<std::array<T, BATCHES>>&          inputs,
                         compute::Var<std::array<T, MAX_OUTPUTS_PER_THREAD>>& outputs,
                         ReductionOp                                          reduction_op)
    {
        reduce_to_layout<true>(inputs, outputs, reduction_op);
    }

    [[nodiscard]] Var<T> Sum(const compute::Var<std::array<T, BATCHES>>& inputs)
    {
        return Reduce(inputs, [](const Var<T>& lhs, const Var<T>& rhs) noexcept { return lhs + rhs; });
    }

    void SumToStriped(const compute::Var<std::array<T, BATCHES>>&          inputs,
                      compute::Var<std::array<T, MAX_OUTPUTS_PER_THREAD>>& outputs)
    {
        ReduceToStriped(inputs,
                        outputs,
                        [](const Var<T>& lhs, const Var<T>& rhs) noexcept { return lhs + rhs; });
    }

    void SumToBlocked(const compute::Var<std::array<T, BATCHES>>&          inputs,
                      compute::Var<std::array<T, MAX_OUTPUTS_PER_THREAD>>& outputs)
    {
        ReduceToBlocked(inputs,
                        outputs,
                        [](const Var<T>& lhs, const Var<T>& rhs) noexcept { return lhs + rhs; });
    }

  private:
    template <bool TO_BLOCKED, typename ReductionOp>
    static void reduce_to_layout(const compute::Var<std::array<T, BATCHES>>&          inputs,
                                 compute::Var<std::array<T, MAX_OUTPUTS_PER_THREAD>>& outputs,
                                 ReductionOp                                          reduction_op)
    {
        if constexpr(BATCHES > 0u)
        {
            for(size_t batch = 0u; batch < BATCHES; ++batch)
            {
                constexpr auto max_outputs = MAX_OUTPUTS_PER_THREAD;
                size_t owner_lane = TO_BLOCKED ? batch / max_outputs : batch % LOGICAL_WARP_THREADS;
                size_t output_index = TO_BLOCKED ? batch % max_outputs : batch / LOGICAL_WARP_THREADS;
                Var<T> aggregate = reduce_batch(inputs, batch, reduction_op);
                $if(logical_lane() == compute::UInt(owner_lane))
                {
                    outputs[output_index] = aggregate;
                };
            }
        }
    }

    template <typename ReductionOp>
    static Var<T> reduce_batch(const compute::Var<std::array<T, BATCHES>>& inputs, size_t batch, ReductionOp reduction_op)
    {
        Var<T> lane_zero_result = WarpReduce<T, LOGICAL_WARP_THREADS>().Reduce(inputs[batch], reduction_op);
        return WarpReadLane(lane_zero_result, logical_warp_base());
    }

    static compute::UInt logical_lane()
    {
        return compute::warp_lane_id() % compute::UInt(LOGICAL_WARP_THREADS);
    }

    static compute::UInt logical_warp_base()
    {
        compute::UInt physical_lane = compute::warp_lane_id();
        return physical_lane / compute::UInt(LOGICAL_WARP_THREADS) * compute::UInt(LOGICAL_WARP_THREADS);
    }

    [[maybe_unused]] static constexpr bool sync_physical_warp = SYNC_PHYSICAL_WARP;
};

}  // namespace luisa::parallel_primitive
