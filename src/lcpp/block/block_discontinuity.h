#pragma once

#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>

#include <cstddef>
#include <utility>

namespace luisa::parallel_primitive
{

template <NumericT Type4Byte, size_t BlockSize = details::BLOCK_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockDiscontinuity : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BlockSize * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<Type4Byte> first_element;
        SmemTypePtr<Type4Byte> last_element;
    };
    using TempStorage = tempStorage;

    BlockDiscontinuity()
        : m_shared_data{new SmemType<Type4Byte>{BLOCK_THREADS}, new SmemType<Type4Byte>{BLOCK_THREADS}}
    {
    }

    explicit BlockDiscontinuity(tempStorage shared_data)
        : m_shared_data(shared_data)
    {
    }

    template <typename FlagArray, typename FlagOp>
    void FlagHeads(FlagArray& head_flags, const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input, FlagOp flag_op) const
    {
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> predecessors;
        flag_heads(head_flags, predecessors, input, flag_op, nullptr);
    }

    // Repository-compatible overload: predecessor outputs precede the input array.
    template <typename FlagArray, typename PredecessorArray, typename FlagOp>
    void FlagHeads(FlagArray&                                            head_flags,
                   PredecessorArray&                                     predecessors,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                                                flag_op) const
    {
        flag_heads(head_flags, predecessors, input, flag_op, nullptr);
    }

    template <typename FlagArray, typename FlagOp>
    void FlagHeads(FlagArray&                                            head_flags,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                                                flag_op,
                   const compute::Var<Type4Byte>& tile_predecessor_item) const
    {
        compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD> predecessors;
        flag_heads(head_flags, predecessors, input, flag_op, &tile_predecessor_item);
    }

    template <typename FlagArray, typename PredecessorArray, typename FlagOp>
    void FlagHeads(FlagArray&                                            head_flags,
                   PredecessorArray&                                     predecessors,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                                                flag_op,
                   const compute::Var<Type4Byte>& tile_predecessor_item) const
    {
        flag_heads(head_flags, predecessors, input, flag_op, &tile_predecessor_item);
    }

    template <typename FlagArray, typename FlagOp>
    void FlagTails(FlagArray& tail_flags, const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input, FlagOp flag_op) const
    {
        flag_tails(tail_flags, input, flag_op, nullptr);
    }

    template <typename FlagArray, typename FlagOp>
    void FlagTails(FlagArray&                                            tail_flags,
                   const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                   FlagOp                                                flag_op,
                   const compute::Var<Type4Byte>&                        tile_successor_item) const
    {
        flag_tails(tail_flags, input, flag_op, &tile_successor_item);
    }

    // Compatibility for the original repository's singular spelling.
    template <typename... Args>
    void FlagTail(Args&&... args) const
    {
        FlagTails(std::forward<Args>(args)...);
    }

    template <typename HeadFlagArray, typename TailFlagArray, typename FlagOp>
    void FlagHeadsAndTails(HeadFlagArray&                                        head_flags,
                           TailFlagArray&                                        tail_flags,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp                                                flag_op) const
    {
        FlagHeads(head_flags, input, flag_op);
        FlagTails(tail_flags, input, flag_op);
    }

    template <typename HeadFlagArray, typename TailFlagArray, typename FlagOp>
    void FlagHeadsAndTails(HeadFlagArray&                 head_flags,
                           TailFlagArray&                 tail_flags,
                           const compute::Var<Type4Byte>& tile_successor_item,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp                                                flag_op) const
    {
        FlagHeads(head_flags, input, flag_op);
        FlagTails(tail_flags, input, flag_op, tile_successor_item);
    }

    template <typename HeadFlagArray, typename TailFlagArray, typename FlagOp>
    void FlagHeadsAndTails(HeadFlagArray&                 head_flags,
                           const compute::Var<Type4Byte>& tile_predecessor_item,
                           TailFlagArray&                 tail_flags,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp                                                flag_op) const
    {
        FlagHeads(head_flags, input, flag_op, tile_predecessor_item);
        FlagTails(tail_flags, input, flag_op);
    }

    template <typename HeadFlagArray, typename TailFlagArray, typename FlagOp>
    void FlagHeadsAndTails(HeadFlagArray&                 head_flags,
                           const compute::Var<Type4Byte>& tile_predecessor_item,
                           TailFlagArray&                 tail_flags,
                           const compute::Var<Type4Byte>& tile_successor_item,
                           const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                           FlagOp                                                flag_op) const
    {
        FlagHeads(head_flags, input, flag_op, tile_predecessor_item);
        FlagTails(tail_flags, input, flag_op, tile_successor_item);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BlockSize) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    void share_boundaries(const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input) const
    {
        compute::UInt tid = linear_thread_id();
        m_shared_data.first_element->write(tid, input[0]);
        m_shared_data.last_element->write(tid, input[ITEMS_PER_THREAD - 1u]);
        compute::sync_block();
    }

    template <typename FlagArray, typename PredecessorArray, typename FlagOp>
    void flag_heads(FlagArray&                                            head_flags,
                    PredecessorArray&                                     predecessors,
                    const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                    FlagOp                                                flag_op,
                    const compute::Var<Type4Byte>* tile_predecessor_item) const
    {
        share_boundaries(input);
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            if(item == 0u)
            {
                compute::UInt previous_tid = compute::select(0u, tid - 1u, tid > 0u);
                compute::Var<Type4Byte> predecessor = m_shared_data.last_element->read(previous_tid);
                if(tile_predecessor_item != nullptr)
                {
                    predecessor = compute::select(*tile_predecessor_item, predecessor, tid > 0u);
                    head_flags[item]   = apply_op(flag_op, predecessor, input[item], index);
                    predecessors[item] = predecessor;
                }
                else
                {
                    $if(tid == 0u)
                    {
                        head_flags[item] = 1;
                    }
                    $else
                    {
                        head_flags[item]   = apply_op(flag_op, predecessor, input[item], index);
                        predecessors[item] = predecessor;
                    };
                }
            }
            else
            {
                predecessors[item] = input[item - 1u];
                head_flags[item]   = apply_op(flag_op, input[item - 1u], input[item], index);
            }
        }
        compute::sync_block();
    }

    template <typename FlagArray, typename FlagOp>
    void flag_tails(FlagArray&                                            tail_flags,
                    const compute::ArrayVar<Type4Byte, ITEMS_PER_THREAD>& input,
                    FlagOp                                                flag_op,
                    const compute::Var<Type4Byte>*                        tile_successor_item) const
    {
        share_boundaries(input);
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt next_index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item + 1u);
            if(item + 1u < ITEMS_PER_THREAD)
            {
                tail_flags[item] = apply_op(flag_op, input[item], input[item + 1u], next_index);
            }
            else
            {
                compute::UInt next_tid = compute::select(compute::UInt(BLOCK_THREADS - 1u),
                                                         tid + 1u,
                                                         tid + 1u < compute::UInt(BLOCK_THREADS));
                compute::Var<Type4Byte> successor = m_shared_data.first_element->read(next_tid);
                if(tile_successor_item != nullptr)
                {
                    successor =
                        compute::select(*tile_successor_item, successor, tid + 1u < compute::UInt(BLOCK_THREADS));
                    tail_flags[item] = apply_op(flag_op, input[item], successor, next_index);
                }
                else
                {
                    $if(tid + 1u == compute::UInt(BLOCK_THREADS))
                    {
                        tail_flags[item] = 1;
                    }
                    $else
                    {
                        tail_flags[item] = apply_op(flag_op, input[item], successor, next_index);
                    };
                }
            }
        }
        compute::sync_block();
    }

    template <typename FlagOp>
    static compute::Bool apply_op(FlagOp                         flag_op,
                                  const compute::Var<Type4Byte>& a,
                                  const compute::Var<Type4Byte>& b,
                                  compute::UInt                  b_index)
    {
        if constexpr(requires { flag_op(a, b, b_index); })
        {
            return flag_op(a, b, b_index);
        }
        else
        {
            return flag_op(a, b);
        }
    }

  private:
    tempStorage m_shared_data;
};

}  // namespace luisa::parallel_primitive
