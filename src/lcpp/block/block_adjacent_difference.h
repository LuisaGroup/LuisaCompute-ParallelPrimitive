#pragma once

#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>

#include <array>
#include <cstddef>

namespace luisa::parallel_primitive
{

template <typename T, size_t BLOCK_DIM_X, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockAdjacentDifference : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static_assert(BLOCK_THREADS > 0u);

  public:
    struct tempStorage
    {
        SmemTypePtr<T> first_items;
        SmemTypePtr<T> last_items;
    };

    BlockAdjacentDifference()
        : m_shared_data{new SmemType<T>{BLOCK_THREADS}, new SmemType<T>{BLOCK_THREADS}}
    {
    }

    explicit BlockAdjacentDifference(tempStorage shared_data)
        : m_shared_data(shared_data)
    {
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractLeft(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                      OutputArray&                                         output,
                      DifferenceOp                                         difference_op) const
    {
        subtract_left(input, output, difference_op, compute::UInt(BLOCK_THREADS * ITEMS_PER_THREAD), nullptr);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractLeft(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                      OutputArray&                                         output,
                      DifferenceOp                                         difference_op,
                      const compute::Var<T>& tile_predecessor_item) const
    {
        subtract_left(input, output, difference_op, compute::UInt(BLOCK_THREADS * ITEMS_PER_THREAD), &tile_predecessor_item);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractLeftPartialTile(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                                 OutputArray&                                         output,
                                 DifferenceOp                                         difference_op,
                                 compute::UInt valid_items) const
    {
        subtract_left(input, output, difference_op, valid_items, nullptr);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractLeftPartialTile(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                                 OutputArray&                                         output,
                                 DifferenceOp                                         difference_op,
                                 compute::UInt                                        valid_items,
                                 const compute::Var<T>& tile_predecessor_item) const
    {
        subtract_left(input, output, difference_op, valid_items, &tile_predecessor_item);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractRight(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                       OutputArray&                                         output,
                       DifferenceOp                                         difference_op) const
    {
        subtract_right(input, output, difference_op, compute::UInt(BLOCK_THREADS * ITEMS_PER_THREAD), nullptr);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractRight(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                       OutputArray&                                         output,
                       DifferenceOp                                         difference_op,
                       const compute::Var<T>& tile_successor_item) const
    {
        subtract_right(input, output, difference_op, compute::UInt(BLOCK_THREADS * ITEMS_PER_THREAD), &tile_successor_item);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractRightPartialTile(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                                  OutputArray&                                         output,
                                  DifferenceOp  difference_op,
                                  compute::UInt valid_items) const
    {
        subtract_right(input, output, difference_op, valid_items, nullptr);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void SubtractRightPartialTile(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                                  OutputArray&                                         output,
                                  DifferenceOp           difference_op,
                                  compute::UInt          valid_items,
                                  const compute::Var<T>& tile_successor_item) const
    {
        subtract_right(input, output, difference_op, valid_items, &tile_successor_item);
    }

    template <size_t ITEMS_PER_THREAD, typename FlagArray, typename FlagOp>
    void FlagHeads(FlagArray& head_flags, const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input, FlagOp flag_op) const
    {
        flag_heads(head_flags, input, flag_op);
    }

    template <size_t ITEMS_PER_THREAD, typename FlagArray, typename FlagOp>
    void FlagTails(FlagArray& tail_flags, const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input, FlagOp flag_op) const
    {
        flag_tails(tail_flags, input, flag_op);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void subtract_left(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                       OutputArray&                                         output,
                       DifferenceOp                                         difference_op,
                       compute::UInt                                        valid_items,
                       const compute::Var<T>* tile_predecessor_item) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            source[item] = input[item];
        }
        compute::UInt tid                = linear_thread_id();
        (*m_shared_data.last_items)[tid] = source[ITEMS_PER_THREAD - 1u];
        compute::sync_block();

        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt   index  = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            compute::Var<T> result = source[item];
            if(item > 0u)
            {
                result = difference_op(source[item], source[item - 1u]);
            }
            else
            {
                compute::UInt previous_tid = compute::select(compute::UInt(0u), tid - 1u, tid > 0u);
                compute::Var<T> previous   = (*m_shared_data.last_items)[previous_tid];
                compute::Var<T> cross_thread = difference_op(source[0], previous);
                if(tile_predecessor_item != nullptr)
                {
                    compute::Var<T> first = difference_op(source[0], *tile_predecessor_item);
                    result                = compute::select(first, cross_thread, tid > 0u);
                }
                else
                {
                    result = compute::select(source[0], cross_thread, tid > 0u);
                }
            }
            output[item] = compute::select(source[item], result, index < valid_items);
        }
        compute::sync_block();
    }

    template <size_t ITEMS_PER_THREAD, typename OutputArray, typename DifferenceOp>
    void subtract_right(const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input,
                        OutputArray&                                         output,
                        DifferenceOp                                         difference_op,
                        compute::UInt                                        valid_items,
                        const compute::Var<T>* tile_successor_item) const
    {
        compute::ArrayVar<T, ITEMS_PER_THREAD> source;
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            source[item] = input[item];
        }
        compute::UInt tid                 = linear_thread_id();
        (*m_shared_data.first_items)[tid] = source[0];
        compute::sync_block();

        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::UInt   index  = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item);
            compute::Var<T> result = source[item];
            if(item + 1u < ITEMS_PER_THREAD)
            {
                result = difference_op(source[item], source[item + 1u]);
            }
            else
            {
                compute::UInt   next_tid     = compute::select(compute::UInt(BLOCK_THREADS - 1u),
                                                         tid + 1u,
                                                         tid + 1u < compute::UInt(BLOCK_THREADS));
                compute::Var<T> next         = (*m_shared_data.first_items)[next_tid];
                compute::Var<T> cross_thread = difference_op(source[item], next);
                if(tile_successor_item != nullptr)
                {
                    compute::Var<T> last = difference_op(source[item], *tile_successor_item);
                    result = compute::select(last, cross_thread, tid + 1u < compute::UInt(BLOCK_THREADS));
                }
                else
                {
                    result = compute::select(source[item], cross_thread, tid + 1u < compute::UInt(BLOCK_THREADS));
                }
            }

            compute::Bool has_successor = index + 1u < valid_items;
            if(tile_successor_item != nullptr)
            {
                compute::Bool   is_last_valid = index < valid_items & index + 1u == valid_items;
                compute::Var<T> boundary      = difference_op(source[item], *tile_successor_item);
                result                        = compute::select(result, boundary, is_last_valid);
            }
            output[item] = compute::select(
                source[item], result, index < valid_items & (has_successor | index + 1u == valid_items));
        }
        compute::sync_block();
    }

    template <size_t ITEMS_PER_THREAD, typename FlagArray, typename FlagOp>
    void flag_heads(FlagArray& flags, const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input, FlagOp flag_op) const
    {
        compute::UInt tid                = linear_thread_id();
        (*m_shared_data.last_items)[tid] = input[ITEMS_PER_THREAD - 1u];
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::Bool flag;
            if(item > 0u)
            {
                flag = flag_op(input[item], input[item - 1u]);
            }
            else
            {
                compute::UInt previous_tid = compute::select(compute::UInt(0u), tid - 1u, tid > 0u);
                compute::Bool cross = flag_op(input[0], (*m_shared_data.last_items)[previous_tid]);
                flag                = compute::select(compute::Bool(true), cross, tid > 0u);
            }
            flags[item] = compute::select(compute::Int(0), compute::Int(1), flag);
        }
        compute::sync_block();
    }

    template <size_t ITEMS_PER_THREAD, typename FlagArray, typename FlagOp>
    void flag_tails(FlagArray& flags, const compute::Var<std::array<T, ITEMS_PER_THREAD>>& input, FlagOp flag_op) const
    {
        compute::UInt tid                 = linear_thread_id();
        (*m_shared_data.first_items)[tid] = input[0];
        compute::sync_block();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            compute::Bool flag;
            if(item + 1u < ITEMS_PER_THREAD)
            {
                flag = flag_op(input[item], input[item + 1u]);
            }
            else
            {
                compute::UInt next_tid = compute::select(compute::UInt(BLOCK_THREADS - 1u),
                                                         tid + 1u,
                                                         tid + 1u < compute::UInt(BLOCK_THREADS));
                compute::Bool cross = flag_op(input[item], (*m_shared_data.first_items)[next_tid]);
                flag = compute::select(compute::Bool(true), cross, tid + 1u < compute::UInt(BLOCK_THREADS));
            }
            flags[item] = compute::select(compute::Int(0), compute::Int(1), flag);
        }
        compute::sync_block();
    }

    tempStorage m_shared_data;
};

}  // namespace luisa::parallel_primitive
