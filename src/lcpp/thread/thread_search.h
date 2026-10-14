/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <lcpp/runtime/core.h>
#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{
namespace details
{
    template <typename InputT>
    [[nodiscard]] auto ThreadSearchRead(const InputT& input, compute::UInt index)
    {
        if constexpr(requires { input.read(index); })
        {
            return input.read(index);
        }
        else
        {
            return input[index];
        }
    }
}  // namespace details

template <typename InputT, typename ValueT>
[[nodiscard]] compute::UInt LowerBound(const InputT& input, compute::UInt num_items, const compute::Var<ValueT>& value)
{
    compute::UInt result = 0u;
    $while(num_items > 0u)
    {
        compute::UInt half = num_items >> 1u;
        $if(details::ThreadSearchRead(input, result + half) < value)
        {
            result += half + 1u;
            num_items -= half + 1u;
        }
        $else
        {
            num_items = half;
        };
    };
    return result;
}

template <typename InputT, typename ValueT>
[[nodiscard]] compute::UInt UpperBound(const InputT& input, compute::UInt num_items, const compute::Var<ValueT>& value)
{
    compute::UInt result = 0u;
    $while(num_items > 0u)
    {
        compute::UInt half = num_items >> 1u;
        $if(value < details::ThreadSearchRead(input, result + half))
        {
            num_items = half;
        }
        $else
        {
            result += half + 1u;
            num_items -= half + 1u;
        };
    };
    return result;
}

template <typename InputA, typename InputB>
[[nodiscard]] compute::UInt2 MergePathSearch(
    compute::UInt diagonal, const InputA& a, const InputB& b, compute::UInt a_length, compute::UInt b_length)
{
    compute::UInt split_min = compute::select(0u, diagonal - b_length, diagonal > b_length);
    compute::UInt split_max = compute::min(diagonal, a_length);
    $while(split_min < split_max)
    {
        compute::UInt split_pivot = (split_min + split_max) >> 1u;
        $if(details::ThreadSearchRead(a, split_pivot) <= details::ThreadSearchRead(b, diagonal - split_pivot - 1u))
        {
            split_min = split_pivot + 1u;
        }
        $else
        {
            split_max = split_pivot;
        };
    };
    compute::UInt a_coordinate = compute::min(split_min, a_length);
    return compute::make_uint2(a_coordinate, diagonal - a_coordinate);
}

template <typename InputA, typename InputB>
void MergePathSearch(compute::UInt   diagonal,
                     const InputA&   a,
                     const InputB&   b,
                     compute::UInt   a_length,
                     compute::UInt   b_length,
                     compute::UInt2& path_coordinate)
{
    path_coordinate = MergePathSearch(diagonal, a, b, a_length, b_length);
}
}  // namespace luisa::parallel_primitive
