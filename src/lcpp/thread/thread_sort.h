// Copyright 2026 LuisaCompute Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>

#include <luisa/dsl/sugar.h>

namespace luisa::parallel_primitive
{

template <typename KeyT, typename ValueT, size_t ITEMS_PER_THREAD, typename CompareOp>
void StableOddEvenSort(compute::Var<std::array<KeyT, ITEMS_PER_THREAD>>&   keys,
                       compute::Var<std::array<ValueT, ITEMS_PER_THREAD>>& values,
                       CompareOp                                           compare_op)
{
    for(size_t pass = 0u; pass < ITEMS_PER_THREAD; ++pass)
    {
        for(size_t item = pass & 1u; item + 1u < ITEMS_PER_THREAD; item += 2u)
        {
            $if(compare_op(keys[item + 1u], keys[item]))
            {
                compute::Var<KeyT> key = keys[item];
                keys[item]             = keys[item + 1u];
                keys[item + 1u]        = key;

                compute::Var<ValueT> value = values[item];
                values[item]               = values[item + 1u];
                values[item + 1u]          = value;
            };
        }
    }
}

template <typename KeyT, size_t ITEMS_PER_THREAD, typename CompareOp>
void StableOddEvenSort(compute::Var<std::array<KeyT, ITEMS_PER_THREAD>>& keys, CompareOp compare_op)
{
    for(size_t pass = 0u; pass < ITEMS_PER_THREAD; ++pass)
    {
        for(size_t item = pass & 1u; item + 1u < ITEMS_PER_THREAD; item += 2u)
        {
            $if(compare_op(keys[item + 1u], keys[item]))
            {
                compute::Var<KeyT> key = keys[item];
                keys[item]             = keys[item + 1u];
                keys[item + 1u]        = key;
            };
        }
    }
}

}  // namespace luisa::parallel_primitive
