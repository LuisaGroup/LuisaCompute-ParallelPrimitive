#pragma once

#include <lcpp/agent/radix_rank_sort_operations.h>
#include <lcpp/block/block_exchange.h>
#include <lcpp/common/null_type.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/runtime/core.h>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace luisa::parallel_primitive
{

template <typename KeyT, size_t BLOCK_DIM_X, size_t ITEMS_PER_THREAD, typename ValueT = NullType, size_t RADIX_BITS = 4u, size_t BLOCK_DIM_Y = 1u, size_t BLOCK_DIM_Z = 1u>
class BlockRadixSort : public LuisaModule
{
    static constexpr size_t BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr size_t BLOCK_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD;
    static_assert(BLOCK_THREADS > 0u);
    static_assert(ITEMS_PER_THREAD > 0u);
    static_assert(RADIX_BITS > 0u);

    template <typename Decomposer>
    using DecomposedTuple =
        std::remove_cvref_t<decltype(std::declval<Decomposer&>()(std::declval<compute::Var<KeyT>&>()))>;

    template <typename Tuple, size_t... I>
    [[nodiscard]] static consteval uint decomposed_bit_count_impl(std::index_sequence<I...>)
    {
        static_assert((NumericT<expr_value_t<std::tuple_element_t<I, Tuple>>> && ...),
                      "BlockRadixSort decomposer components must be arithmetic DSL values.");
        return (0u + ... + static_cast<uint>(sizeof(expr_value_t<std::tuple_element_t<I, Tuple>>) * 8u));
    }

    template <typename Decomposer>
    [[nodiscard]] static consteval uint decomposed_bit_count()
    {
        using Tuple = DecomposedTuple<Decomposer>;
        static_assert(std::tuple_size_v<Tuple> > 0u, "BlockRadixSort decomposer result must not be empty.");
        return decomposed_bit_count_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }

  public:
    struct tempStorage
    {
        SmemTypePtr<KeyT> ordered_keys;
    };
    using TempStorage = tempStorage;

    BlockRadixSort()
        : m_shared_keys(new SmemType<KeyT>{BLOCK_ITEMS})
    {
    }

    explicit BlockRadixSort(tempStorage storage)
        : m_shared_keys(storage.ordered_keys)
    {
    }

    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
              compute::UInt                              begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_keys<false, false>(keys, begin_bit, end_bit);
    }

    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              compute::UInt                                begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_pairs<false, false>(keys, values, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
              Decomposer                                 decomposer,
              compute::UInt                              begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_keys<false, false>(keys, decomposer, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void Sort(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
              Decomposer                                   decomposer,
              compute::UInt                                begin_bit = 0u,
              compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_pairs<false, false>(keys, values, decomposer, begin_bit, end_bit);
    }

    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                        compute::UInt                              begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_keys<true, false>(keys, begin_bit, end_bit);
    }

    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                        compute::UInt                                begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_pairs<true, false>(keys, values, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                        Decomposer                                 decomposer,
                        compute::UInt                              begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_keys<true, false>(keys, decomposer, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortDescending(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                        Decomposer                                   decomposer,
                        compute::UInt                                begin_bit = 0u,
                        compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_pairs<true, false>(keys, values, decomposer, begin_bit, end_bit);
    }

    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              compute::UInt                              begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_keys<false, true>(keys, begin_bit, end_bit);
    }

    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                              compute::UInt                                begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_pairs<false, true>(keys, values, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              Decomposer                                 decomposer,
                              compute::UInt                              begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_keys<false, true>(keys, decomposer, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                              compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                              Decomposer                                   decomposer,
                              compute::UInt                                begin_bit = 0u,
                              compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_pairs<false, true>(keys, values, decomposer, begin_bit, end_bit);
    }

    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                                        compute::UInt                              begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_keys<true, true>(keys, begin_bit, end_bit);
    }

    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                                        compute::UInt                                begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(sizeof(KeyT) * 8u)) const
        requires NumericT<KeyT>
    {
        sort_numeric_pairs<true, true>(keys, values, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                                        Decomposer                                 decomposer,
                                        compute::UInt                              begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_keys<true, true>(keys, decomposer, begin_bit, end_bit);
    }

    template <typename Decomposer>
        requires requires(Decomposer& decomposer, compute::Var<KeyT>& key) { decomposer(key); }
    void SortDescendingBlockedToStriped(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                                        compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                                        Decomposer                                   decomposer,
                                        compute::UInt                                begin_bit = 0u,
                                        compute::UInt end_bit = compute::UInt(decomposed_bit_count<Decomposer>())) const
    {
        sort_decomposed_pairs<true, true>(keys, values, decomposer, begin_bit, end_bit);
    }

  private:
    static compute::UInt linear_thread_id()
    {
        compute::UInt3 tid = compute::thread_id();
        return tid.x + compute::UInt(BLOCK_DIM_X) * (tid.y + compute::UInt(BLOCK_DIM_Y) * tid.z);
    }

    static auto ordered_bits(const compute::Var<KeyT>& key)
        requires NumericT<KeyT>
    {
        using UnsignedBits              = typename Traits<KeyT>::UnsignedBits;
        compute::Var<UnsignedBits> bits = Traits<KeyT>::TwiddleIn(compute::as<UnsignedBits>(key));
        return BaseDigitExtractor<KeyT>::ProcessFloatMinusZero(bits);
    }

    static auto selected_bits(const compute::Var<KeyT>& key, compute::UInt begin_bit, compute::UInt end_bit)
        requires NumericT<KeyT>
    {
        using UnsignedBits                   = typename Traits<KeyT>::UnsignedBits;
        constexpr auto             BIT_WIDTH = sizeof(KeyT) * 8u;
        compute::Var<UnsignedBits> selected  = UnsignedBits{0};
        $if(end_bit > begin_bit)
        {
            compute::UInt width = compute::min(end_bit - begin_bit, compute::UInt(BIT_WIDTH));
            compute::Var<UnsignedBits> mask =
                UnsignedBits(-1) >> compute::cast<UnsignedBits>(compute::UInt(BIT_WIDTH) - width);
            selected = (ordered_bits(key) >> compute::cast<UnsignedBits>(begin_bit)) & mask;
        };
        return selected;
    }

    template <size_t INDEX, typename Tuple>
    [[nodiscard]] static consteval uint decomposed_suffix_bits()
    {
        if constexpr(INDEX + 1u == std::tuple_size_v<Tuple>)
        {
            return 0u;
        }
        else
        {
            using NextT = expr_value_t<std::tuple_element_t<INDEX + 1u, Tuple>>;
            return static_cast<uint>(sizeof(NextT) * 8u) + decomposed_suffix_bits<INDEX + 1u, Tuple>();
        }
    }

    template <typename Value>
    [[nodiscard]] static auto decomposed_ordered_bits(const Value& value)
    {
        using ComponentT                = expr_value_t<Value>;
        using UnsignedBits              = typename Traits<ComponentT>::UnsignedBits;
        compute::Var<UnsignedBits> bits = compute::as<UnsignedBits>(value);
        bits                            = Traits<ComponentT>::TwiddleIn(bits);
        return BaseDigitExtractor<ComponentT>::ProcessFloatMinusZero(bits);
    }

    template <size_t INDEX, typename Tuple>
    static void compare_decomposed_component(const Tuple&   lhs,
                                             const Tuple&   rhs,
                                             compute::UInt  begin_bit,
                                             compute::UInt  end_bit,
                                             compute::Bool& less,
                                             compute::Bool& equal)
    {
        using ComponentT               = expr_value_t<std::tuple_element_t<INDEX, Tuple>>;
        using UnsignedBits             = typename Traits<ComponentT>::UnsignedBits;
        constexpr uint component_bits  = static_cast<uint>(sizeof(ComponentT) * 8u);
        constexpr uint component_begin = decomposed_suffix_bits<INDEX, Tuple>();
        constexpr uint component_end   = component_begin + component_bits;
        compute::UInt  overlap_begin   = compute::max(begin_bit, compute::UInt(component_begin));
        compute::UInt  overlap_end     = compute::min(end_bit, compute::UInt(component_end));
        compute::Bool  active          = overlap_begin < overlap_end;
        compute::UInt  width           = compute::select(0u, overlap_end - overlap_begin, active);
        compute::UInt  safe_width      = compute::select(1u, width, active);
        compute::UInt  local_begin =
            compute::select(0u, overlap_begin - compute::UInt(component_begin), active);
        compute::Var<UnsignedBits> mask =
            UnsignedBits(-1) >> compute::cast<UnsignedBits>(compute::UInt(component_bits) - safe_width);
        auto lhs_bits =
            (decomposed_ordered_bits(std::get<INDEX>(lhs)) >> compute::cast<UnsignedBits>(local_begin)) & mask;
        auto rhs_bits =
            (decomposed_ordered_bits(std::get<INDEX>(rhs)) >> compute::cast<UnsignedBits>(local_begin)) & mask;
        less  = less | (equal & active & (lhs_bits < rhs_bits));
        equal = equal & (!active | (lhs_bits == rhs_bits));
        if constexpr(INDEX + 1u < std::tuple_size_v<Tuple>)
        {
            compare_decomposed_component<INDEX + 1u>(lhs, rhs, begin_bit, end_bit, less, equal);
        }
    }

    template <typename Tuple>
    [[nodiscard]] static compute::Bool decomposed_less(const Tuple&  lhs,
                                                       const Tuple&  rhs,
                                                       compute::UInt begin_bit,
                                                       compute::UInt end_bit)
    {
        compute::Bool less  = false;
        compute::Bool equal = true;
        compare_decomposed_component<0u>(lhs, rhs, begin_bit, end_bit, less, equal);
        return less;
    }

    template <bool DESCENDING, typename Decomposer>
    [[nodiscard]] static auto decomposed_compare(Decomposer decomposer, compute::UInt begin_bit, compute::UInt end_bit)
    {
        return [decomposer, begin_bit, end_bit](compute::Var<KeyT> lhs, compute::Var<KeyT> rhs) noexcept
        {
            auto lhs_tuple = decomposer(lhs);
            auto rhs_tuple = decomposer(rhs);
            if constexpr(DESCENDING)
            {
                return decomposed_less(rhs_tuple, lhs_tuple, begin_bit, end_bit);
            }
            else
            {
                return decomposed_less(lhs_tuple, rhs_tuple, begin_bit, end_bit);
            }
        };
    }

    template <typename CompareOp>
    void compute_ranks(const compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                       compute::ArrayVar<uint, ITEMS_PER_THREAD>&       ranks,
                       CompareOp                                        compare_op) const
    {
        compute::UInt tid = linear_thread_id();
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            m_shared_keys->write(tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(item), keys[item]);
        }
        compute::sync_block();

        for(size_t target_item = 0u; target_item < ITEMS_PER_THREAD; ++target_item)
        {
            compute::UInt target_index = tid * compute::UInt(ITEMS_PER_THREAD) + compute::UInt(target_item);
            compute::UInt rank = 0u;
            $for(source_index, compute::UInt(BLOCK_ITEMS))
            {
                compute::Var<KeyT> candidate        = m_shared_keys->read(source_index);
                compute::Bool      candidate_before = compare_op(candidate, keys[target_item]);
                compute::Bool      target_before    = compare_op(keys[target_item], candidate);
                compute::Bool      equivalent       = !(candidate_before | target_before);
                compute::Bool precedes = candidate_before | (equivalent & (source_index < target_index));
                rank += compute::select(0u, 1u, precedes);
            };
            ranks[target_item] = rank;
        }
        compute::sync_block();
    }

    template <bool STRIPED, typename CompareOp>
    void sort_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys, CompareOp compare_op) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD> ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD> sorted_keys;
        compute_ranks(keys, ranks, compare_op);
        scatter<STRIPED, KeyT>(keys, sorted_keys, ranks);
        copy_items<KeyT>(sorted_keys, keys);
    }

    template <bool STRIPED, typename CompareOp>
    void sort_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                    compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                    CompareOp                                    compare_op) const
    {
        compute::ArrayVar<uint, ITEMS_PER_THREAD>   ranks;
        compute::ArrayVar<KeyT, ITEMS_PER_THREAD>   sorted_keys;
        compute::ArrayVar<ValueT, ITEMS_PER_THREAD> sorted_values;
        compute_ranks(keys, ranks, compare_op);
        scatter<STRIPED, KeyT>(keys, sorted_keys, ranks);
        scatter<STRIPED, ValueT>(values, sorted_values, ranks);
        copy_items<KeyT>(sorted_keys, keys);
        copy_items<ValueT>(sorted_values, values);
    }

    template <bool DESCENDING, bool STRIPED>
    void sort_numeric_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                           compute::UInt                              begin_bit,
                           compute::UInt                              end_bit) const
        requires NumericT<KeyT>
    {
        auto compare = [begin_bit, end_bit](compute::Var<KeyT> lhs, compute::Var<KeyT> rhs) noexcept
        {
            auto lhs_bits = selected_bits(lhs, begin_bit, end_bit);
            auto rhs_bits = selected_bits(rhs, begin_bit, end_bit);
            if constexpr(DESCENDING)
            {
                return lhs_bits > rhs_bits;
            }
            else
            {
                return lhs_bits < rhs_bits;
            }
        };
        sort_keys<STRIPED>(keys, compare);
    }

    template <bool DESCENDING, bool STRIPED>
    void sort_numeric_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                            compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                            compute::UInt                                begin_bit,
                            compute::UInt                                end_bit) const
        requires NumericT<KeyT>
    {
        auto compare = [begin_bit, end_bit](compute::Var<KeyT> lhs, compute::Var<KeyT> rhs) noexcept
        {
            auto lhs_bits = selected_bits(lhs, begin_bit, end_bit);
            auto rhs_bits = selected_bits(rhs, begin_bit, end_bit);
            if constexpr(DESCENDING)
            {
                return lhs_bits > rhs_bits;
            }
            else
            {
                return lhs_bits < rhs_bits;
            }
        };
        sort_pairs<STRIPED>(keys, values, compare);
    }

    template <bool DESCENDING, bool STRIPED, typename Decomposer>
    void sort_decomposed_keys(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>& keys,
                              Decomposer                                 decomposer,
                              compute::UInt                              begin_bit,
                              compute::UInt                              end_bit) const
    {
        sort_keys<STRIPED>(keys, decomposed_compare<DESCENDING>(decomposer, begin_bit, end_bit));
    }

    template <bool DESCENDING, bool STRIPED, typename Decomposer>
    void sort_decomposed_pairs(compute::ArrayVar<KeyT, ITEMS_PER_THREAD>&   keys,
                               compute::ArrayVar<ValueT, ITEMS_PER_THREAD>& values,
                               Decomposer                                   decomposer,
                               compute::UInt                                begin_bit,
                               compute::UInt                                end_bit) const
    {
        sort_pairs<STRIPED>(keys, values, decomposed_compare<DESCENDING>(decomposer, begin_bit, end_bit));
    }

    template <bool STRIPED, typename ItemT>
    static void scatter(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input,
                        compute::ArrayVar<ItemT, ITEMS_PER_THREAD>&       output,
                        const compute::ArrayVar<uint, ITEMS_PER_THREAD>&  ranks)
    {
        BlockExchange<ItemT, BLOCK_DIM_X, ITEMS_PER_THREAD, BLOCK_DIM_Y, BLOCK_DIM_Z> exchange;
        if constexpr(STRIPED)
        {
            exchange.ScatterToStriped(input, output, ranks);
        }
        else
        {
            exchange.ScatterToBlocked(input, output, ranks);
        }
    }

    template <typename ItemT>
    static void copy_items(const compute::ArrayVar<ItemT, ITEMS_PER_THREAD>& input,
                           compute::ArrayVar<ItemT, ITEMS_PER_THREAD>&       output)
    {
        for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
        {
            output[item] = input[item];
        }
    }

  private:
    SmemTypePtr<KeyT> m_shared_keys;
};

}  // namespace luisa::parallel_primitive
