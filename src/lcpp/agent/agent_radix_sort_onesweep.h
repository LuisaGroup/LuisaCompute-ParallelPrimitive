/*
 * @Author: Ligo
 * @Date: 2025-11-12 15:04:20
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 15:38:14
 */

#pragma once
#include <cstddef>
#include <lcpp/agent/policy.h>
#include <lcpp/agent/radix_rank_sort_operations.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_radix_rank.h>
#include <lcpp/common/grid_even_shared.h>
#include <lcpp/common/cuda_tile_state.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/warp/warp_reduce.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/stmt.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <type_traits>

namespace luisa::parallel_primitive
{

template <uint NominalBlockThreads4B, uint NominalItemsPerThread4B, typename ComputeT, uint RankNumParts, uint RadixBits>
struct AgentRadixSortOneSweepPolicy : RegBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>
{
    static constexpr uint RANK_NUM_PARTS = RankNumParts;
    static constexpr uint RADIX_BITS     = RadixBits;
};

namespace details
{
    using namespace luisa::compute;
    template <NumericT KeyType, NumericT ValueType, bool KEYS_ONLY, size_t RADIX_BITS, size_t RANK_NUM_PARTS, bool IS_DESCENDING, size_t BLOCK_SIZE, size_t WARP_SIZE, size_t ITEMS_PER_THREAD>
    class AgentRadixSortOneSweep : public LuisaModule
    {
      public:
        // constants
        static constexpr uint TILE_ITEMS            = BLOCK_SIZE * ITEMS_PER_THREAD;
        static constexpr uint RADIX_DIGITS          = 1 << RADIX_BITS;
        static constexpr uint BINS_PER_THREAD       = (RADIX_DIGITS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        static constexpr bool FULL_BINS             = BINS_PER_THREAD * BLOCK_SIZE == RADIX_DIGITS;
        static constexpr uint WARP_THREADS          = WARP_SIZE;
        static constexpr uint BLOCK_WARPS           = BLOCK_SIZE / WARP_THREADS;
        static constexpr uint RANK_STORAGE_ITEMS    = BLOCK_WARPS * RADIX_DIGITS * RANK_NUM_PARTS;
        static constexpr uint WARP_MASK             = ~0;
        static constexpr uint LOOKBACK_PARTIAL_MASK = 1 << (uint(sizeof(uint)) * 8 - 2);
        static constexpr uint LOOKBACK_GLOBAL_MASK  = 1 << (uint(sizeof(uint)) * 8 - 1);
        static constexpr uint LOOKBACK_KIND_MASK    = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK;
        static constexpr uint LOOKBACK_VALUE_MASK   = ~LOOKBACK_KIND_MASK;

        using traits                 = radix::traits_t<KeyType>;
        using bit_ordered_type       = typename traits::bit_ordered_type;
        using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

        using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyType>;
        using digit_extractor_t = typename traits::template digit_extractor_t<fundamental_digit_extractor_t>;

        using Twiddle = RadixSortTwiddle<IS_DESCENDING, KeyType>;

        using BlockRadixRankT =
            BlockRadixRankMatchEarlyCounts<BLOCK_SIZE, RADIX_BITS, IS_DESCENDING, WarpMatchAlgorithm::WARP_MATCH_ANY>;


        static inline Callable ThreadBin = [](UInt u) -> UInt
        { return thread_id().x * BINS_PER_THREAD + u; };

        digit_extractor_t digit_extractor() const
        {
            return digit_extractor_t(current_bit, num_bits);
        }

        UInt Digit(Var<bit_ordered_type> key) { return digit_extractor().Digit(key); };

        struct CountsCallback
        {
            using RadixSortOneSweepPolicy =
                AgentRadixSortOneSweepPolicy<BLOCK_SIZE, ITEMS_PER_THREAD, KeyType, 1u, RADIX_BITS>;
            using AgentT =
                AgentRadixSortOneSweep<KeyType, ValueType, KEYS_ONLY, RADIX_BITS, RadixSortOneSweepPolicy::RANK_NUM_PARTS, IS_DESCENDING, BLOCK_SIZE, WARP_SIZE, ITEMS_PER_THREAD>;
            AgentT&                                       agent;
            ArrayVar<uint, BINS_PER_THREAD>&              bins;
            ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys;
            CountsCallback(AgentT&                                       agent,
                           ArrayVar<uint, BINS_PER_THREAD>&              bins,
                           ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
                : agent(agent)
                , bins(bins)
                , keys(keys)
            {
            }
            void operator()(const ArrayVar<uint, BINS_PER_THREAD>& other_bins)
            {
                for(auto u = 0u; u < BINS_PER_THREAD; ++u)
                {
                    bins[u] = other_bins[u];
                }
                agent.LookbackPartial(bins);
                // agent.TryShortCircuit(keys, bins);
            }
        };

        AgentRadixSortOneSweep(BufferVar<uint>&            d_lookback,
                               BufferVar<uint>&            d_ctrs,
                               const BufferVar<uint>&      d_bins_in,
                               BufferVar<uint>&            d_bins_out,
                               const ByteBufferVar&        keys_in,
                               ByteBufferVar&              keys_out,
                               const BufferVar<ValueType>& values_in,
                               BufferVar<ValueType>&       values_out,
                               UInt                        num_items,
                               UInt                        current_bit,
                               UInt                        num_bits,
                               bool                        cuda_tile_state = false)
            : d_lookback(d_lookback)
            , d_ctrs(d_ctrs)
            , d_bins_in(d_bins_in)
            , d_bins_out(d_bins_out)
            , d_keys_in(keys_in)
            , d_keys_out(keys_out)
            , d_values_in(values_in)
            , d_values_out(values_out)
            , num_items(num_items)
            , current_bit(current_bit)
            , num_bits(num_bits)
            , warp(thread_id().x / UInt(WARP_SIZE))
            , lane_id(warp_lane_id())
        {
            m_cuda_tile_state = cuda_tile_state;
            if constexpr(std::is_same_v<bit_ordered_type, uint>)
            {
                // Rank offsets are dead after RankKeys. Reuse the key tile
                // allocation for 32-bit keys instead of reserving a second
                // shared array for the rank workspace.
                constexpr uint shared_items = TILE_ITEMS > RANK_STORAGE_ITEMS ? TILE_ITEMS : RANK_STORAGE_ITEMS;
                m_shared_rank_storage = new SmemType<uint>(shared_items);
                m_shared_keys         = m_shared_rank_storage;
            }
            else
            {
                m_shared_keys         = new SmemType<bit_ordered_type>(TILE_ITEMS);
                m_shared_rank_storage = new SmemType<uint>(RANK_STORAGE_ITEMS);
            }
            m_shared_block_idx = new SmemType<uint>(1);
            m_shared_values    = KEYS_ONLY ? nullptr : new SmemType<ValueType>(TILE_ITEMS);
            m_global_offsets   = new SmemType<uint>(RADIX_DIGITS);

            $if(thread_id().x == 0)
            {
                m_shared_block_idx->write(0, d_ctrs.atomic(0u).fetch_add(1u));
            };

            sync_block();
            block_idx  = m_shared_block_idx->read(0);
            full_block = (block_idx + 1u) * UInt(TILE_ITEMS) <= num_items;
        }


        void LoadKeys(UInt tile_offset, ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys)
        {
            $if(full_block)
            {
                LoadDirectWarpStriped<bit_ordered_type, ITEMS_PER_THREAD>(thread_id().x, d_keys_in, tile_offset, keys);
            }
            $else
            {
                LoadDirectWarpStriped<bit_ordered_type, ITEMS_PER_THREAD>(
                    thread_id().x, d_keys_in, tile_offset, keys, num_items - tile_offset, Twiddle::DefaultKey());
            };

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                keys[i] = Twiddle::In(keys[i]);
            }
        }

        void LoadValues(UInt tile_offset, ArrayVar<ValueType, ITEMS_PER_THREAD>& values)
        {
            $if(full_block)
            {
                LoadDirectWarpStriped<ValueType, ITEMS_PER_THREAD>(thread_id().x, d_values_in, tile_offset, values);
            }
            $else
            {
                LoadDirectWarpStriped<ValueType, ITEMS_PER_THREAD>(
                    thread_id().x, d_values_in, tile_offset, values, num_items - tile_offset);
            };
        }

        void ComputeKeyDigits(ArrayVar<uint, ITEMS_PER_THREAD>& digits)
        {
            for(auto u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                UInt idx  = thread_id().x + u * UInt(BLOCK_SIZE);
                digits[u] = Digit(m_shared_keys->read(idx));
            }
        }

        void GatherScatterValues(const ArrayVar<uint, ITEMS_PER_THREAD>& ranks)
        {
            ArrayVar<uint, ITEMS_PER_THREAD> digits;
            ComputeKeyDigits(digits);

            ArrayVar<ValueType, ITEMS_PER_THREAD> values;
            LoadValues(block_idx * TILE_ITEMS, values);

            sync_block();
            ScatterValuesShared(values, ranks);
            sync_block();
            ScatterValuesGlobal(digits);
        }

        void Process()
        {
            ArrayVar<bit_ordered_type, ITEMS_PER_THREAD> keys;
            LoadKeys(block_idx * TILE_ITEMS, keys);

            ArrayVar<uint, ITEMS_PER_THREAD> ranks;
            ArrayVar<uint, BINS_PER_THREAD>  exclusive_digit_prefix;
            ArrayVar<uint, BINS_PER_THREAD>  bins;
            BlockRadixRankT(m_shared_rank_storage).template RankKeys<bit_ordered_type, ITEMS_PER_THREAD, digit_extractor_t, CountsCallback>(
                keys, ranks, digit_extractor(), exclusive_digit_prefix, CountsCallback(*this, bins, keys));

            sync_block();
            ScatterKeysShared(keys, ranks);

            LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
            LookbackGlobal(bins);
            UpdateBinsGlobal(bins, exclusive_digit_prefix);

            sync_block();
            ScatterKeysGlobal();

            if constexpr(!KEYS_ONLY)
            {
                GatherScatterValues(ranks);
            }
        }


        void LookbackPartial(const ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                UInt bin = ThreadBin(i);
                $if(FULL_BINS | bin < RADIX_DIGITS)
                {
                    UInt value = bins[i] | LOOKBACK_PARTIAL_MASK;
                    if(m_cuda_tile_state)
                    {
                        cuda_tile_state_store<uint>(d_lookback, block_idx * RADIX_DIGITS + bin, value);
                    }
                    else
                    {
                        d_lookback.volatile_write(block_idx * RADIX_DIGITS + bin, value);
                    }
                };
            }
        }

        void TryShortCircuit(ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys,
                             ArrayVar<uint, BINS_PER_THREAD>&              bins)
        {
            // check if any bins can be short-circuited
            UInt short_circuit = 0u;

            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                $if(FULL_BINS | ThreadBin(i) < RADIX_DIGITS)
                {
                    short_circuit = short_circuit | select(0u, 1u, bins[i] == TILE_ITEMS);
                };
            }

            short_circuit = warp_active_bit_or(short_circuit);
            $if(short_circuit != 0u)
            {
                return;
            };

            ShortCircuitCopy(keys, bins);
        }


        void LoadBinsToOffsetsGlobal(const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                UInt bin = ThreadBin(i);
                $if(FULL_BINS | bin < RADIX_DIGITS)
                {
                    UInt global_offset = d_bins_in.read(bin) - offsets[i];
                    m_global_offsets->write(bin, global_offset);
                };
            }
        }

        void LookbackGlobal(const ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto u = 0u; u < BINS_PER_THREAD; ++u)
            {
                UInt bin = ThreadBin(u);
                $if(FULL_BINS | bin < RADIX_DIGITS)
                {
                    UInt inc_sum   = bins[u];
                    Int  want_mask = ~0;

                    Int block_jdx = Int(block_idx) - 1;
                    $while(block_jdx >= 0)
                    {
                        UInt loc_j   = block_jdx * RADIX_DIGITS + bin;
                        UInt value_j = 0;
                        $while(value_j == 0)
                        {
                            if(m_cuda_tile_state)
                            {
                                value_j = cuda_tile_state_load(d_lookback, loc_j);
                            }
                            else
                            {
                                value_j = d_lookback.volatile_read(loc_j);
                            }
                        };
                        inc_sum += value_j & LOOKBACK_VALUE_MASK;
                        want_mask = warp_active_bit_mask((value_j & LOOKBACK_KIND_MASK) == 0).x;
                        $if((value_j & LOOKBACK_GLOBAL_MASK) != 0)
                        {
                            $break;
                        };
                        block_jdx -= 1;
                    };

                    UInt loc_i   = block_idx * RADIX_DIGITS + bin;
                    UInt value_i = inc_sum | LOOKBACK_GLOBAL_MASK;
                    if(m_cuda_tile_state)
                    {
                        cuda_tile_state_store<uint>(d_lookback, loc_i, value_i);
                    }
                    else
                    {
                        d_lookback.volatile_write(loc_i, value_i);
                    }
                    m_global_offsets->write(bin, m_global_offsets->read(bin) + inc_sum - bins[u]);
                };
            }
        }

        void UpdateBinsGlobal(const ArrayVar<uint, BINS_PER_THREAD>& bins,
                              const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            Bool last_block = (block_idx + 1) * RADIX_DIGITS >= num_items;

            $if(last_block)
            {
                for(auto i = 0u; i < BINS_PER_THREAD; ++i)
                {
                    UInt bin = ThreadBin(i);
                    $if(FULL_BINS | bin < RADIX_DIGITS)
                    {
                        d_bins_out.write(bin, m_global_offsets->read(bin) + offsets[i] + bins[i]);
                    };
                }
            };
        }


        void ShortCircuitCopy(ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys,
                              ArrayVar<uint, BINS_PER_THREAD>&              bins)
        {
            UInt common_bits = Digit(keys[0]);

            ArrayVar<uint, BINS_PER_THREAD> offsets;

            for(auto i = 0u; i < BINS_PER_THREAD; ++i)
            {
                UInt bin   = ThreadBin(i);
                offsets[i] = select(0u, TILE_ITEMS, bin > common_bits);
            }

            LoadBinsToOffsetsGlobal(offsets);
            LookbackGlobal(bins);
            UpdateBinsGlobal(bins, offsets);

            sync_block();

            UInt global_offset = m_global_offsets->read(common_bits);

            for(auto i = 0u; i < ITEMS_PER_THREAD; ++i)
            {
                keys[i] = Twiddle::Out(keys[i]);
            }

            $if(full_block)
            {
                StoreDirectWarpStriped<bit_ordered_type, ITEMS_PER_THREAD>(
                    thread_id().x, d_keys_out, global_offset, keys);
            }
            $else
            {
                UInt tile_items = num_items - block_idx * TILE_ITEMS;
                StoreDirectWarpStriped<bit_ordered_type, ITEMS_PER_THREAD>(
                    thread_id().x, d_keys_out, global_offset, keys, tile_items);
            };

            if constexpr(!KEYS_ONLY)
            {
                ArrayVar<ValueType, ITEMS_PER_THREAD> values;
                LoadValues(block_idx * TILE_ITEMS, values);
                $if(full_block)
                {
                    StoreDirectWarpStriped<ValueType, ITEMS_PER_THREAD>(
                        thread_id().x, d_values_out, global_offset, values);
                }
                $else
                {
                    UInt tile_items = num_items - block_idx * TILE_ITEMS;
                    StoreDirectWarpStriped<ValueType, ITEMS_PER_THREAD>(
                        thread_id().x, d_values_out, global_offset, values, tile_items);
                };
            }
        }

        void ScatterKeysShared(const ArrayVar<bit_ordered_type, ITEMS_PER_THREAD>& keys,
                               const ArrayVar<uint, ITEMS_PER_THREAD>&             ranks)
        {
            // write to shared memory
            for(int u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                m_shared_keys->write(ranks[u], keys[u]);
            }
        }


        void ScatterKeysGlobal()
        {
            $if(full_block)
            {
                ScatterKeysGlobalDirect<true>();
            }
            $else
            {
                ScatterKeysGlobalDirect<false>();
            };
        }

        template <bool FULL_TILE>
        void ScatterKeysGlobalDirect()
        {
            UInt tile_items = select(num_items - block_idx * UInt(TILE_ITEMS), UInt(TILE_ITEMS), FULL_TILE);

            for(auto u = 0u; u < ITEMS_PER_THREAD; ++u)
            {
                UInt                  idx        = thread_id().x + u * UInt(BLOCK_SIZE);
                Var<bit_ordered_type> key        = m_shared_keys->read(idx);
                UInt                  global_idx = idx + m_global_offsets->read(Digit(key));
                $if(FULL_TILE | idx < tile_items)
                {
                    d_keys_out.write(global_idx * (uint)sizeof(bit_ordered_type), Twiddle::Out(key));
                };
            }
        }

        void ScatterValuesShared(const ArrayVar<ValueType, ITEMS_PER_THREAD>& values,
                                 const ArrayVar<uint, ITEMS_PER_THREAD>&      ranks)
        {
            // write to shared memory
            for(int u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                m_shared_values->write(ranks[u], values[u]);
            }
        }

        void ScatterValuesGlobal(const ArrayVar<uint, ITEMS_PER_THREAD>& Digit)
        {
            $if(full_block)
            {
                ScatterValuesGlobalDirect<true>(Digit);
            }
            $else
            {
                ScatterValuesGlobalDirect<false>(Digit);
            };
        }

        template <bool FULL_TILE>
        void ScatterValuesGlobalDirect(const ArrayVar<uint, ITEMS_PER_THREAD>& Digit)
        {
            UInt tile_items = select(num_items - block_idx * UInt(TILE_ITEMS), UInt(TILE_ITEMS), FULL_TILE);

            for(auto u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                UInt           idx        = thread_id().x + u * UInt(BLOCK_SIZE);
                Var<ValueType> value      = m_shared_values->read(idx);
                UInt           global_idx = idx + m_global_offsets->read(Digit[u]);
                $if(FULL_TILE | idx < tile_items)
                {
                    d_values_out.write(global_idx, value);
                };
            }
        }


      private:
        SmemTypePtr<bit_ordered_type> m_shared_keys;
        SmemTypePtr<uint>              m_shared_rank_storage;
        SmemTypePtr<ValueType>        m_shared_values;
        SmemTypePtr<uint>             m_global_offsets;
        SmemTypePtr<uint>             m_shared_block_idx;

        bool                   m_cuda_tile_state = false;
        BufferVar<uint>&       d_lookback;
        BufferVar<uint>&       d_ctrs;
        BufferVar<uint>&       d_bins_out;
        const BufferVar<uint>& d_bins_in;

        const ByteBufferVar&        d_keys_in;
        ByteBufferVar&              d_keys_out;
        const BufferVar<ValueType>& d_values_in;
        BufferVar<ValueType>&       d_values_out;

        UInt num_items;
        UInt current_bit;
        UInt num_bits;

        UInt warp;
        UInt lane_id;
        UInt block_idx;
        Bool full_block;
    };
};  // namespace details
}  // namespace luisa::parallel_primitive
