/*
 * @Author: Ligo
 * @Date: 2025-11-13 16:31:28
 * @Last Modified by: Ligo
 * @Last Modified time: 2026-02-06 10:12:57
 */
#pragma once
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/core/basic_traits.h>
#include <lcpp/block/detail/block_scan_warp.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/thread/thread_reduce.h>
#include <lcpp/thread/thread_scan.h>
#include <lcpp/block/block_scan.h>

#include <limits>

namespace luisa::parallel_primitive
{
enum class RadixRankAlgorithm
{
    SHARED_MEMORY,
    WARP_SHUFFLE
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};
using namespace luisa::compute;

// empty callback
template <int BINS_PER_THREAD>
struct BlockRadixRankEmptyCallback
{
    inline void operator()(const ArrayVar<int, BINS_PER_THREAD>&) {}
};

namespace details
{
    template <int Bits, int PartialWarpThreads, int PartialWarpId>
    struct warp_in_block_matcher_t
    {
        inline static Callable match_any = [](UInt label, UInt warp_id)
        {
            $if(warp_id == UInt(PartialWarpId))
            {
                return MatchAny<Bits, PartialWarpThreads>(label);
            };

            return MatchAny<Bits>(label);
        };
    };

    template <int Bits, int PartialWarpId>
    struct warp_in_block_matcher_t<Bits, 0, PartialWarpId>
    {
        inline static Callable match_any = [](UInt label, UInt warp_id)
        { return MatchAny<Bits>(label); };
    };
}  // namespace details

template <uint BLOCK_DIM_X, uint RadixBits, bool IsDescending = false, bool MEMOIZE_OUTER_SCAN = true, uint BLOCK_DIM_Y = 1u, uint BLOCK_DIM_Z = 1u>
class BlockRadixRank : public LuisaModule
{
    static constexpr uint BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  public:
    static constexpr uint RADIX_DIGITS = 1u << RadixBits;
    static constexpr uint BINS_TRACKED_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1u) / BLOCK_THREADS;

    struct tempStorage
    {
        SmemTypePtr<uint> thread_digit_counts;
        SmemTypePtr<uint> digit_prefixes;
    };
    using TempStorage = tempStorage;

    BlockRadixRank()
        : m_thread_digit_counts(new SmemType<uint>{BLOCK_THREADS * RADIX_DIGITS})
        , m_digit_prefixes(new SmemType<uint>{RADIX_DIGITS})
    {
    }

    explicit BlockRadixRank(tempStorage storage)
        : m_thread_digit_counts(storage.thread_digit_counts)
        , m_digit_prefixes(storage.digit_prefixes)
    {
    }

    template <NumericT UnsignedBits, size_t KEYS_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const ArrayVar<UnsignedBits, KEYS_PER_THREAD>& keys,
                  ArrayVar<uint, KEYS_PER_THREAD>&               ranks,
                  DigitExtractorT                                digit_extractor) const
    {
        rank_keys<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT>(keys, ranks, digit_extractor);
    }

    template <NumericT UnsignedBits, size_t KEYS_PER_THREAD, typename DigitExtractorT, typename PrefixArray>
    void RankKeys(const ArrayVar<UnsignedBits, KEYS_PER_THREAD>& keys,
                  ArrayVar<uint, KEYS_PER_THREAD>&               ranks,
                  DigitExtractorT                                digit_extractor,
                  PrefixArray&                                   exclusive_digit_prefix) const
    {
        rank_keys<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT>(keys, ranks, digit_extractor);

        UInt tid = linear_thread_id();
        for(uint tracked_bin = 0u; tracked_bin < BINS_TRACKED_PER_THREAD; ++tracked_bin)
        {
            UInt bin    = tid * UInt(BINS_TRACKED_PER_THREAD) + tracked_bin;
            UInt prefix = 0u;
            $if(bin < UInt(RADIX_DIGITS))
            {
                prefix = m_digit_prefixes->read(bin);
            };
            exclusive_digit_prefix[tracked_bin] = prefix;
        }
    }

  private:
    static UInt linear_thread_id()
    {
        UInt3 tid = thread_id();
        return tid.x + UInt(BLOCK_DIM_X) * (tid.y + UInt(BLOCK_DIM_Y) * tid.z);
    }

    static UInt ordered_digit(UInt digit)
    {
        if constexpr(IsDescending)
        {
            return UInt(RADIX_DIGITS - 1u) - digit;
        }
        else
        {
            return digit;
        }
    }

    template <NumericT UnsignedBits, size_t KEYS_PER_THREAD, typename DigitExtractorT>
    void rank_keys(const ArrayVar<UnsignedBits, KEYS_PER_THREAD>& keys,
                   ArrayVar<uint, KEYS_PER_THREAD>&               ranks,
                   DigitExtractorT                                digit_extractor) const
    {
        static_assert(BLOCK_THREADS > 0u);
        static_assert(RadixBits > 0u && RadixBits < 31u);
        static_assert(BLOCK_THREADS * KEYS_PER_THREAD <= std::numeric_limits<uint>::max());
        static_cast<void>(MEMOIZE_OUTER_SCAN);

        UInt tid       = linear_thread_id();
        UInt row_begin = tid * UInt(RADIX_DIGITS);
        $for(bin, UInt(RADIX_DIGITS))
        {
            m_thread_digit_counts->write(row_begin + bin, 0u);
        };
        sync_block();

        ArrayVar<uint, KEYS_PER_THREAD> digits;
        for(uint item = 0u; item < KEYS_PER_THREAD; ++item)
        {
            digits[item]     = ordered_digit(digit_extractor.Digit(keys[item]));
            UInt count_index = row_begin + digits[item];
            m_thread_digit_counts->write(count_index, m_thread_digit_counts->read(count_index) + 1u);
        }
        sync_block();

        $if(tid == 0u)
        {
            UInt running = 0u;
            $for(bin, UInt(RADIX_DIGITS))
            {
                m_digit_prefixes->write(bin, running);
                $for(source_thread, UInt(BLOCK_THREADS))
                {
                    running += m_thread_digit_counts->read(source_thread * UInt(RADIX_DIGITS) + bin);
                };
            };
        };
        sync_block();

        for(uint item = 0u; item < KEYS_PER_THREAD; ++item)
        {
            UInt digit = digits[item];
            UInt rank  = m_digit_prefixes->read(digit);
            $for(source_thread, UInt(0u), tid)
            {
                rank += m_thread_digit_counts->read(source_thread * UInt(RADIX_DIGITS) + digit);
            };
            for(uint previous_item = 0u; previous_item < item; ++previous_item)
            {
                rank += cast<uint>(digits[previous_item] == digit);
            }
            ranks[item] = rank;
        }
        sync_block();
    }

  private:
    SmemTypePtr<uint> m_thread_digit_counts;
    SmemTypePtr<uint> m_digit_prefixes;
};


template <uint BLOCK_THREADS, uint RadixBits, bool IsDescending, WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, uint NUM_PARTS = 1, uint ITEMS_PER_THREAD = details::ITEMS_PER_THREAD, uint WARP_SIZE = details::WARP_SIZE>
class BlockRadixRankMatchEarlyCounts : public LuisaModule
{
  public:
    static constexpr uint RADIX_DIGITS    = 1 << RadixBits;
    static constexpr uint BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
    static constexpr uint BINS_TRACKED_PER_THREAD = BINS_PER_THREAD;
    static constexpr bool FULL_BINS               = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS;
    static constexpr uint PARTIAL_WARP_THREADS    = BLOCK_THREADS % WARP_SIZE;
    static constexpr uint BLOCK_WARPS             = BLOCK_THREADS / WARP_SIZE;
    static constexpr uint PARTIAL_WARP_ID         = BLOCK_WARPS - 1;
    static constexpr uint WARP_MASK               = ~0;
    static constexpr uint NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0;
    static constexpr uint MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS;

  private:
    using BlockScanT = luisa::parallel_primitive::BlockScan<uint, BLOCK_THREADS, BINS_PER_THREAD, WARP_SIZE>;

  public:
    BlockRadixRankMatchEarlyCounts() = default;
    explicit BlockRadixRankMatchEarlyCounts(SmemTypePtr<uint> warp_offsets)
        : m_warp_offsets(warp_offsets)
    {
    }
    ~BlockRadixRankMatchEarlyCounts() = default;

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        SmemTypePtr<uint> m_warp_offsets;
        SmemTypePtr<uint> m_match_masks;
        DigitExtractorT   m_digit_extractor;
        CountsCallback    m_counts_callback;
        UInt              m_warp;
        UInt              m_lane;

        BlockRadixRankMatchInternal(DigitExtractorT digit_extractor,
                                    CountsCallback    callback,
                                    SmemTypePtr<uint> warp_offsets)
            : m_digit_extractor(digit_extractor)
            , m_counts_callback(callback)
            , m_warp(thread_id().x / WARP_SIZE)
            , m_lane(warp_lane_id())
        {
            m_warp_offsets = warp_offsets != nullptr ? warp_offsets : new SmemType<uint>(BLOCK_WARPS * RADIX_DIGITS * NUM_PARTS);
            m_match_masks  = new SmemType<uint>(MATCH_MASKS_ALLOC_SIZE * RADIX_DIGITS);
        }


        UInt Digit(Var<UnsignedBits> key)
        {
            UInt digit = m_digit_extractor.Digit(key);
            return IsDescending ? RADIX_DIGITS - 1 - digit : digit;
        };

        inline static Callable ThreadBin = [](UInt u) -> UInt
        {
            UInt bin = thread_id().x * BINS_PER_THREAD + u;
            return IsDescending ? RADIX_DIGITS - 1 - bin : bin;
        };

        void ComputeHistogramWarp(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys)
        {
            $for(bin, m_lane, UInt(RADIX_DIGITS), UInt(WARP_SIZE))
            {
                for(auto part = 0u; part < NUM_PARTS; ++part)
                {
                    m_warp_offsets->write(m_warp * RADIX_DIGITS * NUM_PARTS + UInt(bin) * NUM_PARTS + part, 0u);
                }
            };

            if constexpr(MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
            {
                $for(bin, m_lane, UInt(RADIX_DIGITS), UInt(WARP_SIZE))
                {
                    m_match_masks->write(bin, 0u);
                };
            }

            // TODO: sync_warp(WARP_MASK);
            sync_block();

            for(auto i = 0u; i < KEY_PER_THREAD; ++i)
            {
                UInt bin = Digit(keys[i]);
                UInt index = m_warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS + (m_lane % NUM_PARTS);
                m_warp_offsets->atomic(index).fetch_add(1u);
            }

            if constexpr(NUM_PARTS > 1)
            {
                // TODO: sync_warp(WARP_MASK);
                sync_block();
                // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
                constexpr uint WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_SIZE;

                ArrayVar<uint, WARP_BINS_PER_THREAD> local_bins;

                for(auto u = 0u; u < WARP_BINS_PER_THREAD; ++u)
                {
                    UInt                      bin = m_lane + u * WARP_SIZE;
                    ArrayVar<uint, NUM_PARTS> bin_counts;
                    for(auto part = 0u; part < NUM_PARTS; ++part)
                    {
                        bin_counts[part] =
                            m_warp_offsets->read(m_warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS + part);
                    }

                    local_bins[u] = ThreadReduce<uint, NUM_PARTS>().Reduce(
                        bin_counts, [](const UInt& a, const UInt& b) { return a + b; });
                }

                sync_block();

                for(auto u = 0u; u < WARP_BINS_PER_THREAD; ++u)
                {
                    UInt bin = m_lane + u * WARP_SIZE;
                    m_warp_offsets->write(m_warp * RADIX_DIGITS * NUM_PARTS + bin * NUM_PARTS, local_bins[u]);
                }
            }
        }

        void ComputeOffsetsWarpUpSweep(ArrayVar<uint, BINS_PER_THREAD>& bins)
        {
            for(auto u = 0u; u < BINS_PER_THREAD; ++u)
            {
                bins[u]  = 0;
                UInt bin = ThreadBin(u);
                $if(FULL_BINS | bin < UInt(RADIX_DIGITS))
                {
                    for(auto j_warp = 0u; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        auto warp_offset = m_warp_offsets->read(j_warp * RADIX_DIGITS + bin);
                        m_warp_offsets->write(j_warp * RADIX_DIGITS + bin, bins[u]);
                        bins[u] += warp_offset;
                    }
                };
            }
        }

        void ComputeOffsetsWarpDownSweep(const ArrayVar<uint, BINS_PER_THREAD>& offsets)
        {
            for(auto u = 0u; u < BINS_PER_THREAD; ++u)
            {
                UInt bin = ThreadBin(u);
                $if(FULL_BINS | bin < RADIX_DIGITS)
                {
                    UInt digit_offset = offsets[u];
                    for(auto j_warp = 0u; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        UInt offset = m_warp_offsets->read(j_warp * RADIX_DIGITS + bin) + digit_offset;
                        m_warp_offsets->write(j_warp * RADIX_DIGITS + bin, offset);
                    }
                };
            };
        }

        void ComputeRanksItem(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                              ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                              details::constant_t<WARP_MATCH_ATOMIC_OR>)
        {
            UInt lane_mask = 1u << m_lane;

            for(auto u = 0u; u < KEY_PER_THREAD; ++u)
            {
                UInt bin = Digit(keys[u]);

                m_match_masks->atomic(bin).fetch_or(lane_mask);

                // TODO: sync_warp(WARP_MASK);
                sync_block();
                UInt bin_mask    = m_match_masks->read(bin);
                UInt leader      = (WARP_SIZE - 1) - luisa::compute::clz(bin_mask);
                UInt warp_offset = 0;
                UInt popc        = popcount(bin_mask & get_lane_mask_le());

                $if(m_lane == leader)
                {
                    // warp_offset
                    warp_offset = m_warp_offsets->atomic(m_warp * RADIX_DIGITS + UInt(bin)).fetch_add(popc);
                };
                warp_offset = warp_read_lane(warp_offset, leader);

                $if(m_lane == leader)
                {
                    m_match_masks->write(bin, 0u);
                };

                // TODO: sync_warp(WARP_MASK);
                sync_block();
                ranks[u] = warp_offset + popc - 1;
            }
        }

        void ComputeRanksItem(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                              ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                              details::constant_t<WARP_MATCH_ANY>)
        {
            for(auto u = 0u; u < KEY_PER_THREAD; ++u)
            {
                UInt bin = Digit(keys[u]);

                UInt bin_mask =
                    details::warp_in_block_matcher_t<RadixBits, PARTIAL_WARP_THREADS, PARTIAL_WARP_ID - 1>::match_any(
                        bin, m_warp);
                UInt leader      = (WARP_SIZE - 1) - luisa::compute::clz(bin_mask);
                UInt warp_offset = 0;
                UInt popc        = popcount(bin_mask & get_lane_mask_le());
                $if(m_lane == leader)
                {
                    // warp_offset
                    warp_offset =
                        m_warp_offsets->atomic(m_warp * UInt(RADIX_DIGITS) + UInt(bin)).fetch_add(popc);
                };
                // __shfl_sync = warp_read_lane
                warp_offset = warp_read_lane(warp_offset, leader);
                ranks[u]    = warp_offset + popc - 1;
            }
        }


        void RankKeys(const ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                      ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                      compute::ArrayVar<uint, BINS_PER_THREAD>&     exclusive_digit_prefix)
        {
            ComputeHistogramWarp(keys);

            sync_block();

            ArrayVar<uint, BINS_PER_THREAD> bins;
            ComputeOffsetsWarpUpSweep(bins);

            m_counts_callback(bins);

            BlockScanT().ExclusiveSum(bins, exclusive_digit_prefix);

            ComputeOffsetsWarpDownSweep(exclusive_digit_prefix);
            sync_block();

            ComputeRanksItem(keys, ranks, details::constant_v<MATCH_ALGORITHM>);
        };
    };


    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT, typename CountsCallback>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_PER_THREAD>&              exclusive_digit_prefix,
                  CountsCallback                                         counts_callback)
    {
        BlockRadixRankMatchInternal<UnsignedBits, KEY_PER_THREAD, DigitExtractorT, CountsCallback> internal(
            digit_extractor, counts_callback, m_warp_offsets);
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor,
                  compute::ArrayVar<uint, BINS_PER_THREAD>&              exclusive_digit_prefix)
    {
        using CountsCallback = BlockRadixRankEmptyCallback<BINS_PER_THREAD>;
        BlockRadixRankMatchInternal<UnsignedBits, KEY_PER_THREAD, DigitExtractorT, CountsCallback> internal(
            digit_extractor, CountsCallback(), m_warp_offsets);
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, uint KEY_PER_THREAD, typename DigitExtractorT>
    void RankKeys(const compute::ArrayVar<UnsignedBits, KEY_PER_THREAD>& keys,
                  compute::ArrayVar<uint, KEY_PER_THREAD>&               ranks,
                  DigitExtractorT                                        digit_extractor)
    {
        compute::ArrayVar<int, BINS_PER_THREAD> exclusive_digit_prefix;
        RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }

  private:
    SmemTypePtr<uint> m_warp_offsets = nullptr;
};
}  // namespace luisa::parallel_primitive
