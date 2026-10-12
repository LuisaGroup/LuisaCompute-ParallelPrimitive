// Test for BlockRadixRank.
// Covers stable ascending and descending ranks plus exclusive digit prefixes.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint block_threads   = 32u;
constexpr uint keys_per_thread = 2u;
constexpr uint block_items     = block_threads * keys_per_thread;
constexpr uint radix_bits      = 3u;
constexpr uint radix_digits    = 1u << radix_bits;
}  // namespace

int main(int argc, char* argv[])
{
    luisa::string_view context_directory;
    luisa::string_view backend_name = "cuda";
    for(int i = 1; i < argc; ++i)
    {
        luisa::string_view argument(argv[i]);
        if(argument.starts_with("--ctx="))
        {
            context_directory = argument.substr(6);
        }
        else if(argument.starts_with("--backend="))
        {
            backend_name = argument.substr(10);
        }
    }

    Context context{context_directory.empty() ? argv[0] : context_directory};
    Device  device = context.create_device(backend_name);
    Stream  stream = device.create_stream();

    "block_radix_rank_produces_stable_ranks_and_digit_prefixes"_test = [&]
    {
        auto ascending_rank_buffer    = device.create_buffer<uint>(block_items);
        auto descending_rank_buffer   = device.create_buffer<uint>(block_items);
        auto ascending_prefix_buffer  = device.create_buffer<uint>(radix_digits);
        auto descending_prefix_buffer = device.create_buffer<uint>(radix_digits);

        Kernel1D kernel = [](BufferVar<uint> ascending_ranks,
                             BufferVar<uint> descending_ranks,
                             BufferVar<uint> ascending_prefixes,
                             BufferVar<uint> descending_prefixes) noexcept
        {
            set_block_size(block_threads);
            UInt                            tid = thread_id().x;
            ArrayVar<uint, keys_per_thread> keys;
            ArrayVar<uint, keys_per_thread> ascending;
            ArrayVar<uint, keys_per_thread> descending;
            ArrayVar<uint, BlockRadixRank<block_threads, radix_bits>::BINS_TRACKED_PER_THREAD> ascending_prefix;
            ArrayVar<uint, BlockRadixRank<block_threads, radix_bits, true>::BINS_TRACKED_PER_THREAD> descending_prefix;
            for(uint item = 0u; item < keys_per_thread; ++item)
            {
                UInt index = tid * keys_per_thread + item;
                keys[item] = (index * 5u + 3u) & (radix_digits - 1u);
            }

            ShiftDigitExtractor<uint>                 digit_extractor{0u, radix_bits};
            BlockRadixRank<block_threads, radix_bits> ascending_rank;
            ascending_rank.template RankKeys<uint, keys_per_thread>(keys, ascending, digit_extractor, ascending_prefix);

            BlockRadixRank<block_threads, radix_bits, true> descending_rank;
            descending_rank.template RankKeys<uint, keys_per_thread>(keys, descending, digit_extractor, descending_prefix);

            for(uint item = 0u; item < keys_per_thread; ++item)
            {
                UInt index = tid * keys_per_thread + item;
                ascending_ranks.write(index, ascending[item]);
                descending_ranks.write(index, descending[item]);
            }
            $if(tid < radix_digits)
            {
                ascending_prefixes.write(tid, ascending_prefix[0u]);
                descending_prefixes.write(tid, descending_prefix[0u]);
            };
        };

        auto                shader = device.compile(kernel);
        luisa::vector<uint> ascending_ranks(block_items);
        luisa::vector<uint> descending_ranks(block_items);
        luisa::vector<uint> ascending_prefixes(radix_digits);
        luisa::vector<uint> descending_prefixes(radix_digits);
        stream << shader(ascending_rank_buffer, descending_rank_buffer, ascending_prefix_buffer, descending_prefix_buffer)
                      .dispatch(block_threads)
               << ascending_rank_buffer.copy_to(luisa::span{ascending_ranks})
               << descending_rank_buffer.copy_to(luisa::span{descending_ranks})
               << ascending_prefix_buffer.copy_to(luisa::span{ascending_prefixes})
               << descending_prefix_buffer.copy_to(luisa::span{descending_prefixes}) << synchronize();

        auto digit_at = [](uint index) { return (index * 5u + 3u) & (radix_digits - 1u); };
        luisa::vector<uint> ascending_order(block_items);
        std::iota(ascending_order.begin(), ascending_order.end(), 0u);
        std::stable_sort(ascending_order.begin(),
                         ascending_order.end(),
                         [&](uint lhs, uint rhs) { return digit_at(lhs) < digit_at(rhs); });
        luisa::vector<uint> descending_order(block_items);
        std::iota(descending_order.begin(), descending_order.end(), 0u);
        std::stable_sort(descending_order.begin(),
                         descending_order.end(),
                         [&](uint lhs, uint rhs) { return digit_at(lhs) > digit_at(rhs); });

        luisa::vector<uint> expected_ascending_ranks(block_items);
        luisa::vector<uint> expected_descending_ranks(block_items);
        for(uint rank = 0u; rank < block_items; ++rank)
        {
            expected_ascending_ranks[ascending_order[rank]]   = rank;
            expected_descending_ranks[descending_order[rank]] = rank;
        }
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(ascending_ranks[index] == expected_ascending_ranks[index]);
            expect(descending_ranks[index] == expected_descending_ranks[index]);
        }

        uint ascending_running  = 0u;
        uint descending_running = 0u;
        for(uint ordered_bin = 0u; ordered_bin < radix_digits; ++ordered_bin)
        {
            expect(ascending_prefixes[ordered_bin] == ascending_running);
            expect(descending_prefixes[ordered_bin] == descending_running);
            ascending_running += static_cast<uint>(
                std::count_if(ascending_order.begin(),
                              ascending_order.end(),
                              [&](uint index) { return digit_at(index) == ordered_bin; }));
            uint descending_digit = radix_digits - 1u - ordered_bin;
            descending_running += static_cast<uint>(
                std::count_if(descending_order.begin(),
                              descending_order.end(),
                              [&](uint index) { return digit_at(index) == descending_digit; }));
        }
    };
}
