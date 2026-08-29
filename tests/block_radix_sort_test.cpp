// Tests for BlockRadixSort.
// Covers signed pairs, ordering direction, striped output, and bit ranges.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/dsl/builtin.h>
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
constexpr uint block_threads    = 32u;
constexpr uint items_per_thread = 2u;
constexpr uint block_items      = block_threads * items_per_thread;
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

    "block_radix_sort_handles_signed_pairs_descending_striped_and_bit_ranges"_test = [&]
    {
        auto ascending_keys   = device.create_buffer<int>(block_items);
        auto ascending_values = device.create_buffer<uint>(block_items);
        auto descending_keys  = device.create_buffer<int>(block_items);
        auto striped_keys     = device.create_buffer<int>(block_items);
        auto bit_range_keys   = device.create_buffer<uint>(block_items);
        auto empty_range_keys = device.create_buffer<uint>(block_items);

        luisa::vector<int>  host_ascending_keys(block_items);
        luisa::vector<uint> host_ascending_values(block_items);
        luisa::vector<int>  host_descending_keys(block_items);
        luisa::vector<int>  host_striped_keys(block_items);
        luisa::vector<uint> host_bit_range_keys(block_items);
        luisa::vector<uint> host_empty_range_keys(block_items);

        Kernel1D kernel = [](BufferVar<int>  ascending_keys_buffer,
                             BufferVar<uint> ascending_values_buffer,
                             BufferVar<int>  descending_keys_buffer,
                             BufferVar<int>  striped_keys_buffer,
                             BufferVar<uint> bit_range_keys_buffer,
                             BufferVar<uint> empty_range_keys_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt                             tid = thread_id().x;
            ArrayVar<int, items_per_thread>  keys;
            ArrayVar<uint, items_per_thread> values;
            ArrayVar<int, items_per_thread>  descending;
            ArrayVar<int, items_per_thread>  striped;
            ArrayVar<uint, items_per_thread> bit_range;
            ArrayVar<uint, items_per_thread> empty_range;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index        = tid * items_per_thread + item;
                Int  key          = cast<int>((index * 17u + 3u) % 13u) - 6;
                keys[item]        = key;
                values[item]      = index;
                descending[item]  = key;
                striped[item]     = key;
                bit_range[item]   = (index * 37u + 9u) & 255u;
                empty_range[item] = (index * 29u + 7u) & 255u;
            }

            BlockRadixSort<int, block_threads, items_per_thread, uint> pair_sort;
            pair_sort.Sort(keys, values);

            BlockRadixSort<int, block_threads, items_per_thread> descending_sort;
            descending_sort.SortDescending(descending);

            BlockRadixSort<int, block_threads, items_per_thread> striped_sort;
            striped_sort.SortBlockedToStriped(striped);

            BlockRadixSort<uint, block_threads, items_per_thread> bit_sort;
            bit_sort.Sort(bit_range, 2u, 6u);

            BlockRadixSort<uint, block_threads, items_per_thread> empty_range_sort;
            empty_range_sort.Sort(empty_range, 5u, 5u);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                ascending_keys_buffer.write(index, keys[item]);
                ascending_values_buffer.write(index, values[item]);
                descending_keys_buffer.write(index, descending[item]);
                striped_keys_buffer.write(index, striped[item]);
                bit_range_keys_buffer.write(index, bit_range[item]);
                empty_range_keys_buffer.write(index, empty_range[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(ascending_keys, ascending_values, descending_keys, striped_keys, bit_range_keys, empty_range_keys)
                      .dispatch(block_threads)
               << ascending_keys.copy_to(luisa::span{host_ascending_keys})
               << ascending_values.copy_to(luisa::span{host_ascending_values})
               << descending_keys.copy_to(luisa::span{host_descending_keys})
               << striped_keys.copy_to(luisa::span{host_striped_keys})
               << bit_range_keys.copy_to(luisa::span{host_bit_range_keys})
               << empty_range_keys.copy_to(luisa::span{host_empty_range_keys}) << synchronize();

        auto signed_key_at = [](uint index)
        { return static_cast<int>((index * 17u + 3u) % 13u) - 6; };
        luisa::vector<uint> order(block_items);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](uint lhs, uint rhs) { return signed_key_at(lhs) < signed_key_at(rhs); });
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(host_ascending_keys[index] == signed_key_at(order[index]));
            expect(host_ascending_values[index] == order[index]);
            expect(host_descending_keys[index] == signed_key_at(order[block_items - 1u - index]));

            uint lane         = index / items_per_thread;
            uint item         = index % items_per_thread;
            uint sorted_index = lane + item * block_threads;
            expect(host_striped_keys[index] == signed_key_at(order[sorted_index]));
        }

        luisa::vector<uint> bit_order(block_items);
        std::iota(bit_order.begin(), bit_order.end(), 0u);
        auto bit_key_at = [](uint index) { return (index * 37u + 9u) & 255u; };
        std::stable_sort(bit_order.begin(),
                         bit_order.end(),
                         [&](uint lhs, uint rhs) {
                             return ((bit_key_at(lhs) >> 2u) & 15u) < ((bit_key_at(rhs) >> 2u) & 15u);
                         });
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(host_bit_range_keys[index] == bit_key_at(bit_order[index]));
            expect(host_empty_range_keys[index] == ((index * 29u + 7u) & 255u));
        }
    };
}
