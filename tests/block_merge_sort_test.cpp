// Test for BlockMergeSort.
// Covers stable key-value ordering, descending comparison, partial tiles, and
// blocked-to-striped output layout.

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
constexpr uint valid_items      = 47u;
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

    "block_merge_sort_handles_stable_pairs_descending_partial_and_striped_layouts"_test = [&]
    {
        auto stable_keys    = device.create_buffer<uint>(block_items);
        auto stable_values  = device.create_buffer<uint>(block_items);
        auto descending     = device.create_buffer<uint>(block_items);
        auto partial_keys   = device.create_buffer<uint>(block_items);
        auto partial_values = device.create_buffer<uint>(block_items);
        auto striped_keys   = device.create_buffer<uint>(block_items);
        auto striped_values = device.create_buffer<uint>(block_items);

        luisa::vector<uint> host_stable_keys(block_items);
        luisa::vector<uint> host_stable_values(block_items);
        luisa::vector<uint> host_descending(block_items);
        luisa::vector<uint> host_partial_keys(block_items);
        luisa::vector<uint> host_partial_values(block_items);
        luisa::vector<uint> host_striped_keys(block_items);
        luisa::vector<uint> host_striped_values(block_items);

        Kernel1D kernel = [](BufferVar<uint> stable_keys_buffer,
                             BufferVar<uint> stable_values_buffer,
                             BufferVar<uint> descending_buffer,
                             BufferVar<uint> partial_keys_buffer,
                             BufferVar<uint> partial_values_buffer,
                             BufferVar<uint> striped_keys_buffer,
                             BufferVar<uint> striped_values_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt                             tid = thread_id().x;
            ArrayVar<uint, items_per_thread> keys;
            ArrayVar<uint, items_per_thread> values;
            ArrayVar<uint, items_per_thread> descending_keys;
            ArrayVar<uint, items_per_thread> guarded_keys;
            ArrayVar<uint, items_per_thread> guarded_values;
            ArrayVar<uint, items_per_thread> striped_sort_keys;
            ArrayVar<uint, items_per_thread> striped_sort_values;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index                = tid * items_per_thread + item;
                UInt key                  = (index * 13u + 5u) % 9u;
                keys[item]                = key;
                values[item]              = index;
                descending_keys[item]     = key;
                guarded_keys[item]        = key;
                guarded_values[item]      = index;
                striped_sort_keys[item]   = key;
                striped_sort_values[item] = index;
            }

            auto less    = [](const UInt& lhs, const UInt& rhs) noexcept { return lhs < rhs; };
            auto greater = [](const UInt& lhs, const UInt& rhs) noexcept { return lhs > rhs; };

            BlockMergeSort<uint, block_threads, items_per_thread, uint> pair_sort;
            pair_sort.StableSort(keys, values, less);

            BlockMergeSort<uint, block_threads, items_per_thread> key_sort;
            key_sort.Sort(descending_keys, greater);

            BlockMergeSort<uint, block_threads, items_per_thread, uint> partial_sort;
            partial_sort.StableSort(guarded_keys, guarded_values, less, valid_items, 999u);

            BlockMergeSort<uint, block_threads, items_per_thread, uint> striped_sort;
            striped_sort.SortBlockedToStriped(striped_sort_keys, striped_sort_values, less);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                stable_keys_buffer.write(index, keys[item]);
                stable_values_buffer.write(index, values[item]);
                descending_buffer.write(index, descending_keys[item]);
                partial_keys_buffer.write(index, guarded_keys[item]);
                partial_values_buffer.write(index, guarded_values[item]);
                striped_keys_buffer.write(index, striped_sort_keys[item]);
                striped_values_buffer.write(index, striped_sort_values[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(stable_keys, stable_values, descending, partial_keys, partial_values, striped_keys, striped_values)
                      .dispatch(block_threads)
               << stable_keys.copy_to(luisa::span{host_stable_keys})
               << stable_values.copy_to(luisa::span{host_stable_values})
               << descending.copy_to(luisa::span{host_descending})
               << partial_keys.copy_to(luisa::span{host_partial_keys})
               << partial_values.copy_to(luisa::span{host_partial_values})
               << striped_keys.copy_to(luisa::span{host_striped_keys})
               << striped_values.copy_to(luisa::span{host_striped_values}) << synchronize();

        luisa::vector<uint> order(block_items);
        std::iota(order.begin(), order.end(), 0u);
        auto key_at = [](uint index) { return (index * 13u + 5u) % 9u; };
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](uint lhs, uint rhs) { return key_at(lhs) < key_at(rhs); });
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(host_stable_keys[index] == key_at(order[index]));
            expect(host_stable_values[index] == order[index]);
            if(index > 0u)
            {
                expect(host_descending[index - 1u] >= host_descending[index]);
            }
            uint thread        = index / items_per_thread;
            uint item          = index % items_per_thread;
            uint striped_index = thread + item * block_threads;
            expect(host_striped_keys[index] == key_at(order[striped_index]));
            expect(host_striped_values[index] == order[striped_index]);
        }

        luisa::vector<uint> partial_order(valid_items);
        std::iota(partial_order.begin(), partial_order.end(), 0u);
        std::stable_sort(partial_order.begin(),
                         partial_order.end(),
                         [&](uint lhs, uint rhs) { return key_at(lhs) < key_at(rhs); });
        for(uint index = 0u; index < block_items; ++index)
        {
            if(index < valid_items)
            {
                expect(host_partial_keys[index] == key_at(partial_order[index]));
                expect(host_partial_values[index] == partial_order[index]);
            }
            else
            {
                expect(host_partial_keys[index] == 999u);
            }
        }
    };
}
