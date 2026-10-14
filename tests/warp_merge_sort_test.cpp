// Tests for WarpMergeSort.
// Covers stable pairs, custom comparators, descending order, and guarded tiles.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/dsl/builtin.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <utility>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint logical_warp_threads  = 8u;
constexpr uint items_per_thread      = 3u;
constexpr uint physical_warp_threads = 32u;
constexpr uint logical_warp_items    = logical_warp_threads * items_per_thread;
constexpr uint item_count            = physical_warp_threads * items_per_thread;
constexpr uint guarded_valid_items   = 17u;
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

    "warp_merge_sort_supports_stable_pairs_comparators_and_guarded_tiles"_test = [&]
    {
        auto sorted_keys   = device.create_buffer<int>(item_count);
        auto sorted_values = device.create_buffer<int>(item_count);
        auto descending    = device.create_buffer<int>(item_count);
        auto guarded       = device.create_buffer<int>(item_count);

        luisa::vector<int> host_sorted_keys(item_count);
        luisa::vector<int> host_sorted_values(item_count);
        luisa::vector<int> host_descending(item_count);
        luisa::vector<int> host_guarded(item_count);

        Kernel1D kernel = [](BufferVar<int> sorted_keys_buffer,
                             BufferVar<int> sorted_values_buffer,
                             BufferVar<int> descending_buffer,
                             BufferVar<int> guarded_buffer) noexcept
        {
            set_block_size(physical_warp_threads);
            UInt thread_index      = dispatch_id().x;
            UInt lane              = warp_lane_id() % logical_warp_threads;
            UInt logical_warp_base = thread_index / logical_warp_threads * logical_warp_items;

            ArrayVar<int, items_per_thread> keys;
            ArrayVar<int, items_per_thread> values;
            ArrayVar<int, items_per_thread> descending_keys;
            ArrayVar<int, items_per_thread> guarded_keys;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt linear_item      = lane * items_per_thread + item;
                keys[item]            = cast<int>((linear_item * 7u + 3u) % 5u);
                values[item]          = cast<int>(logical_warp_base + linear_item);
                descending_keys[item] = keys[item];
                guarded_keys[item]    = cast<int>((linear_item * 11u + 2u) % 23u);
            }

            auto less    = [](const Int& lhs, const Int& rhs) noexcept { return lhs < rhs; };
            auto greater = [](const Int& lhs, const Int& rhs) noexcept { return lhs > rhs; };

            WarpMergeSort<int, items_per_thread, logical_warp_threads, int>().StableSort(keys, values, less);
            WarpMergeSort<int, items_per_thread, logical_warp_threads>().Sort(descending_keys, greater);
            WarpMergeSort<int, items_per_thread, logical_warp_threads>().Sort(
                guarded_keys, less, guarded_valid_items, 1000);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt output_index = thread_index * items_per_thread + item;
                sorted_keys_buffer.write(output_index, keys[item]);
                sorted_values_buffer.write(output_index, values[item]);
                descending_buffer.write(output_index, descending_keys[item]);
                guarded_buffer.write(output_index, guarded_keys[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(sorted_keys, sorted_values, descending, guarded).dispatch(physical_warp_threads)
               << sorted_keys.copy_to(luisa::span{host_sorted_keys})
               << sorted_values.copy_to(luisa::span{host_sorted_values})
               << descending.copy_to(luisa::span{host_descending})
               << guarded.copy_to(luisa::span{host_guarded}) << synchronize();

        for(uint logical_warp = 0u; logical_warp < physical_warp_threads / logical_warp_threads; ++logical_warp)
        {
            std::vector<std::pair<int, int>> expected_pairs;
            std::vector<int>                 expected_descending;
            std::vector<int>                 expected_guarded;
            expected_pairs.reserve(logical_warp_items);
            expected_descending.reserve(logical_warp_items);
            expected_guarded.reserve(logical_warp_items);

            uint logical_warp_base = logical_warp * logical_warp_items;
            for(uint item = 0u; item < logical_warp_items; ++item)
            {
                int key = static_cast<int>((item * 7u + 3u) % 5u);
                expected_pairs.emplace_back(key, static_cast<int>(logical_warp_base + item));
                expected_descending.emplace_back(key);
                expected_guarded.emplace_back(
                    item < guarded_valid_items ? static_cast<int>((item * 11u + 2u) % 23u) : 1000);
            }

            std::stable_sort(expected_pairs.begin(),
                             expected_pairs.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            std::sort(expected_descending.begin(), expected_descending.end(), std::greater<>());
            std::sort(expected_guarded.begin(), expected_guarded.end());

            for(uint item = 0u; item < logical_warp_items; ++item)
            {
                uint output_index = logical_warp_base + item;
                expect(host_sorted_keys[output_index] == expected_pairs[item].first);
                expect(host_sorted_values[output_index] == expected_pairs[item].second);
                expect(host_descending[output_index] == expected_descending[item]);
                expect(host_guarded[output_index] == expected_guarded[item]);
            }
        }
    };
}
