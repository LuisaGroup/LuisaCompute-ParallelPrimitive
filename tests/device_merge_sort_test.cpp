// Tests for DeviceMergeSort.
// Covers key/pair entry points, custom ordering, stability, and partial passes.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <functional>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

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

    DeviceMergeSort<> merge_sort;
    merge_sort.create(device);

    "device_merge_sort_orders_keys_and_stable_pairs"_test = [&]
    {
        constexpr uint      item_count = 513u;
        luisa::vector<int>  keys(item_count);
        luisa::vector<uint> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            keys[i]   = static_cast<int>((i * 37u) % 23u) - 11;
            values[i] = i;
        }

        auto expected_ascending_keys  = keys;
        auto expected_descending_keys = keys;
        std::stable_sort(expected_ascending_keys.begin(), expected_ascending_keys.end());
        std::stable_sort(expected_descending_keys.begin(), expected_descending_keys.end(), std::greater<>{});

        luisa::vector<uint> ascending_order(item_count);
        luisa::vector<uint> descending_order(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            ascending_order[i]  = i;
            descending_order[i] = i;
        }
        std::stable_sort(ascending_order.begin(),
                         ascending_order.end(),
                         [&](uint lhs, uint rhs) { return keys[lhs] < keys[rhs]; });
        std::stable_sort(descending_order.begin(),
                         descending_order.end(),
                         [&](uint lhs, uint rhs) { return keys[lhs] > keys[rhs]; });

        auto keys_input         = device.create_buffer<int>(item_count);
        auto values_input       = device.create_buffer<uint>(item_count);
        auto sorted_keys        = device.create_buffer<int>(item_count);
        auto stable_sorted_keys = device.create_buffer<int>(item_count);
        auto sorted_pair_keys   = device.create_buffer<int>(item_count);
        auto sorted_pair_values = device.create_buffer<uint>(item_count);
        auto stable_pair_keys   = device.create_buffer<int>(item_count);
        auto stable_pair_values = device.create_buffer<uint>(item_count);
        auto temp_bytes   = DeviceMergeSort<>::GetSortPairsTempStorageBytes<int, uint>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        Callable<bool(int, int)> greater = [](Int lhs, Int rhs) noexcept { return lhs > rhs; };

        CommandList commands;
        merge_sort.SortKeys(commands, temp_storage.view(), keys_input.view(), sorted_keys.view(), item_count);
        merge_sort.StableSortKeys(
            commands, temp_storage.view(), keys_input.view(), stable_sorted_keys.view(), item_count, greater);
        merge_sort.SortPairs(commands,
                             temp_storage.view(),
                             keys_input.view(),
                             sorted_pair_keys.view(),
                             values_input.view(),
                             sorted_pair_values.view(),
                             item_count);
        merge_sort.StableSortPairs(commands,
                                   temp_storage.view(),
                                   keys_input.view(),
                                   stable_pair_keys.view(),
                                   values_input.view(),
                                   stable_pair_values.view(),
                                   item_count,
                                   greater);
        stream << keys_input.copy_from(luisa::span{keys})
               << values_input.copy_from(luisa::span{values}) << commands.commit();

        luisa::vector<int>  sorted_keys_result(item_count);
        luisa::vector<int>  stable_sorted_keys_result(item_count);
        luisa::vector<int>  sorted_pair_keys_result(item_count);
        luisa::vector<uint> sorted_pair_values_result(item_count);
        luisa::vector<int>  stable_pair_keys_result(item_count);
        luisa::vector<uint> stable_pair_values_result(item_count);
        stream << sorted_keys.copy_to(luisa::span{sorted_keys_result})
               << stable_sorted_keys.copy_to(luisa::span{stable_sorted_keys_result})
               << sorted_pair_keys.copy_to(luisa::span{sorted_pair_keys_result})
               << sorted_pair_values.copy_to(luisa::span{sorted_pair_values_result})
               << stable_pair_keys.copy_to(luisa::span{stable_pair_keys_result})
               << stable_pair_values.copy_to(luisa::span{stable_pair_values_result}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(sorted_keys_result[i] == expected_ascending_keys[i]);
            expect(stable_sorted_keys_result[i] == expected_descending_keys[i]);
            expect(sorted_pair_keys_result[i] == keys[ascending_order[i]]);
            expect(sorted_pair_values_result[i] == values[ascending_order[i]]);
            expect(stable_pair_keys_result[i] == keys[descending_order[i]]);
            expect(stable_pair_values_result[i] == values[descending_order[i]]);
        }
    };

    "device_merge_sort_handles_empty_and_singleton_inputs"_test = [&]
    {
        CommandList empty_commands;
        merge_sort.SortKeys(empty_commands, BufferView<uint>{}, BufferView<int>{}, BufferView<int>{}, 0u);
        stream << empty_commands.commit() << synchronize();

        int         value  = 42;
        auto        input  = device.create_buffer<int>(1u);
        auto        output = device.create_buffer<int>(1u);
        CommandList singleton_commands;
        merge_sort.StableSortKeys(singleton_commands, BufferView<uint>{}, input.view(), output.view(), 1u);
        stream << input.copy_from(luisa::span{&value, 1u}) << singleton_commands.commit()
               << output.copy_to(luisa::span{&value, 1u}) << synchronize();
        expect(value == 42);
        expect(DeviceMergeSort<>::GetSortKeysTempStorageBytes<int>(1u) == 0u);
    };

    "device_merge_sort_copy_entry_points_preserve_inputs"_test = [&]
    {
        constexpr uint               item_count = 9u;
        std::array<int, item_count>  keys{4, 1, 3, 1, 2, 4, 0, 3, 2};
        std::array<uint, item_count> values{0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto                         expected_keys = keys;
        std::stable_sort(expected_keys.begin(), expected_keys.end(), std::greater<>{});
        auto expected_order = values;
        std::stable_sort(expected_order.begin(),
                         expected_order.end(),
                         [&](uint lhs, uint rhs) { return keys[lhs] < keys[rhs]; });

        auto keys_input       = device.create_buffer<int>(item_count);
        auto values_input     = device.create_buffer<uint>(item_count);
        auto sorted_keys      = device.create_buffer<int>(item_count);
        auto stable_keys      = device.create_buffer<int>(item_count);
        auto sorted_pair_keys = device.create_buffer<int>(item_count);
        auto sorted_values    = device.create_buffer<uint>(item_count);
        auto temp_bytes = DeviceMergeSort<>::GetSortPairsTempStorageBytes<int, uint>(item_count);
        auto temp       = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
        Callable<bool(int, int)> greater = [](Int lhs, Int rhs) noexcept { return lhs > rhs; };

        CommandList commands;
        merge_sort.SortKeysCopy(commands, temp.view(), keys_input.view(), sorted_keys.view(), item_count, greater);
        merge_sort.StableSortKeysCopy(commands, temp.view(), keys_input.view(), stable_keys.view(), item_count, greater);
        merge_sort.SortPairsCopy(commands,
                                 temp.view(),
                                 keys_input.view(),
                                 sorted_pair_keys.view(),
                                 values_input.view(),
                                 sorted_values.view(),
                                 item_count);
        stream << keys_input.copy_from(luisa::span{keys})
               << values_input.copy_from(luisa::span{values}) << commands.commit();

        std::array<int, item_count>  keys_after{};
        std::array<uint, item_count> values_after{};
        std::array<int, item_count>  sorted_keys_result{};
        std::array<int, item_count>  stable_keys_result{};
        std::array<int, item_count>  pair_keys_result{};
        std::array<uint, item_count> pair_values_result{};
        stream << keys_input.copy_to(luisa::span{keys_after})
               << values_input.copy_to(luisa::span{values_after})
               << sorted_keys.copy_to(luisa::span{sorted_keys_result})
               << stable_keys.copy_to(luisa::span{stable_keys_result})
               << sorted_pair_keys.copy_to(luisa::span{pair_keys_result})
               << sorted_values.copy_to(luisa::span{pair_values_result}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(keys_after[i] == keys[i]);
            expect(values_after[i] == values[i]);
            expect(sorted_keys_result[i] == expected_keys[i]);
            expect(stable_keys_result[i] == expected_keys[i]);
            expect(pair_keys_result[i] == keys[expected_order[i]]);
            expect(pair_values_result[i] == values[expected_order[i]]);
        }
    };

    "device_merge_sort_supports_in_place_entry_points"_test = [&]
    {
        constexpr uint               item_count = 7u;
        std::array<int, item_count>  keys{3, 1, 2, 1, 3, 0, 2};
        std::array<uint, item_count> values{0, 1, 2, 3, 4, 5, 6};
        auto                         expected_order = values;
        std::stable_sort(expected_order.begin(),
                         expected_order.end(),
                         [&](uint lhs, uint rhs) { return keys[lhs] < keys[rhs]; });

        auto unstable_keys      = device.create_buffer<int>(item_count);
        auto stable_keys        = device.create_buffer<int>(item_count);
        auto pair_keys          = device.create_buffer<int>(item_count);
        auto pair_values        = device.create_buffer<uint>(item_count);
        auto stable_pair_keys   = device.create_buffer<int>(item_count);
        auto stable_pair_values = device.create_buffer<uint>(item_count);
        auto temp_bytes = DeviceMergeSort<>::GetSortPairsTempStorageBytes<int, uint>(item_count);
        auto temp       = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        merge_sort.SortKeys(commands, temp.view(), unstable_keys.view(), item_count);
        merge_sort.StableSortKeys(commands, temp.view(), stable_keys.view(), item_count);
        merge_sort.SortPairs(commands, temp.view(), pair_keys.view(), pair_values.view(), item_count);
        merge_sort.StableSortPairs(
            commands, temp.view(), stable_pair_keys.view(), stable_pair_values.view(), item_count);
        stream << unstable_keys.copy_from(luisa::span{keys}) << stable_keys.copy_from(luisa::span{keys})
               << pair_keys.copy_from(luisa::span{keys}) << pair_values.copy_from(luisa::span{values})
               << stable_pair_keys.copy_from(luisa::span{keys})
               << stable_pair_values.copy_from(luisa::span{values}) << commands.commit();

        std::array<int, item_count>  unstable_keys_result{};
        std::array<int, item_count>  stable_keys_result{};
        std::array<int, item_count>  pair_keys_result{};
        std::array<uint, item_count> pair_values_result{};
        std::array<int, item_count>  stable_pair_keys_result{};
        std::array<uint, item_count> stable_pair_values_result{};
        stream << unstable_keys.copy_to(luisa::span{unstable_keys_result})
               << stable_keys.copy_to(luisa::span{stable_keys_result})
               << pair_keys.copy_to(luisa::span{pair_keys_result})
               << pair_values.copy_to(luisa::span{pair_values_result})
               << stable_pair_keys.copy_to(luisa::span{stable_pair_keys_result})
               << stable_pair_values.copy_to(luisa::span{stable_pair_values_result}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(unstable_keys_result[i] == keys[expected_order[i]]);
            expect(stable_keys_result[i] == keys[expected_order[i]]);
            expect(pair_keys_result[i] == keys[expected_order[i]]);
            expect(pair_values_result[i] == values[expected_order[i]]);
            expect(stable_pair_keys_result[i] == keys[expected_order[i]]);
            expect(stable_pair_values_result[i] == values[expected_order[i]]);
        }
    };
}
