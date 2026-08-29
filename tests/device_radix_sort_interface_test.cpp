// Interface and edge-case tests for DeviceRadixSort.
// Covers public bit ranges, stability, zero-width ranges, and empty input.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <boost/ut.hpp>
#include <numeric>
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

    DeviceRadixSort<> radix_sort;
    radix_sort.create(device, &stream);

    "device_radix_sort_exposes_stable_partial_bit_ranges"_test = [&]
    {
        constexpr uint      item_count = 257u;
        constexpr uint      begin_bit  = 1u;
        constexpr uint      end_bit    = 6u;
        luisa::vector<uint> keys(item_count);
        luisa::vector<uint> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            keys[i]   = ((i * 53u) % 64u) | (((i * 17u) % 11u) << 8u);
            values[i] = i;
        }
        auto selected_bits = [](uint key)
        { return (key >> begin_bit) & ((1u << (end_bit - begin_bit)) - 1u); };
        luisa::vector<uint> ascending_order(item_count);
        luisa::vector<uint> descending_order(item_count);
        std::iota(ascending_order.begin(), ascending_order.end(), 0u);
        std::iota(descending_order.begin(), descending_order.end(), 0u);
        std::stable_sort(ascending_order.begin(),
                         ascending_order.end(),
                         [&](uint lhs, uint rhs)
                         { return selected_bits(keys[lhs]) < selected_bits(keys[rhs]); });
        std::stable_sort(descending_order.begin(),
                         descending_order.end(),
                         [&](uint lhs, uint rhs)
                         { return selected_bits(keys[lhs]) > selected_bits(keys[rhs]); });

        auto keys_input             = device.create_buffer<uint>(item_count);
        auto values_input           = device.create_buffer<uint>(item_count);
        auto ascending_keys         = device.create_buffer<uint>(item_count);
        auto descending_keys        = device.create_buffer<uint>(item_count);
        auto ascending_pair_keys    = device.create_buffer<uint>(item_count);
        auto ascending_pair_values  = device.create_buffer<uint>(item_count);
        auto descending_pair_keys   = device.create_buffer<uint>(item_count);
        auto descending_pair_values = device.create_buffer<uint>(item_count);
        auto temp_bytes   = DeviceRadixSort<>::GetSortPairsTempStorageBytes<uint, uint>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        radix_sort.SortKeys(
            commands, temp_storage.view(), keys_input.view(), ascending_keys.view(), item_count, begin_bit, end_bit);
        radix_sort.SortKeysDescending(
            commands, temp_storage.view(), keys_input.view(), descending_keys.view(), item_count, begin_bit, end_bit);
        radix_sort.SortPairs(commands,
                             temp_storage.view(),
                             keys_input.view(),
                             ascending_pair_keys.view(),
                             values_input.view(),
                             ascending_pair_values.view(),
                             item_count,
                             begin_bit,
                             end_bit);
        radix_sort.SortPairsDescending(commands,
                                       temp_storage.view(),
                                       keys_input.view(),
                                       descending_pair_keys.view(),
                                       values_input.view(),
                                       descending_pair_values.view(),
                                       item_count,
                                       begin_bit,
                                       end_bit);
        stream << keys_input.copy_from(luisa::span{keys})
               << values_input.copy_from(luisa::span{values}) << commands.commit();

        luisa::vector<uint> ascending_keys_result(item_count);
        luisa::vector<uint> descending_keys_result(item_count);
        luisa::vector<uint> ascending_pair_keys_result(item_count);
        luisa::vector<uint> ascending_pair_values_result(item_count);
        luisa::vector<uint> descending_pair_keys_result(item_count);
        luisa::vector<uint> descending_pair_values_result(item_count);
        stream << ascending_keys.copy_to(luisa::span{ascending_keys_result})
               << descending_keys.copy_to(luisa::span{descending_keys_result})
               << ascending_pair_keys.copy_to(luisa::span{ascending_pair_keys_result})
               << ascending_pair_values.copy_to(luisa::span{ascending_pair_values_result})
               << descending_pair_keys.copy_to(luisa::span{descending_pair_keys_result})
               << descending_pair_values.copy_to(luisa::span{descending_pair_values_result})
               << synchronize();

        auto matches_order = [&](const auto& result, const auto& source, const auto& order)
        {
            return std::equal(result.begin(),
                              result.end(),
                              order.begin(),
                              [&](const auto& value, uint index) { return value == source[index]; });
        };
        expect(matches_order(ascending_keys_result, keys, ascending_order));
        expect(matches_order(descending_keys_result, keys, descending_order));
        expect(matches_order(ascending_pair_keys_result, keys, ascending_order));
        expect(matches_order(ascending_pair_values_result, values, ascending_order));
        expect(matches_order(descending_pair_keys_result, keys, descending_order));
        expect(matches_order(descending_pair_values_result, values, descending_order));
    };

    "device_radix_sort_copies_zero_width_ranges"_test = [&]
    {
        constexpr uint      item_count = 17u;
        luisa::vector<int>  keys(item_count);
        luisa::vector<uint> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            keys[i]   = static_cast<int>(i * 7u) - 31;
            values[i] = item_count - i;
        }
        auto keys_input    = device.create_buffer<int>(item_count);
        auto values_input  = device.create_buffer<uint>(item_count);
        auto keys_output   = device.create_buffer<int>(item_count);
        auto values_output = device.create_buffer<uint>(item_count);

        CommandList commands;
        radix_sort.SortPairs(commands,
                             BufferView<uint>{},
                             keys_input.view(),
                             keys_output.view(),
                             values_input.view(),
                             values_output.view(),
                             item_count,
                             9u,
                             9u);
        stream << keys_input.copy_from(luisa::span{keys})
               << values_input.copy_from(luisa::span{values}) << commands.commit();

        luisa::vector<int>  keys_result(item_count);
        luisa::vector<uint> values_result(item_count);
        stream << keys_output.copy_to(luisa::span{keys_result})
               << values_output.copy_to(luisa::span{values_result}) << synchronize();
        expect(std::equal(keys_result.begin(), keys_result.end(), keys.begin()));
        expect(std::equal(values_result.begin(), values_result.end(), values.begin()));
    };

    "device_radix_sort_handles_empty_input"_test = [&]
    {
        CommandList commands;
        radix_sort.SortKeys(commands, BufferView<uint>{}, BufferView<int>{}, BufferView<int>{}, 0u);
        stream << commands.commit() << synchronize();
        expect(DeviceRadixSort<>::GetSortKeysTempStorageBytes<int>(0u) == 0u);
    };
}
