// Tests for DeviceSegmentedSort and DeviceSegmentedRadixSort.
// Covers non-contiguous and empty segments, untouched gaps, stability, and radix bit ranges.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <boost/ut.hpp>
#include <functional>
#include <limits>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint item_count     = 19u;
constexpr uint segment_count  = 4u;
constexpr int  key_sentinel   = -777;
constexpr uint value_sentinel = std::numeric_limits<uint>::max();

const luisa::vector<uint> begin_offsets{1u, 5u, 5u, 10u};
const luisa::vector<uint> end_offsets{4u, 5u, 9u, item_count};

[[nodiscard]] bool is_segment_item(uint index)
{
    for(uint segment = 0u; segment < segment_count; ++segment)
    {
        if(index >= begin_offsets[segment] && index < end_offsets[segment])
        {
            return true;
        }
    }
    return false;
}

template <typename KeyT, typename Less>
[[nodiscard]] luisa::vector<uint> segmented_order(const luisa::vector<KeyT>& keys, Less less)
{
    luisa::vector<uint> order(item_count, value_sentinel);
    for(uint segment = 0u; segment < segment_count; ++segment)
    {
        luisa::vector<uint> segment_order(end_offsets[segment] - begin_offsets[segment]);
        std::iota(segment_order.begin(), segment_order.end(), begin_offsets[segment]);
        std::stable_sort(segment_order.begin(),
                         segment_order.end(),
                         [&](uint lhs, uint rhs) { return less(keys[lhs], keys[rhs]); });
        std::copy(segment_order.begin(), segment_order.end(), order.begin() + begin_offsets[segment]);
    }
    return order;
}
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

    DeviceSegmentedSort<>      segmented_sort;
    DeviceSegmentedRadixSort<> segmented_radix_sort;
    segmented_sort.create(device);
    segmented_radix_sort.create(device);

    "device_segmented_sort_covers_all_key_and_pair_entry_points"_test = [&]
    {
        const luisa::vector<int> keys{91, 3, -2, 3, 92, 8, 1, 8, 4, 93, 5, -1, 5, 2, -1, 7, 2, 5, 0};
        luisa::vector<uint> values(item_count);
        std::iota(values.begin(), values.end(), 0u);
        auto ascending_order  = segmented_order(keys, std::less<>{});
        auto descending_order = segmented_order(keys, std::greater<>{});

        auto keys_input   = device.create_buffer<int>(item_count);
        auto values_input = device.create_buffer<uint>(item_count);
        auto begins       = device.create_buffer<uint>(segment_count);
        auto ends         = device.create_buffer<uint>(segment_count);

        auto sort_keys                     = device.create_buffer<int>(item_count);
        auto sort_keys_descending          = device.create_buffer<int>(item_count);
        auto stable_sort_keys              = device.create_buffer<int>(item_count);
        auto stable_sort_keys_descending   = device.create_buffer<int>(item_count);
        auto sort_pair_keys                = device.create_buffer<int>(item_count);
        auto sort_pair_values              = device.create_buffer<uint>(item_count);
        auto sort_pair_descending_keys     = device.create_buffer<int>(item_count);
        auto sort_pair_descending_values   = device.create_buffer<uint>(item_count);
        auto stable_pair_keys              = device.create_buffer<int>(item_count);
        auto stable_pair_values            = device.create_buffer<uint>(item_count);
        auto stable_pair_descending_keys   = device.create_buffer<int>(item_count);
        auto stable_pair_descending_values = device.create_buffer<uint>(item_count);

        luisa::vector<int>  key_initial(item_count, key_sentinel);
        luisa::vector<uint> value_initial(item_count, value_sentinel);
        auto temp_bytes = DeviceSegmentedSort<>::GetSortPairsTempStorageBytes<int, uint>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        segmented_sort.SortKeys(commands,
                                temp_storage.view(),
                                keys_input.view(),
                                sort_keys.view(),
                                item_count,
                                segment_count,
                                begins.view(),
                                ends.view());
        segmented_sort.SortKeysDescending(commands,
                                          temp_storage.view(),
                                          keys_input.view(),
                                          sort_keys_descending.view(),
                                          item_count,
                                          segment_count,
                                          begins.view(),
                                          ends.view());
        segmented_sort.StableSortKeys(commands,
                                      temp_storage.view(),
                                      keys_input.view(),
                                      stable_sort_keys.view(),
                                      item_count,
                                      segment_count,
                                      begins.view(),
                                      ends.view());
        segmented_sort.StableSortKeysDescending(commands,
                                                temp_storage.view(),
                                                keys_input.view(),
                                                stable_sort_keys_descending.view(),
                                                item_count,
                                                segment_count,
                                                begins.view(),
                                                ends.view());
        segmented_sort.SortPairs(commands,
                                 temp_storage.view(),
                                 keys_input.view(),
                                 sort_pair_keys.view(),
                                 values_input.view(),
                                 sort_pair_values.view(),
                                 item_count,
                                 segment_count,
                                 begins.view(),
                                 ends.view());
        segmented_sort.SortPairsDescending(commands,
                                           temp_storage.view(),
                                           keys_input.view(),
                                           sort_pair_descending_keys.view(),
                                           values_input.view(),
                                           sort_pair_descending_values.view(),
                                           item_count,
                                           segment_count,
                                           begins.view(),
                                           ends.view());
        segmented_sort.StableSortPairs(commands,
                                       temp_storage.view(),
                                       keys_input.view(),
                                       stable_pair_keys.view(),
                                       values_input.view(),
                                       stable_pair_values.view(),
                                       item_count,
                                       segment_count,
                                       begins.view(),
                                       ends.view());
        segmented_sort.StableSortPairsDescending(commands,
                                                 temp_storage.view(),
                                                 keys_input.view(),
                                                 stable_pair_descending_keys.view(),
                                                 values_input.view(),
                                                 stable_pair_descending_values.view(),
                                                 item_count,
                                                 segment_count,
                                                 begins.view(),
                                                 ends.view());

        stream
            << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
            << begins.copy_from(luisa::span{begin_offsets}) << ends.copy_from(luisa::span{end_offsets})
            << sort_keys.copy_from(luisa::span{key_initial})
            << sort_keys_descending.copy_from(luisa::span{key_initial})
            << stable_sort_keys.copy_from(luisa::span{key_initial})
            << stable_sort_keys_descending.copy_from(luisa::span{key_initial})
            << sort_pair_keys.copy_from(luisa::span{key_initial})
            << sort_pair_values.copy_from(luisa::span{value_initial})
            << sort_pair_descending_keys.copy_from(luisa::span{key_initial})
            << sort_pair_descending_values.copy_from(luisa::span{value_initial})
            << stable_pair_keys.copy_from(luisa::span{key_initial})
            << stable_pair_values.copy_from(luisa::span{value_initial})
            << stable_pair_descending_keys.copy_from(luisa::span{key_initial})
            << stable_pair_descending_values.copy_from(luisa::span{value_initial}) << commands.commit();

        luisa::vector<int>  sort_keys_result(item_count);
        luisa::vector<int>  sort_keys_descending_result(item_count);
        luisa::vector<int>  stable_sort_keys_result(item_count);
        luisa::vector<int>  stable_sort_keys_descending_result(item_count);
        luisa::vector<int>  sort_pair_keys_result(item_count);
        luisa::vector<uint> sort_pair_values_result(item_count);
        luisa::vector<int>  sort_pair_descending_keys_result(item_count);
        luisa::vector<uint> sort_pair_descending_values_result(item_count);
        luisa::vector<int>  stable_pair_keys_result(item_count);
        luisa::vector<uint> stable_pair_values_result(item_count);
        luisa::vector<int>  stable_pair_descending_keys_result(item_count);
        luisa::vector<uint> stable_pair_descending_values_result(item_count);
        stream << sort_keys.copy_to(luisa::span{sort_keys_result})
               << sort_keys_descending.copy_to(luisa::span{sort_keys_descending_result})
               << stable_sort_keys.copy_to(luisa::span{stable_sort_keys_result})
               << stable_sort_keys_descending.copy_to(luisa::span{stable_sort_keys_descending_result})
               << sort_pair_keys.copy_to(luisa::span{sort_pair_keys_result})
               << sort_pair_values.copy_to(luisa::span{sort_pair_values_result})
               << sort_pair_descending_keys.copy_to(luisa::span{sort_pair_descending_keys_result})
               << sort_pair_descending_values.copy_to(luisa::span{sort_pair_descending_values_result})
               << stable_pair_keys.copy_to(luisa::span{stable_pair_keys_result})
               << stable_pair_values.copy_to(luisa::span{stable_pair_values_result})
               << stable_pair_descending_keys.copy_to(luisa::span{stable_pair_descending_keys_result})
               << stable_pair_descending_values.copy_to(luisa::span{stable_pair_descending_values_result})
               << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            if(!is_segment_item(i))
            {
                expect(sort_keys_result[i] == key_sentinel);
                expect(sort_keys_descending_result[i] == key_sentinel);
                expect(stable_sort_keys_result[i] == key_sentinel);
                expect(stable_sort_keys_descending_result[i] == key_sentinel);
                expect(sort_pair_keys_result[i] == key_sentinel);
                expect(sort_pair_values_result[i] == value_sentinel);
                expect(sort_pair_descending_keys_result[i] == key_sentinel);
                expect(sort_pair_descending_values_result[i] == value_sentinel);
                expect(stable_pair_keys_result[i] == key_sentinel);
                expect(stable_pair_values_result[i] == value_sentinel);
                expect(stable_pair_descending_keys_result[i] == key_sentinel);
                expect(stable_pair_descending_values_result[i] == value_sentinel);
                continue;
            }

            expect(sort_keys_result[i] == keys[ascending_order[i]]);
            expect(sort_keys_descending_result[i] == keys[descending_order[i]]);
            expect(stable_sort_keys_result[i] == keys[ascending_order[i]]);
            expect(stable_sort_keys_descending_result[i] == keys[descending_order[i]]);
            expect(sort_pair_keys_result[i] == keys[ascending_order[i]]);
            expect(sort_pair_values_result[i] == values[ascending_order[i]]);
            expect(sort_pair_descending_keys_result[i] == keys[descending_order[i]]);
            expect(sort_pair_descending_values_result[i] == values[descending_order[i]]);
            expect(stable_pair_keys_result[i] == keys[ascending_order[i]]);
            expect(stable_pair_values_result[i] == values[ascending_order[i]]);
            expect(stable_pair_descending_keys_result[i] == keys[descending_order[i]]);
            expect(stable_pair_descending_values_result[i] == values[descending_order[i]]);
        }
    };

    "device_segmented_radix_sort_honors_partial_bit_ranges"_test = [&]
    {
        const luisa::vector<uint> keys{
            999u, 44u, 12u, 76u, 998u, 63u, 31u, 95u, 27u, 997u, 84u, 20u, 52u, 16u, 80u, 48u, 24u, 56u, 4u};
        luisa::vector<uint> values(item_count);
        std::iota(values.begin(), values.end(), 0u);
        constexpr uint begin_bit   = 2u;
        constexpr uint end_bit     = 6u;
        auto           radix_field = [](uint key)
        { return (key >> begin_bit) & ((1u << (end_bit - begin_bit)) - 1u); };
        auto ascending_order =
            segmented_order(keys,
                            [&](uint lhs, uint rhs) { return radix_field(lhs) < radix_field(rhs); });
        auto descending_order =
            segmented_order(keys,
                            [&](uint lhs, uint rhs) { return radix_field(lhs) > radix_field(rhs); });

        auto                keys_input             = device.create_buffer<uint>(item_count);
        auto                values_input           = device.create_buffer<uint>(item_count);
        auto                begins                 = device.create_buffer<uint>(segment_count);
        auto                ends                   = device.create_buffer<uint>(segment_count);
        auto                ascending_keys         = device.create_buffer<uint>(item_count);
        auto                descending_keys        = device.create_buffer<uint>(item_count);
        auto                ascending_pair_keys    = device.create_buffer<uint>(item_count);
        auto                ascending_pair_values  = device.create_buffer<uint>(item_count);
        auto                descending_pair_keys   = device.create_buffer<uint>(item_count);
        auto                descending_pair_values = device.create_buffer<uint>(item_count);
        luisa::vector<uint> initial(item_count, value_sentinel);
        auto temp_bytes = DeviceSegmentedRadixSort<>::GetSortPairsTempStorageBytes<uint, uint>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        segmented_radix_sort.SortKeys(commands,
                                      temp_storage.view(),
                                      keys_input.view(),
                                      ascending_keys.view(),
                                      item_count,
                                      segment_count,
                                      begins.view(),
                                      ends.view(),
                                      begin_bit,
                                      end_bit);
        segmented_radix_sort.SortKeysDescending(commands,
                                                temp_storage.view(),
                                                keys_input.view(),
                                                descending_keys.view(),
                                                item_count,
                                                segment_count,
                                                begins.view(),
                                                ends.view(),
                                                begin_bit,
                                                end_bit);
        segmented_radix_sort.SortPairs(commands,
                                       temp_storage.view(),
                                       keys_input.view(),
                                       ascending_pair_keys.view(),
                                       values_input.view(),
                                       ascending_pair_values.view(),
                                       item_count,
                                       segment_count,
                                       begins.view(),
                                       ends.view(),
                                       begin_bit,
                                       end_bit);
        segmented_radix_sort.SortPairsDescending(commands,
                                                 temp_storage.view(),
                                                 keys_input.view(),
                                                 descending_pair_keys.view(),
                                                 values_input.view(),
                                                 descending_pair_values.view(),
                                                 item_count,
                                                 segment_count,
                                                 begins.view(),
                                                 ends.view(),
                                                 begin_bit,
                                                 end_bit);
        stream << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
               << begins.copy_from(luisa::span{begin_offsets}) << ends.copy_from(luisa::span{end_offsets})
               << ascending_keys.copy_from(luisa::span{initial})
               << descending_keys.copy_from(luisa::span{initial})
               << ascending_pair_keys.copy_from(luisa::span{initial})
               << ascending_pair_values.copy_from(luisa::span{initial})
               << descending_pair_keys.copy_from(luisa::span{initial})
               << descending_pair_values.copy_from(luisa::span{initial}) << commands.commit();

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

        for(uint i = 0u; i < item_count; ++i)
        {
            if(!is_segment_item(i))
            {
                expect(ascending_keys_result[i] == value_sentinel);
                expect(descending_keys_result[i] == value_sentinel);
                expect(ascending_pair_keys_result[i] == value_sentinel);
                expect(ascending_pair_values_result[i] == value_sentinel);
                expect(descending_pair_keys_result[i] == value_sentinel);
                expect(descending_pair_values_result[i] == value_sentinel);
                continue;
            }
            expect(ascending_keys_result[i] == keys[ascending_order[i]]);
            expect(descending_keys_result[i] == keys[descending_order[i]]);
            expect(ascending_pair_keys_result[i] == keys[ascending_order[i]]);
            expect(ascending_pair_values_result[i] == values[ascending_order[i]]);
            expect(descending_pair_keys_result[i] == keys[descending_order[i]]);
            expect(descending_pair_values_result[i] == values[descending_order[i]]);
        }
    };

    "device_segmented_sort_handles_empty_work"_test = [&]
    {
        CommandList commands;
        segmented_sort.SortKeys(commands,
                                BufferView<uint>{},
                                BufferView<int>{},
                                BufferView<int>{},
                                0u,
                                0u,
                                BufferView<uint>{},
                                BufferView<uint>{});
        segmented_radix_sort.SortKeys(commands,
                                      BufferView<uint>{},
                                      BufferView<uint>{},
                                      BufferView<uint>{},
                                      0u,
                                      0u,
                                      BufferView<uint>{},
                                      BufferView<uint>{});
        stream << commands.commit() << synchronize();
        expect(DeviceSegmentedSort<>::GetSortKeysTempStorageBytes<int>(1u) == 0u);
        expect(DeviceSegmentedRadixSort<>::GetSortKeysTempStorageBytes<uint>(1u) == 0u);
    };

    "device_segmented_radix_sort_handles_signed_and_zero_width_keys"_test = [&]
    {
        constexpr uint           count = 8u;
        const luisa::vector<int> keys{41, -7, 3, -7, 0, 22, -1, 3};
        luisa::vector<uint>      values(count);
        std::iota(values.begin(), values.end(), 0u);
        luisa::vector<int> expected_keys = keys;
        std::stable_sort(expected_keys.begin(), expected_keys.end());

        auto keys_input        = device.create_buffer<int>(count);
        auto values_input      = device.create_buffer<uint>(count);
        auto sorted_keys       = device.create_buffer<int>(count);
        auto zero_width_keys   = device.create_buffer<int>(count);
        auto zero_width_values = device.create_buffer<uint>(count);
        auto begins            = device.create_buffer<uint>(1u);
        auto ends              = device.create_buffer<uint>(1u);
        uint begin             = 0u;
        uint end               = count;
        auto temp_bytes = DeviceSegmentedRadixSort<>::GetSortPairsTempStorageBytes<int, uint>(count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        segmented_radix_sort.SortKeys(
            commands, temp_storage.view(), keys_input.view(), sorted_keys.view(), count, 1u, begins.view(), ends.view());
        segmented_radix_sort.SortPairs(commands,
                                       temp_storage.view(),
                                       keys_input.view(),
                                       zero_width_keys.view(),
                                       values_input.view(),
                                       zero_width_values.view(),
                                       count,
                                       1u,
                                       begins.view(),
                                       ends.view(),
                                       13u,
                                       13u);
        stream << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
               << begins.copy_from(luisa::span{&begin, 1u}) << ends.copy_from(luisa::span{&end, 1u})
               << commands.commit();

        luisa::vector<int>  sorted_keys_result(count);
        luisa::vector<int>  zero_width_keys_result(count);
        luisa::vector<uint> zero_width_values_result(count);
        stream << sorted_keys.copy_to(luisa::span{sorted_keys_result})
               << zero_width_keys.copy_to(luisa::span{zero_width_keys_result})
               << zero_width_values.copy_to(luisa::span{zero_width_values_result}) << synchronize();

        for(uint i = 0u; i < count; ++i)
        {
            expect(sorted_keys_result[i] == expected_keys[i]);
            expect(zero_width_keys_result[i] == keys[i]);
            expect(zero_width_values_result[i] == values[i]);
        }
    };
}
