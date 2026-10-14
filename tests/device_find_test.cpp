// Tests for DeviceFind.
// Covers first-match/no-match behavior, duplicate bounds, custom ordering,
// empty ranges, and zero-work calls.

#include <lcpp/parallel_primitive.h>
#include <algorithm>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
struct HeterogeneousLess
{
    [[nodiscard]] Bool operator()(const Int& lhs, const Float& rhs) const noexcept
    {
        return lhs < rhs;
    }
    [[nodiscard]] Bool operator()(const Float& lhs, const Int& rhs) const noexcept
    {
        return lhs < rhs;
    }
};
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

    DeviceFind<> find;
    find.create(device);

    "device_find_if_returns_first_match_or_item_count"_test = [&]
    {
        const luisa::vector<int> input_values{9, 4, 7, 4, 2, 6};
        auto                     input        = device.create_buffer<int>(input_values.size());
        auto                     first_even   = device.create_buffer<uint>(1u);
        auto                     no_match     = device.create_buffer<uint>(1u);
        auto                     empty_result = device.create_buffer<uint>(1u);
        stream << input.copy_from(luisa::span{input_values});

        auto        is_even          = [](const Int& value) noexcept { return value % 2 == 0; };
        auto        greater_than_ten = [](const Int& value) noexcept { return value > 10; };
        CommandList commands;
        find.FindIf(
            commands, BufferView<uint>{}, input.view(), first_even.view(), is_even, input_values.size());
        find.FindIf(commands,
                    BufferView<uint>{},
                    input.view(),
                    no_match.view(),
                    greater_than_ten,
                    input_values.size());
        find.FindIf(commands, BufferView<uint>{}, BufferView<int>{}, empty_result.view(), is_even, 0u);

        uint first_even_index = ~0u;
        uint no_match_index   = ~0u;
        uint empty_index      = ~0u;
        stream << commands.commit() << first_even.copy_to(luisa::span{&first_even_index, 1u})
               << no_match.copy_to(luisa::span{&no_match_index, 1u})
               << empty_result.copy_to(luisa::span{&empty_index, 1u}) << synchronize();

        expect(first_even_index == 1u);
        expect(no_match_index == input_values.size());
        expect(empty_index == 0u);
    };

    "device_find_tiled_tail_and_dense_matches"_test = [&]
    {
        constexpr uint     count = 100003u;
        luisa::vector<int> values(count, 0);
        auto               input   = device.create_buffer<int>(count);
        auto               output  = device.create_buffer<uint>(1u);
        auto               matches = [](const Int& value) noexcept { return value != 0; };
        for(uint position : {count, count - 1u, 4096u, 31u, 0u})
        {
            std::fill(values.begin(), values.end(), 0);
            if(position < count)
            {
                std::fill(values.begin() + position, values.end(), 1);
            }
            CommandList commands;
            find.FindIf(commands, BufferView<uint>{}, input.view(), output.view(), matches, count);
            uint result = ~0u;
            stream << input.copy_from(luisa::span{values}) << commands.commit()
                   << output.copy_to(luisa::span{&result, 1u}) << synchronize();
            expect(result == position);
        }
        auto indexed = [](const Int&) noexcept { return dispatch_id().x + 1u == dispatch_size().x; };
        CommandList commands;
        find.FindIf(commands, BufferView<uint>{}, input.view(), output.view(), indexed, count);
        uint result = ~0u;
        stream << commands.commit() << output.copy_to(luisa::span{&result, 1u}) << synchronize();
        expect(result == count - 1u);
    };

    "device_find_bounds_cover_duplicates_and_custom_order"_test = [&]
    {
        const luisa::vector<int>  ascending_range{1, 2, 2, 2, 5, 8};
        const luisa::vector<int>  ascending_queries{0, 2, 3, 8, 9};
        const luisa::vector<uint> expected_lower{0u, 1u, 4u, 5u, 6u};
        const luisa::vector<uint> expected_upper{0u, 4u, 4u, 6u, 6u};
        const luisa::vector<int>  descending_range{9, 7, 7, 4, 1};
        const luisa::vector<int>  descending_queries{10, 7, 6, 1, 0};
        const luisa::vector<uint> expected_descending_lower{0u, 1u, 3u, 4u, 5u};
        const luisa::vector<uint> expected_descending_upper{0u, 3u, 3u, 5u, 5u};

        auto ascending         = device.create_buffer<int>(ascending_range.size());
        auto ascending_values  = device.create_buffer<int>(ascending_queries.size());
        auto lower             = device.create_buffer<uint>(ascending_queries.size());
        auto upper             = device.create_buffer<uint>(ascending_queries.size());
        auto descending        = device.create_buffer<int>(descending_range.size());
        auto descending_values = device.create_buffer<int>(descending_queries.size());
        auto descending_lower  = device.create_buffer<uint>(descending_queries.size());
        auto descending_upper  = device.create_buffer<uint>(descending_queries.size());
        stream << ascending.copy_from(luisa::span{ascending_range})
               << ascending_values.copy_from(luisa::span{ascending_queries})
               << descending.copy_from(luisa::span{descending_range})
               << descending_values.copy_from(luisa::span{descending_queries});

        auto        less    = [](const Int& lhs, const Int& rhs) noexcept { return lhs < rhs; };
        auto        greater = [](const Int& lhs, const Int& rhs) noexcept { return lhs > rhs; };
        CommandList commands;
        find.LowerBound(commands,
                        BufferView<uint>{},
                        ascending.view(),
                        ascending_range.size(),
                        ascending_values.view(),
                        ascending_queries.size(),
                        lower.view(),
                        less);
        find.UpperBound(commands,
                        BufferView<uint>{},
                        ascending.view(),
                        ascending_range.size(),
                        ascending_values.view(),
                        ascending_queries.size(),
                        upper.view(),
                        less);
        find.LowerBound(commands,
                        BufferView<uint>{},
                        descending.view(),
                        descending_range.size(),
                        descending_values.view(),
                        descending_queries.size(),
                        descending_lower.view(),
                        greater);
        find.UpperBound(commands,
                        BufferView<uint>{},
                        descending.view(),
                        descending_range.size(),
                        descending_values.view(),
                        descending_queries.size(),
                        descending_upper.view(),
                        greater);

        luisa::vector<uint> lower_result(ascending_queries.size());
        luisa::vector<uint> upper_result(ascending_queries.size());
        luisa::vector<uint> descending_lower_result(descending_queries.size());
        luisa::vector<uint> descending_upper_result(descending_queries.size());
        stream << commands.commit() << lower.copy_to(luisa::span{lower_result})
               << upper.copy_to(luisa::span{upper_result})
               << descending_lower.copy_to(luisa::span{descending_lower_result})
               << descending_upper.copy_to(luisa::span{descending_upper_result}) << synchronize();

        for(size_t index = 0u; index < ascending_queries.size(); ++index)
        {
            expect(lower_result[index] == expected_lower[index]);
            expect(upper_result[index] == expected_upper[index]);
            expect(descending_lower_result[index] == expected_descending_lower[index]);
            expect(descending_upper_result[index] == expected_descending_upper[index]);
        }
    };

    "device_find_sorted_bound_variants_match_sorted_query_contract"_test = [&]
    {
        const luisa::vector<int>  range_values{1, 2, 2, 4, 8};
        const luisa::vector<int>  query_values{0, 2, 2, 5, 9};
        const luisa::vector<uint> expected_lower{0u, 1u, 1u, 4u, 5u};
        const luisa::vector<uint> expected_upper{0u, 3u, 3u, 4u, 5u};
        auto                      range  = device.create_buffer<int>(range_values.size());
        auto                      values = device.create_buffer<int>(query_values.size());
        auto                      lower  = device.create_buffer<uint>(query_values.size());
        auto                      upper  = device.create_buffer<uint>(query_values.size());
        stream << range.copy_from(luisa::span{range_values}) << values.copy_from(luisa::span{query_values});

        auto        less = [](const Int& lhs, const Int& rhs) noexcept { return lhs < rhs; };
        CommandList commands;
        find.LowerBoundSortedValues(commands,
                                    BufferView<uint>{},
                                    range.view(),
                                    range_values.size(),
                                    values.view(),
                                    query_values.size(),
                                    lower.view(),
                                    less);
        find.UpperBoundSortedValues(commands,
                                    BufferView<uint>{},
                                    range.view(),
                                    range_values.size(),
                                    values.view(),
                                    query_values.size(),
                                    upper.view(),
                                    less);

        luisa::vector<uint> lower_result(query_values.size());
        luisa::vector<uint> upper_result(query_values.size());
        stream << commands.commit() << lower.copy_to(luisa::span{lower_result})
               << upper.copy_to(luisa::span{upper_result}) << synchronize();
        for(size_t index = 0u; index < query_values.size(); ++index)
        {
            expect(lower_result[index] == expected_lower[index]);
            expect(upper_result[index] == expected_upper[index]);
        }
    };

    "device_find_bounds_handle_empty_ranges_and_zero_queries"_test = [&]
    {
        expect(DeviceFind<>::GetTempStorageBytes<int>(17u) == 0u);
        expect(DeviceFind<>::GetTempStorageBytes<int, int>(0u, 3u) == 0u);

        const luisa::vector<int> queries{1, 2, 3};
        auto                     query_values = device.create_buffer<int>(queries.size());
        auto                     lower        = device.create_buffer<uint>(queries.size());
        auto                     upper        = device.create_buffer<uint>(queries.size());
        auto                     valid_range  = device.create_buffer<int>(1u);
        stream << query_values.copy_from(luisa::span{queries});

        auto        less = [](const Int& lhs, const Int& rhs) noexcept { return lhs < rhs; };
        CommandList commands;
        find.LowerBound(commands,
                        BufferView<uint>{},
                        BufferView<int>{},
                        0u,
                        query_values.view(),
                        queries.size(),
                        lower.view(),
                        less);
        find.UpperBound(commands,
                        BufferView<uint>{},
                        BufferView<int>{},
                        0u,
                        query_values.view(),
                        queries.size(),
                        upper.view(),
                        less);
        find.LowerBound(
            commands, BufferView<uint>{}, valid_range.view(), 1u, BufferView<int>{}, 0u, BufferView<uint>{}, less);

        luisa::vector<uint> lower_result(queries.size(), ~0u);
        luisa::vector<uint> upper_result(queries.size(), ~0u);
        stream << commands.commit() << lower.copy_to(luisa::span{lower_result})
               << upper.copy_to(luisa::span{upper_result}) << synchronize();
        for(size_t index = 0u; index < queries.size(); ++index)
        {
            expect(lower_result[index] == 0u);
            expect(upper_result[index] == 0u);
        }
    };

    "device_find_bounds_support_heterogeneous_relations"_test = [&]
    {
        const luisa::vector<int>   range_values{1, 3, 5};
        const luisa::vector<float> query_values{0.5f, 3.0f, 4.5f, 9.0f};
        const luisa::vector<uint>  expected_lower{0u, 1u, 2u, 3u};
        const luisa::vector<uint>  expected_upper{0u, 2u, 2u, 3u};
        auto                       range   = device.create_buffer<int>(range_values.size());
        auto                       queries = device.create_buffer<float>(query_values.size());
        auto                       lower   = device.create_buffer<uint>(query_values.size());
        auto                       upper   = device.create_buffer<uint>(query_values.size());
        stream << range.copy_from(luisa::span{range_values}) << queries.copy_from(luisa::span{query_values});

        CommandList commands;
        find.LowerBound(commands,
                        BufferView<uint>{},
                        range.view(),
                        range_values.size(),
                        queries.view(),
                        query_values.size(),
                        lower.view(),
                        HeterogeneousLess{});
        find.UpperBound(commands,
                        BufferView<uint>{},
                        range.view(),
                        range_values.size(),
                        queries.view(),
                        query_values.size(),
                        upper.view(),
                        HeterogeneousLess{});

        luisa::vector<uint> lower_result(query_values.size());
        luisa::vector<uint> upper_result(query_values.size());
        stream << commands.commit() << lower.copy_to(luisa::span{lower_result})
               << upper.copy_to(luisa::span{upper_result}) << synchronize();
        for(size_t index = 0u; index < query_values.size(); ++index)
        {
            expect(lower_result[index] == expected_lower[index]);
            expect(upper_result[index] == expected_upper[index]);
        }
    };
}
