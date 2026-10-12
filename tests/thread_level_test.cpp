// Tests for thread-level primitives.
// Covers portable cache modifiers, search utilities, and value preservation.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/core/logging.h>
#include <luisa/dsl/builtin.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
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

    constexpr uint item_count      = 257u;
    auto           input           = device.create_buffer<int>(item_count);
    auto           output          = device.create_buffer<int>(item_count);
    auto           volatile_output = device.create_buffer<int>(item_count);

    luisa::vector<int> host_input(item_count);
    luisa::vector<int> host_output(item_count);
    luisa::vector<int> host_volatile_output(item_count);
    for(uint i = 0u; i < item_count; ++i)
    {
        host_input[i] = static_cast<int>(i * 3u + 1u);
    }

    "thread_load_and_store_preserve_values_for_portable_cache_modifiers"_test = [&]
    {
        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> output_buffer,
                             BufferVar<int> volatile_output_buffer,
                             UInt           count) noexcept
        {
            set_block_size(128u);
            UInt index = dispatch_id().x;
            $if(index < count)
            {
                Int value = ThreadLoad<CacheLoadModifier::LOAD_DEFAULT>(input_buffer, index);
                ThreadStore<CacheStoreModifier::STORE_DEFAULT>(output_buffer, index, value + 7);

                Int volatile_value = ThreadLoad<CacheLoadModifier::LOAD_VOLATILE>(input_buffer, index);
                ThreadStore<CacheStoreModifier::STORE_VOLATILE>(volatile_output_buffer, index, volatile_value - 2);
            };
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << shader(input, output, volatile_output, item_count).dispatch(item_count)
               << output.copy_to(luisa::span{host_output})
               << volatile_output.copy_to(luisa::span{host_volatile_output}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(host_output[i] == host_input[i] + 7);
            expect(host_volatile_output[i] == host_input[i] - 2);
        }
    };

    "thread_search_matches_sorted_range_boundaries_and_merge_paths"_test = [&]
    {
        constexpr uint query_count    = 4u;
        constexpr uint diagonal_count = 8u;
        auto           range          = device.create_buffer<int>(5u);
        auto           queries        = device.create_buffer<int>(query_count);
        auto           merge_a        = device.create_buffer<int>(3u);
        auto           merge_b        = device.create_buffer<int>(4u);
        auto           lower_output   = device.create_buffer<uint>(query_count);
        auto           upper_output   = device.create_buffer<uint>(query_count);
        auto           path_output    = device.create_buffer<uint2>(diagonal_count);

        luisa::vector<int>   host_range{1, 3, 3, 7, 9};
        luisa::vector<int>   host_queries{0, 3, 4, 10};
        luisa::vector<int>   host_merge_a{1, 4, 7};
        luisa::vector<int>   host_merge_b{2, 3, 6, 8};
        luisa::vector<uint>  host_lower(query_count);
        luisa::vector<uint>  host_upper(query_count);
        luisa::vector<uint2> host_paths(diagonal_count);

        Kernel1D kernel = [](BufferVar<int>   range_buffer,
                             BufferVar<int>   query_buffer,
                             BufferVar<int>   a_buffer,
                             BufferVar<int>   b_buffer,
                             BufferVar<uint>  lower_buffer,
                             BufferVar<uint>  upper_buffer,
                             BufferVar<uint2> path_buffer) noexcept
        {
            UInt index = dispatch_id().x;
            $if(index < 4u)
            {
                Int query = query_buffer.read(index);
                lower_buffer.write(index, LowerBound(range_buffer, 5u, query));
                upper_buffer.write(index, UpperBound(range_buffer, 5u, query));
            };
            path_buffer.write(index, MergePathSearch(index, a_buffer, b_buffer, 3u, 4u));
        };

        auto shader = device.compile(kernel);
        stream
            << range.copy_from(luisa::span{host_range}) << queries.copy_from(luisa::span{host_queries})
            << merge_a.copy_from(luisa::span{host_merge_a}) << merge_b.copy_from(luisa::span{host_merge_b})
            << shader(range, queries, merge_a, merge_b, lower_output, upper_output, path_output).dispatch(diagonal_count)
            << lower_output.copy_to(luisa::span{host_lower})
            << upper_output.copy_to(luisa::span{host_upper})
            << path_output.copy_to(luisa::span{host_paths}) << synchronize();

        constexpr uint  expected_lower[]{0u, 1u, 3u, 5u};
        constexpr uint  expected_upper[]{0u, 3u, 3u, 5u};
        constexpr uint2 expected_paths[]{
            {0u, 0u}, {1u, 0u}, {1u, 1u}, {1u, 2u}, {2u, 2u}, {2u, 3u}, {3u, 3u}, {3u, 4u}};
        for(uint i = 0u; i < query_count; ++i)
        {
            expect(host_lower[i] == expected_lower[i]);
            expect(host_upper[i] == expected_upper[i]);
        }
        for(uint i = 0u; i < diagonal_count; ++i)
        {
            expect(host_paths[i].x == expected_paths[i].x);
            expect(host_paths[i].y == expected_paths[i].y);
        }
    };

    "thread_stable_odd_even_sort_orders_keys_and_preserves_equal_key_order"_test = [&]
    {
        constexpr uint sort_item_count = 5u;
        auto           sorted_keys     = device.create_buffer<int>(sort_item_count);
        auto           sorted_values   = device.create_buffer<uint>(sort_item_count);
        auto           descending_keys = device.create_buffer<int>(sort_item_count);

        luisa::vector<int>  host_sorted_keys(sort_item_count);
        luisa::vector<uint> host_sorted_values(sort_item_count);
        luisa::vector<int>  host_descending_keys(sort_item_count);

        Kernel1D kernel = [](BufferVar<int> key_output, BufferVar<uint> value_output, BufferVar<int> descending_output) noexcept
        {
            ArrayVar<int, sort_item_count>  pair_keys;
            ArrayVar<uint, sort_item_count> pair_values;
            ArrayVar<int, sort_item_count>  keys_only;
            constexpr int                   input_keys[]{3, 1, 3, 2, 1};
            for(uint item = 0u; item < sort_item_count; ++item)
            {
                pair_keys[item]   = input_keys[item];
                pair_values[item] = item;
                keys_only[item]   = input_keys[item];
            }

            auto less    = [](const Int& lhs, const Int& rhs) noexcept { return lhs < rhs; };
            auto greater = [](const Int& lhs, const Int& rhs) noexcept { return lhs > rhs; };
            StableOddEvenSort(pair_keys, pair_values, less);
            StableOddEvenSort(keys_only, greater);

            for(uint item = 0u; item < sort_item_count; ++item)
            {
                key_output.write(item, pair_keys[item]);
                value_output.write(item, pair_values[item]);
                descending_output.write(item, keys_only[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(sorted_keys, sorted_values, descending_keys).dispatch(1u)
               << sorted_keys.copy_to(luisa::span{host_sorted_keys})
               << sorted_values.copy_to(luisa::span{host_sorted_values})
               << descending_keys.copy_to(luisa::span{host_descending_keys}) << synchronize();

        constexpr int  expected_keys[]{1, 1, 2, 3, 3};
        constexpr uint expected_values[]{1u, 4u, 3u, 0u, 2u};
        constexpr int  expected_descending[]{3, 3, 2, 1, 1};
        for(uint item = 0u; item < sort_item_count; ++item)
        {
            expect(host_sorted_keys[item] == expected_keys[item]);
            expect(host_sorted_values[item] == expected_values[item]);
            expect(host_descending_keys[item] == expected_descending[item]);
        }
    };

    "thread_exclusive_partial_scan_does_not_touch_invalid_items"_test = [&]
    {
        constexpr uint scan_item_count  = 5u;
        auto           partial_output   = device.create_buffer<int>(scan_item_count);
        auto           empty_output     = device.create_buffer<int>(scan_item_count);
        auto           aggregate_output = device.create_buffer<int>(1u);

        luisa::vector<int> host_partial(scan_item_count);
        luisa::vector<int> host_empty(scan_item_count);
        luisa::vector<int> host_aggregate(1u);

        Kernel1D kernel = [](BufferVar<int> partial_buffer, BufferVar<int> empty_buffer, BufferVar<int> aggregate_buffer) noexcept
        {
            ArrayVar<int, scan_item_count> input;
            ArrayVar<int, scan_item_count> partial;
            ArrayVar<int, scan_item_count> empty;
            constexpr int                  input_values[]{1, 2, 3, 100, 200};
            for(uint item = 0u; item < scan_item_count; ++item)
            {
                input[item]   = input_values[item];
                partial[item] = -99;
                empty[item]   = -99;
            }

            auto add = [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; };
            ThreadScan<int, scan_item_count> scan;
            Int aggregate = scan.ThreadScanExclusivePartial(input, partial, add, 3u, 10);
            scan.ThreadScanExclusivePartial(input, empty, add, 0u, 10);

            for(uint item = 0u; item < scan_item_count; ++item)
            {
                partial_buffer.write(item, partial[item]);
                empty_buffer.write(item, empty[item]);
            }
            aggregate_buffer.write(0u, aggregate);
        };

        auto shader = device.compile(kernel);
        stream << shader(partial_output, empty_output, aggregate_output).dispatch(1u)
               << partial_output.copy_to(luisa::span{host_partial})
               << empty_output.copy_to(luisa::span{host_empty})
               << aggregate_output.copy_to(luisa::span{host_aggregate}) << synchronize();

        constexpr int expected_partial[]{10, 11, 13, -99, -99};
        for(uint item = 0u; item < scan_item_count; ++item)
        {
            expect(host_partial[item] == expected_partial[item]);
            expect(host_empty[item] == -99);
        }
        expect(host_aggregate[0] == 16);
    };

    "thread_partial_reduce_and_inclusive_scan_respect_valid_items"_test = [&]
    {
        constexpr uint partial_item_count = 5u;
        auto           inclusive_output   = device.create_buffer<int>(partial_item_count);
        auto           empty_output       = device.create_buffer<int>(partial_item_count);
        auto           aggregate_output   = device.create_buffer<int>(2u);

        luisa::vector<int> host_inclusive(partial_item_count);
        luisa::vector<int> host_empty(partial_item_count);
        luisa::vector<int> host_aggregate(2u);

        Kernel1D kernel = [](BufferVar<int> inclusive_buffer, BufferVar<int> empty_buffer, BufferVar<int> aggregate_buffer) noexcept
        {
            ArrayVar<int, partial_item_count> input;
            ArrayVar<int, partial_item_count> inclusive;
            ArrayVar<int, partial_item_count> empty;
            constexpr int                     input_values[]{1, 2, 3, 100, 200};
            for(uint item = 0u; item < partial_item_count; ++item)
            {
                input[item]     = input_values[item];
                inclusive[item] = -99;
                empty[item]     = -99;
            }

            auto add     = [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; };
            Int  reduced = ThreadReduce<int, partial_item_count>().ReducePartial(input, add, 3u);
            ThreadScan<int, partial_item_count> scan;
            Int aggregate = scan.ThreadScanInclusivePartial(input, inclusive, add, 3u, 10);
            scan.ThreadScanInclusivePartial(input, empty, add, 0u, 10);

            for(uint item = 0u; item < partial_item_count; ++item)
            {
                inclusive_buffer.write(item, inclusive[item]);
                empty_buffer.write(item, empty[item]);
            }
            aggregate_buffer.write(0u, reduced);
            aggregate_buffer.write(1u, aggregate);
        };

        auto shader = device.compile(kernel);
        stream << shader(inclusive_output, empty_output, aggregate_output).dispatch(1u)
               << inclusive_output.copy_to(luisa::span{host_inclusive})
               << empty_output.copy_to(luisa::span{host_empty})
               << aggregate_output.copy_to(luisa::span{host_aggregate}) << synchronize();

        constexpr int expected_inclusive[]{11, 13, 16, -99, -99};
        for(uint item = 0u; item < partial_item_count; ++item)
        {
            expect(host_inclusive[item] == expected_inclusive[item]);
            expect(host_empty[item] == -99);
        }
        expect(host_aggregate[0] == 6);
        expect(host_aggregate[1] == 16);
    };
}
