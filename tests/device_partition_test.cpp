// Test for DevicePartition.
// This test covers flagged and predicate partition ordering and selected counts.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
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

    DevicePartition<> partition;
    partition.create(device);

    "device_partition_if_preserves_pointwise_predicates"_test = [&]
    {
        auto check_predicates = [&]<typename T>(T)
        {
            for(uint count : {1u, 33u, 4099u})
            {
                luisa::vector<T> input(count);
                for(uint i = 0u; i < count; ++i)
                {
                    input[i] = static_cast<T>(i);
                }
                auto in       = device.create_buffer<T>(count);
                auto out      = device.create_buffer<T>(count);
                auto selected = device.create_buffer<uint>(1u);
                auto calls    = device.create_buffer<uint>(count);
                auto temp = device.create_buffer<uint>(
                    bytes_to_uint_count(DevicePartition<>::GetTempStorageBytes(count)));
                Callable<bool(T)> indexed = [count](Var<T> value) noexcept
                {
                    return (cast<uint>(value) == dispatch_id().x)
                           & (dispatch_size().x == count) & (dispatch_id().x % 3u == 0u);
                };
                Callable<bool(T)> collective = [](Var<T> value) noexcept
                { return warp_prefix_sum(cast<uint>(value)) % 7u == 0u; };
                Callable<bool(T)> atomic = [&calls](Var<T> value) noexcept
                {
                    UInt index    = cast<uint>(value);
                    UInt previous = calls->atomic(index).fetch_add(1u);
                    return (previous == 0u) & (index % 3u == 1u);
                };
                stream << in.copy_from(luisa::span{input});
                for(uint mode = 0u; mode < 3u; ++mode)
                {
                    luisa::vector<T> expected;
                    luisa::vector<uint> flags(count);
                    for(uint i = 0u; i < count; ++i)
                    {
                        uint lane = i % 32u;
                        uint prefix = lane * (i - lane)
                                      + (lane == 0u ? 0u : lane * (lane - 1u) / 2u);
                        flags[i] = mode == 1u ? prefix % 7u == 0u :
                                               i % 3u == (mode == 0u ? 0u : 1u);
                        if(flags[i] != 0u)
                        {
                            expected.emplace_back(input[i]);
                        }
                    }
                    auto expected_count = expected.size();
                    for(uint i = count; i > 0u; --i)
                    {
                        if(flags[i - 1u] == 0u)
                        {
                            expected.emplace_back(input[i - 1u]);
                        }
                    }
                    const auto& predicate = mode == 0u ? indexed : mode == 1u ? collective : atomic;
                    CommandList commands;
                    partition.If(commands, temp.view(), in.view(), out.view(), selected.view(), count, predicate);
                    luisa::vector<T> result(count, static_cast<T>(-1));
                    luisa::vector<uint> call_counts(count, 0u);
                    uint size = 99u;
                    stream << out.copy_from(luisa::span{result})
                           << calls.copy_from(luisa::span{call_counts}) << commands.commit()
                           << out.copy_to(luisa::span{result})
                           << selected.copy_to(luisa::span{&size, 1u})
                           << calls.copy_to(luisa::span{call_counts}) << synchronize();
                    expect(size == expected_count);
                    expect(result == expected);
                    if(mode == 2u)
                    {
                        expect(std::all_of(call_counts.begin(), call_counts.end(),
                                           [](uint calls) { return calls == 1u; }));
                    }
                }
            }
        };
        check_predicates(int{});
        check_predicates(luisa::ulong{});
    };

    "device_partition_flagged_and_if_ordering"_test = [&]
    {
        constexpr uint32_t      item_count = 12u;
        luisa::vector<int32_t>  input(item_count);
        luisa::vector<uint32_t> flags(item_count);
        for(uint32_t i = 0u; i < item_count; ++i)
        {
            input[i] = static_cast<int32_t>(i);
            flags[i] = i % 3u == 0u ? 1u : 0u;
        }

        auto input_buffer     = device.create_buffer<int32_t>(item_count);
        auto flags_buffer     = device.create_buffer<uint32_t>(item_count);
        auto flagged_output   = device.create_buffer<int32_t>(item_count);
        auto predicate_output = device.create_buffer<int32_t>(item_count);
        auto flagged_count    = device.create_buffer<uint32_t>(1u);
        auto predicate_count  = device.create_buffer<uint32_t>(1u);
        auto temp_bytes       = DevicePartition<>::GetTempStorageBytes(item_count);
        auto temp_storage     = device.create_buffer<uint32_t>(bytes_to_uint_count(temp_bytes));

        Callable<bool(int32_t)> less_than_seven = [](Int value) noexcept { return value < 7; };

        CommandList commands;
        partition.Flagged(commands,
                          temp_storage.view(),
                          input_buffer.view(),
                          flags_buffer.view(),
                          flagged_output.view(),
                          flagged_count.view(),
                          item_count);
        partition.If(commands,
                     temp_storage.view(),
                     input_buffer.view(),
                     predicate_output.view(),
                     predicate_count.view(),
                     item_count,
                     less_than_seven);

        stream << input_buffer.copy_from(luisa::span{input})
               << flags_buffer.copy_from(luisa::span{flags}) << commands.commit();

        luisa::vector<int32_t> flagged_result(item_count);
        luisa::vector<int32_t> predicate_result(item_count);
        uint32_t               flagged_size   = 0u;
        uint32_t               predicate_size = 0u;
        stream << flagged_output.copy_to(luisa::span{flagged_result})
               << predicate_output.copy_to(luisa::span{predicate_result})
               << flagged_count.copy_to(luisa::span{&flagged_size, 1u})
               << predicate_count.copy_to(luisa::span{&predicate_size, 1u}) << synchronize();

        luisa::vector<int32_t> expected_flagged;
        luisa::vector<int32_t> expected_predicate;
        for(uint32_t i = 0u; i < item_count; ++i)
        {
            if(flags[i] != 0u)
            {
                expected_flagged.emplace_back(input[i]);
            }
            if(input[i] < 7)
            {
                expected_predicate.emplace_back(input[i]);
            }
        }
        for(uint32_t i = item_count; i-- > 0u;)
        {
            if(flags[i] == 0u)
            {
                expected_flagged.emplace_back(input[i]);
            }
            if(input[i] >= 7)
            {
                expected_predicate.emplace_back(input[i]);
            }
        }

        expect(flagged_size == 4u);
        expect(predicate_size == 7u);
        for(uint32_t i = 0u; i < item_count; ++i)
        {
            expect(flagged_result[i] == expected_flagged[i]);
            expect(predicate_result[i] == expected_predicate[i]);
        }
    };

    "device_partition_preserves_wide_values_and_boolean_flags"_test = [&]
    {
        constexpr uint count = 17u;
        std::array<luisa::ulong, count> values{};
        std::array<bool, count> flags{};
        for(uint i = 0u; i < count; ++i) { values[i] = (luisa::ulong{1} << 40u) + i; flags[i] = i % 3u == 1u; }
        auto in = device.create_buffer<luisa::ulong>(count);
        auto flag = device.create_buffer<bool>(count);
        auto out = device.create_buffer<luisa::ulong>(count);
        auto selected = device.create_buffer<uint>(1u);
        auto temp = device.create_buffer<uint>(bytes_to_uint_count(DevicePartition<>::GetTempStorageBytes(count)));
        Callable<bool(luisa::ulong)> even = [](Var<luisa::ulong> value) noexcept { return value % luisa::ulong{2} == luisa::ulong{0}; };
        stream << in.copy_from(luisa::span{values}) << flag.copy_from(luisa::span{flags});
        for(uint mode = 0u; mode < 2u; ++mode)
        {
            luisa::vector<luisa::ulong> expected;
            for(uint i = 0u; i < count; ++i)
            {
                if(mode == 0u ? flags[i] : values[i] % 2u == 0u) { expected.emplace_back(values[i]); }
            }
            auto expected_count = expected.size();
            for(uint i = count; i > 0u; --i)
            {
                if(!(mode == 0u ? flags[i - 1u] : values[i - 1u] % 2u == 0u)) { expected.emplace_back(values[i - 1u]); }
            }
            CommandList commands;
            if(mode == 0u) { partition.Flagged(commands, temp.view(), in.view(), flag.view(), out.view(), selected.view(), count); }
            else { partition.If(commands, temp.view(), in.view(), out.view(), selected.view(), count, even); }
            luisa::vector<luisa::ulong> result(count);
            uint size = 0u;
            stream << commands.commit() << out.copy_to(luisa::span{result}) << selected.copy_to(luisa::span{&size, 1u}) << synchronize();
            expect(size == expected_count);
            expect(result == expected);
        }
    };

    "device_partition_multitile_reverse_rejects_and_empty_tiles"_test = [&]
    {
        constexpr uint      count = 230413u;
        luisa::vector<int>  input(count);
        luisa::vector<uint> flags(count);
        for(uint i = 0u; i < count; ++i)
        {
            input[i] = static_cast<int>(i);
        }
        auto in             = device.create_buffer<int>(count);
        auto flag           = device.create_buffer<uint>(count);
        auto out            = device.create_buffer<int>(count);
        auto selected_count = device.create_buffer<uint>(1u);
        auto temp =
            device.create_buffer<uint>(bytes_to_uint_count(DevicePartition<>::GetTempStorageBytes(count)));
        stream << in.copy_from(luisa::span{input});
        for(uint pattern = 0u; pattern < 4u; ++pattern)
        {
            luisa::vector<int> expected;
            for(uint i = 0u; i < count; ++i)
            {
                flags[i] = pattern == 0u ? 0u :
                           pattern == 1u ? 1u :
                           pattern == 2u ? i % 3u :
                                           uint(i < 11u || i > 200000u);
                if(flags[i] != 0u)
                {
                    expected.emplace_back(input[i]);
                }
            }
            auto expected_count = expected.size();
            for(uint i = count; i > 0u; --i)
            {
                if(flags[i - 1u] == 0u)
                {
                    expected.emplace_back(input[i - 1u]);
                }
            }
            CommandList commands;
            partition.Flagged(
                commands, temp.view(), in.view(), flag.view(), out.view(), selected_count.view(), count);
            stream << flag.copy_from(luisa::span{flags}) << commands.commit();
            luisa::vector<int> result(count);
            uint               selected = 0u;
            stream << out.copy_to(luisa::span{result})
                   << selected_count.copy_to(luisa::span{&selected, 1u}) << synchronize();
            expect(selected == expected_count);
            expect(result == expected);
        }
    };

    "device_partition_three_way_preserves_category_order"_test = [&]
    {
        constexpr uint32_t     item_count = 12u;
        luisa::vector<int32_t> input(item_count);
        for(uint32_t i = 0u; i < item_count; ++i)
        {
            input[i] = static_cast<int32_t>(i);
        }

        auto input_buffer      = device.create_buffer<int32_t>(item_count);
        auto first_output      = device.create_buffer<int32_t>(item_count);
        auto second_output     = device.create_buffer<int32_t>(item_count);
        auto unselected_output = device.create_buffer<int32_t>(item_count);
        auto selected_counts   = device.create_buffer<uint32_t>(2u);
        auto temp_bytes        = DevicePartition<>::GetThreeWayTempStorageBytes(item_count);
        auto temp_storage      = device.create_buffer<uint32_t>(bytes_to_uint_count(temp_bytes));

        Callable<bool(int32_t)> divisible_by_three = [](Int value) noexcept
        { return value % 3 == 0; };
        Callable<bool(int32_t)> less_than_eight = [](Int value) noexcept { return value < 8; };

        CommandList commands;
        partition.If(commands,
                     temp_storage.view(),
                     input_buffer.view(),
                     first_output.view(),
                     second_output.view(),
                     unselected_output.view(),
                     selected_counts.view(),
                     item_count,
                     divisible_by_three,
                     less_than_eight);
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit();

        luisa::vector<int32_t>   first_result(item_count);
        luisa::vector<int32_t>   second_result(item_count);
        luisa::vector<int32_t>   unselected_result(item_count);
        std::array<uint32_t, 2u> counts{};
        stream << first_output.copy_to(luisa::span{first_result})
               << second_output.copy_to(luisa::span{second_result})
               << unselected_output.copy_to(luisa::span{unselected_result})
               << selected_counts.copy_to(luisa::span{counts}) << synchronize();

        constexpr std::array<int32_t, 4u> expected_first{0, 3, 6, 9};
        constexpr std::array<int32_t, 5u> expected_second{1, 2, 4, 5, 7};
        constexpr std::array<int32_t, 3u> expected_unselected{11, 10, 8};
        expect(counts[0u] == expected_first.size());
        expect(counts[1u] == expected_second.size());
        for(size_t i = 0u; i < expected_first.size(); ++i)
        {
            expect(first_result[i] == expected_first[i]);
        }
        for(size_t i = 0u; i < expected_second.size(); ++i)
        {
            expect(second_result[i] == expected_second[i]);
        }
        for(size_t i = 0u; i < expected_unselected.size(); ++i)
        {
            expect(unselected_result[i] == expected_unselected[i]);
        }
    };

    "device_partition_three_way_zero_input_writes_both_counts"_test = [&]
    {
        std::array<uint32_t, 2u> counts{7u, 9u};
        auto                     count_buffer = device.create_buffer<uint32_t>(counts.size());
        Callable<bool(int32_t)>  first        = [](Int value) noexcept { return value < 0; };
        Callable<bool(int32_t)>  second       = [](Int value) noexcept { return value > 0; };
        CommandList              commands;
        partition.If(commands,
                     BufferView<uint>{},
                     BufferView<int32_t>{},
                     BufferView<int32_t>{},
                     BufferView<int32_t>{},
                     BufferView<int32_t>{},
                     count_buffer.view(),
                     0u,
                     first,
                     second);
        stream << count_buffer.copy_from(luisa::span{counts}) << commands.commit()
               << count_buffer.copy_to(luisa::span{counts}) << synchronize();
        expect(counts[0u] == 0u);
        expect(counts[1u] == 0u);
    };
}
