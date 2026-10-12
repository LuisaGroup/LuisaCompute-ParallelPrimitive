// Test for DevicePartition.
// This test covers flagged and predicate partition ordering and selected counts.

#include <lcpp/parallel_primitive.h>

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
