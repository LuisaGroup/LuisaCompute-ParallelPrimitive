// Tests for DeviceFor.
// Covers bounded dispatch and lowercase compatibility entry points.

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

    DeviceFor device_for;
    device_for.create(device);

    "devicefor_mutates_exactly_the_requested_prefix"_test = [&]
    {
        constexpr uint     item_count   = 301u;
        constexpr uint     prefix_count = 173u;
        auto               buffer       = device.create_buffer<int>(item_count);
        luisa::vector<int> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            values[i] = static_cast<int>(i) - 150;
        }

        Callable<void(int&)> affine = [](Int& value) noexcept { value = value * 3 + 7; };

        CommandList commands;
        device_for.ForEachN(commands, buffer.view(), prefix_count, affine);
        stream << buffer.copy_from(luisa::span{values}) << commands.commit()
               << buffer.copy_to(luisa::span{values}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            int initial  = static_cast<int>(i) - 150;
            int expected = i < prefix_count ? initial * 3 + 7 : initial;
            expect(values[i] == expected);
        }
        expect(DeviceFor::GetTempStorageBytes(item_count) == 0u);
    };

    "devicefor_lowercase_compatibility_entry_points_execute_work"_test = [&]
    {
        constexpr uint     item_count = 193u;
        auto               buffer     = device.create_buffer<int>(item_count);
        luisa::vector<int> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            values[i] = static_cast<int>(i);
        }

        Callable<void(int&)> twice     = [](Int& value) noexcept { value *= 2; };
        Callable<void(int&)> add_three = [](Int& value) noexcept { value += 3; };
        Callable<void(int&)> negate    = [](Int& value) noexcept { value = -value; };

        CommandList commands;
        device_for.for_each_n(commands, buffer.view(), 129u, twice);
        device_for.for_each_extents(commands, buffer.view(), 65u, add_three);
        device_for.for_each(commands, buffer.view(), item_count, negate);
        stream << buffer.copy_from(luisa::span{values}) << commands.commit()
               << buffer.copy_to(luisa::span{values}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            int expected = static_cast<int>(i);
            if(i < 129u)
            {
                expected *= 2;
            }
            if(i < 65u)
            {
                expected += 3;
            }
            expected = -expected;
            expect(values[i] == expected);
        }
    };

    "devicefor_bulk_invokes_the_index_operation"_test = [&]
    {
        constexpr uint       item_count   = 37u;
        auto                 output       = device.create_buffer<uint>(item_count);
        Callable<void(uint)> write_square = [&output](UInt index) noexcept
        { output->write(index, index * index); };

        CommandList commands;
        device_for.Bulk(commands, item_count, write_square);
        device_for.Bulk(commands, 0u, Callable<void(uint)>{[](UInt) noexcept {}});

        std::array<uint, item_count> result{};
        stream << commands.commit() << output.copy_to(luisa::span{result}) << synchronize();
        for(uint index = 0u; index < item_count; ++index)
        {
            expect(result[index] == index * index);
        }
    };

    "devicefor_copy_variants_observe_values_without_mutating_input"_test = [&]
    {
        const std::array<int, 6u> input_values{1, 2, 3, 4, 5, 6};
        auto                      input          = device.create_buffer<int>(input_values.size());
        auto                      prefix_counter = device.create_buffer<uint>(1u);
        auto                      full_counter   = device.create_buffer<uint>(1u);
        Callable<void(int)>       count_prefix_odds = [&prefix_counter](Int value) noexcept
        {
            $if((value & 1) != 0)
            {
                prefix_counter->atomic(0u).fetch_add(1u);
            };
        };
        Callable<void(int)> count_all_odds = [&full_counter](Int value) noexcept
        {
            $if((value & 1) != 0)
            {
                full_counter->atomic(0u).fetch_add(1u);
            };
        };
        Callable<void(int&)> mutate_copy = [](Int& value) noexcept { value = -value; };

        uint        zero = 0u;
        CommandList commands;
        device_for.ForEachCopyN(commands, input.view(), 4u, count_prefix_odds);
        device_for.ForEachCopy(commands, input.view(), count_all_odds);
        device_for.ForEachCopy(commands, input.view(), mutate_copy);

        std::array<int, 6u> result{};
        uint                prefix_count = 0u;
        uint                full_count   = 0u;
        stream << input.copy_from(luisa::span{input_values})
               << prefix_counter.copy_from(luisa::span{&zero, 1u})
               << full_counter.copy_from(luisa::span{&zero, 1u}) << commands.commit()
               << input.copy_to(luisa::span{result})
               << prefix_counter.copy_to(luisa::span{&prefix_count, 1u})
               << full_counter.copy_to(luisa::span{&full_count, 1u}) << synchronize();

        for(size_t index = 0u; index < input_values.size(); ++index)
        {
            expect(result[index] == input_values[index]);
        }
        expect(prefix_count == 2u);
        expect(full_count == 3u);
    };

    "devicefor_extents_and_layout_emit_expected_coordinates"_test = [&]
    {
        constexpr std::array<size_t, 2u> extents{2u, 3u};
        auto                             row_major    = device.create_buffer<uint>(6u);
        auto                             column_major = device.create_buffer<uint>(6u);
        Callable<void(uint, uint, uint)> write_row = [&row_major](UInt linear, UInt row, UInt column) noexcept
        { row_major->write(linear, row * 10u + column); };
        Callable<void(uint, uint, uint)> write_column = [&column_major](UInt linear, UInt row, UInt column) noexcept
        { column_major->write(linear, row * 10u + column); };

        CommandList commands;
        device_for.ForEachInExtents(commands, extents, write_row);
        device_for.ForEachInLayout(commands, extents, DeviceForLayout::COLUMN_MAJOR, write_column);
        device_for.ForEachInExtents(commands,
                                    std::array<size_t, 2u>{2u, 0u},
                                    Callable<void(uint, uint, uint)>{[](UInt, UInt, UInt) noexcept {}});

        std::array<uint, 6u> row_result{};
        std::array<uint, 6u> column_result{};
        stream << commands.commit() << row_major.copy_to(luisa::span{row_result})
               << column_major.copy_to(luisa::span{column_result}) << synchronize();
        const std::array<uint, 6u> expected_row{0u, 1u, 2u, 10u, 11u, 12u};
        const std::array<uint, 6u> expected_column{0u, 10u, 1u, 11u, 2u, 12u};
        for(size_t index = 0u; index < expected_row.size(); ++index)
        {
            expect(row_result[index] == expected_row[index]);
            expect(column_result[index] == expected_column[index]);
        }
    };
}
