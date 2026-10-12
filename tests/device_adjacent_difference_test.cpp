// Tests for DeviceAdjacentDifference.
// Covers left/right, copy/in-place, custom operators, and boundary items.

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

    DeviceAdjacentDifference<> adjacent_difference;
    adjacent_difference.create(device);

    "device_adjacent_difference_supports_all_four_entry_points"_test = [&]
    {
        constexpr uint      item_count = 5u;
        std::array<int, 5u> input{1, 4, 9, 16, 25};
        std::array<int, 5u> expected_left{1, 3, 5, 7, 9};
        std::array<int, 5u> expected_right{-3, -5, -7, -9, 25};
        auto                input_buffer   = device.create_buffer<int>(item_count);
        auto                left_copy      = device.create_buffer<int>(item_count);
        auto                right_copy     = device.create_buffer<int>(item_count);
        auto                left_in_place  = device.create_buffer<int>(item_count);
        auto                right_in_place = device.create_buffer<int>(item_count);
        auto temp_bytes   = DeviceAdjacentDifference<>::GetTempStorageBytes<int>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        adjacent_difference.SubtractLeftCopy(
            commands, temp_storage.view(), input_buffer.view(), left_copy.view(), item_count);
        adjacent_difference.SubtractRightCopy(
            commands, temp_storage.view(), input_buffer.view(), right_copy.view(), item_count);
        adjacent_difference.SubtractLeft(commands, temp_storage.view(), left_in_place.view(), item_count);
        adjacent_difference.SubtractRight(commands, temp_storage.view(), right_in_place.view(), item_count);

        stream << input_buffer.copy_from(luisa::span{input}) << left_in_place.copy_from(luisa::span{input})
               << right_in_place.copy_from(luisa::span{input}) << commands.commit();

        std::array<int, 5u> left_copy_result{};
        std::array<int, 5u> right_copy_result{};
        std::array<int, 5u> left_in_place_result{};
        std::array<int, 5u> right_in_place_result{};
        stream << left_copy.copy_to(luisa::span{left_copy_result})
               << right_copy.copy_to(luisa::span{right_copy_result})
               << left_in_place.copy_to(luisa::span{left_in_place_result})
               << right_in_place.copy_to(luisa::span{right_in_place_result}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(left_copy_result[i] == expected_left[i]);
            expect(right_copy_result[i] == expected_right[i]);
            expect(left_in_place_result[i] == expected_left[i]);
            expect(right_in_place_result[i] == expected_right[i]);
        }
    };

    "device_adjacent_difference_honors_custom_operator_and_boundaries"_test = [&]
    {
        constexpr uint      item_count = 4u;
        std::array<int, 4u> input{2, 3, 5, 7};
        std::array<int, 4u> expected{8, 13, 19, 7};
        auto                input_buffer  = device.create_buffer<int>(item_count);
        auto                output_buffer = device.create_buffer<int>(item_count);
        auto temp_bytes   = DeviceAdjacentDifference<>::GetTempStorageBytes<int>(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        Callable<int(int, int)> combine = [](Int lhs, Int rhs) noexcept { return lhs + 2 * rhs; };

        CommandList commands;
        adjacent_difference.SubtractRightCopy(
            commands, temp_storage.view(), input_buffer.view(), output_buffer.view(), item_count, combine);
        adjacent_difference.SubtractLeftCopy(
            commands, BufferView<uint>{}, BufferView<int>{}, BufferView<int>{}, 0u);
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit();

        std::array<int, 4u> result{};
        stream << output_buffer.copy_to(luisa::span{result}) << synchronize();
        for(uint i = 0u; i < item_count; ++i)
        {
            expect(result[i] == expected[i]);
        }

        int         singleton        = 42;
        auto        singleton_buffer = device.create_buffer<int>(1u);
        CommandList singleton_commands;
        adjacent_difference.SubtractLeft(singleton_commands, temp_storage.view(), singleton_buffer.view(), 1u);
        stream << singleton_buffer.copy_from(luisa::span{&singleton, 1u}) << singleton_commands.commit()
               << singleton_buffer.copy_to(luisa::span{&singleton, 1u}) << synchronize();
        expect(singleton == 42);
    };
}
