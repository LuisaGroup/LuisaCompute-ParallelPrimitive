// Tests for DeviceCopy.
// Covers batched copies across bindless buffer views and aliased input ranges.

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

    DeviceCopy<> device_copy;
    device_copy.create(device);

    "device_copy_batched_copies_bindless_ranges"_test = [&]
    {
        constexpr uint       range_count = 3u;
        std::array<int, 5u>  source_a{1, 2, 3, 4, 5};
        std::array<int, 4u>  source_b{9, 8, 7, 6};
        std::array<int, 12u> destination;
        destination.fill(-1);
        std::array<uint, range_count> sizes{3u, 4u, 2u};

        auto source_a_buffer    = device.create_buffer<int>(source_a.size());
        auto source_b_buffer    = device.create_buffer<int>(source_b.size());
        auto destination_buffer = device.create_buffer<int>(destination.size());
        auto sizes_buffer       = device.create_buffer<uint>(range_count);
        auto input_ranges       = device.create_bindless_array(range_count);
        auto output_ranges      = device.create_bindless_array(range_count);

        input_ranges.emplace_on_update(0u, source_a_buffer.view().subview(1u, 3u));
        input_ranges.emplace_on_update(1u, source_b_buffer.view());
        input_ranges.emplace_on_update(2u, source_a_buffer.view().subview(0u, 2u));
        output_ranges.emplace_on_update(0u, destination_buffer.view().subview(0u, 3u));
        output_ranges.emplace_on_update(1u, destination_buffer.view().subview(4u, 4u));
        output_ranges.emplace_on_update(2u, destination_buffer.view().subview(9u, 2u));

        CommandList commands;
        device_copy.Batched<int>(
            commands, BufferView<uint>{}, input_ranges, output_ranges, sizes_buffer.view(), range_count);
        stream << source_a_buffer.copy_from(luisa::span{source_a})
               << source_b_buffer.copy_from(luisa::span{source_b})
               << destination_buffer.copy_from(luisa::span{destination})
               << sizes_buffer.copy_from(luisa::span{sizes}) << input_ranges.update()
               << output_ranges.update() << commands.commit();

        std::array<int, 12u> result{};
        stream << destination_buffer.copy_to(luisa::span{result}) << synchronize();
        std::array<int, 12u> expected{2, 3, 4, -1, 9, 8, 7, 6, -1, 1, 2, -1};
        for(uint i = 0u; i < expected.size(); ++i)
        {
            expect(result[i] == expected[i]);
        }
    };

    "device_copy_batched_accepts_empty_batches_and_ranges"_test = [&]
    {
        BindlessArray empty_inputs;
        BindlessArray empty_outputs;

        CommandList commands;
        device_copy.Batched<int>(commands, BufferView<uint>{}, empty_inputs, empty_outputs, BufferView<uint>{}, 0u);
        stream << commands.commit() << synchronize();
        expect(DeviceCopy<>::GetTempStorageBytes(0u) == 0u);
    };

    "device_copy_batched_distributes_long_ranges_across_a_block"_test = [&]
    {
        constexpr uint              item_count = 600u;
        std::array<int, item_count> source{};
        std::array<int, item_count> destination{};
        for(uint i = 0u; i < item_count; ++i)
        {
            source[i]      = static_cast<int>(3u * i + 1u);
            destination[i] = -1;
        }
        auto source_buffer      = device.create_buffer<int>(item_count);
        auto destination_buffer = device.create_buffer<int>(item_count);
        auto sizes_buffer       = device.create_buffer<uint>(1u);
        auto input_ranges       = device.create_bindless_array(1u);
        auto output_ranges      = device.create_bindless_array(1u);
        input_ranges.emplace_on_update(0u, source_buffer.view());
        output_ranges.emplace_on_update(0u, destination_buffer.view());

        CommandList commands;
        device_copy.Batched<int>(
            commands, BufferView<uint>{}, input_ranges, output_ranges, sizes_buffer.view(), 1u);
        stream << source_buffer.copy_from(luisa::span{source})
               << destination_buffer.copy_from(luisa::span{destination})
               << sizes_buffer.copy_from(luisa::span{&item_count, 1u}) << input_ranges.update()
               << output_ranges.update() << commands.commit()
               << destination_buffer.copy_to(luisa::span{destination}) << synchronize();
        for(uint i = 0u; i < item_count; ++i)
        {
            expect(destination[i] == source[i]);
        }
    };

    "device_copy_converts_multidimensional_layouts"_test = [&]
    {
        constexpr std::array<size_t, 2u> extents{2u, 3u};
        std::array<int, 6u>              row_major{0, 1, 2, 10, 11, 12};
        std::array<int, 6u>              column_major{0, 10, 1, 11, 2, 12};
        std::array<int, 6u>              row_to_column{};
        std::array<int, 6u>              column_to_row{};

        auto row_buffer           = device.create_buffer<int>(row_major.size());
        auto column_buffer        = device.create_buffer<int>(column_major.size());
        auto row_to_column_buffer = device.create_buffer<int>(row_to_column.size());
        auto column_to_row_buffer = device.create_buffer<int>(column_to_row.size());

        CommandList commands;
        device_copy.Copy(commands,
                         BufferView<uint>{},
                         row_buffer.view(),
                         row_to_column_buffer.view(),
                         extents,
                         DeviceCopyLayout::ROW_MAJOR,
                         DeviceCopyLayout::COLUMN_MAJOR);
        device_copy.Copy(commands,
                         BufferView<uint>{},
                         column_buffer.view(),
                         column_to_row_buffer.view(),
                         extents,
                         DeviceCopyLayout::COLUMN_MAJOR,
                         DeviceCopyLayout::ROW_MAJOR);
        stream << row_buffer.copy_from(luisa::span{row_major})
               << column_buffer.copy_from(luisa::span{column_major}) << commands.commit()
               << row_to_column_buffer.copy_to(luisa::span{row_to_column})
               << column_to_row_buffer.copy_to(luisa::span{column_to_row}) << synchronize();

        for(size_t i = 0u; i < row_major.size(); ++i)
        {
            expect(row_to_column[i] == column_major[i]);
            expect(column_to_row[i] == row_major[i]);
        }
    };

    "device_copy_supports_rank_three_and_assignable_outputs"_test = [&]
    {
        constexpr std::array<size_t, 3u> extents{2u, 2u, 2u};
        std::array<int, 8u>              input{0, 1, 2, 3, 4, 5, 6, 7};
        std::array<float, 8u>            output{};
        std::array<float, 8u>            expected{0.0f, 4.0f, 2.0f, 6.0f, 1.0f, 5.0f, 3.0f, 7.0f};
        auto                             input_buffer  = device.create_buffer<int>(input.size());
        auto                             output_buffer = device.create_buffer<float>(output.size());

        CommandList commands;
        device_copy.Copy(commands,
                         BufferView<uint>{},
                         input_buffer.view(),
                         output_buffer.view(),
                         extents,
                         DeviceCopyLayout::ROW_MAJOR,
                         DeviceCopyLayout::COLUMN_MAJOR);
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit()
               << output_buffer.copy_to(luisa::span{output}) << synchronize();
        for(size_t i = 0u; i < output.size(); ++i)
        {
            expect(output[i] == expected[i]);
        }
    };

    "device_copy_accepts_zero_extents"_test = [&]
    {
        CommandList commands;
        device_copy.Copy(commands,
                         BufferView<uint>{},
                         BufferView<int>{},
                         BufferView<int>{},
                         std::array<size_t, 2u>{3u, 0u},
                         DeviceCopyLayout::COLUMN_MAJOR,
                         DeviceCopyLayout::ROW_MAJOR);
        stream << commands.commit() << synchronize();
        expect(true);
    };
}
