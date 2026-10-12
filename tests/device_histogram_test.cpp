// Test for DeviceHistogram.
// Covers even and range bins, active channels, distinct level types, padded
// two-dimensional regions of interest, and overflow-safe integer binning.

#include <lcpp/parallel_primitive.h>

#include <array>
#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include <limits>

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

    DeviceHistogram<> histogram;
    histogram.create(device);
    auto temp_storage = device.create_buffer<uint>(1u);

    "device_histogram_computes_even_and_range_bins"_test = [&]
    {
        luisa::vector<int>  even_samples{-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        luisa::vector<int>  range_samples{-1, 0, 0, 1, 3, 4, 9, 10};
        luisa::vector<int>  levels{0, 1, 4, 10};
        luisa::vector<uint> even_counts(4u, 99u);
        luisa::vector<uint> range_counts(3u, 99u);

        auto even_input   = device.create_buffer<int>(even_samples.size());
        auto range_input  = device.create_buffer<int>(range_samples.size());
        auto range_levels = device.create_buffer<int>(levels.size());
        auto even_output  = device.create_buffer<uint>(even_counts.size());
        auto range_output = device.create_buffer<uint>(range_counts.size());

        CommandList commands;
        histogram.HistogramEven(commands,
                                temp_storage.view(),
                                even_input.view(),
                                even_output.view(),
                                5u,
                                0,
                                8,
                                even_samples.size());
        histogram.HistogramRange(commands,
                                 temp_storage.view(),
                                 range_input.view(),
                                 range_output.view(),
                                 range_levels.view(),
                                 levels.size(),
                                 range_samples.size());
        stream << even_input.copy_from(luisa::span{even_samples})
               << range_input.copy_from(luisa::span{range_samples})
               << range_levels.copy_from(luisa::span{levels})
               << even_output.copy_from(luisa::span{even_counts})
               << range_output.copy_from(luisa::span{range_counts}) << commands.commit()
               << even_output.copy_to(luisa::span{even_counts})
               << range_output.copy_to(luisa::span{range_counts}) << synchronize();

        std::array<uint, 4u> expected_even{2u, 2u, 2u, 2u};
        std::array<uint, 3u> expected_range{2u, 2u, 2u};
        for(uint i = 0u; i < expected_even.size(); ++i)
        {
            expect(even_counts[i] == expected_even[i]);
        }
        for(uint i = 0u; i < expected_range.size(); ++i)
        {
            expect(range_counts[i] == expected_range[i]);
        }
        expect(DeviceHistogram<>::GetTempStorageBytes(even_samples.size(), 5u) == 0u);
    };

    "device_histogram_computes_independent_active_channels"_test = [&]
    {
        constexpr uint num_channels        = 3u;
        constexpr uint num_active_channels = 2u;
        constexpr uint num_samples         = 6u;
        luisa::vector<int> samples{0, 10, 100, 1, 11, 101, 2, 13, 102, 3, 14, 103, 4, 17, 104, 5, 20, 105};

        auto input          = device.create_buffer<int>(samples.size());
        auto even_output_0  = device.create_buffer<uint>(3u);
        auto even_output_1  = device.create_buffer<uint>(2u);
        auto range_output_0 = device.create_buffer<uint>(3u);
        auto range_output_1 = device.create_buffer<uint>(3u);

        luisa::vector<int> levels_0{0, 2, 4, 6};
        luisa::vector<int> levels_1{10, 12, 15, 21};
        auto               range_levels_0 = device.create_buffer<int>(levels_0.size());
        auto               range_levels_1 = device.create_buffer<int>(levels_1.size());

        std::array<BufferView<uint>, num_active_channels> even_outputs{even_output_0.view(),
                                                                       even_output_1.view()};
        std::array<BufferView<uint>, num_active_channels> range_outputs{range_output_0.view(),
                                                                        range_output_1.view()};
        std::array<BufferView<int>, num_active_channels>  range_levels{range_levels_0.view(),
                                                                      range_levels_1.view()};
        std::array<uint, num_active_channels>             even_num_levels{4u, 3u};
        std::array<uint, num_active_channels>             range_num_levels{4u, 4u};
        std::array<int, num_active_channels>              lower_levels{0, 10};
        std::array<int, num_active_channels>              upper_levels{6, 20};

        CommandList commands;
        histogram.MultiHistogramEven<num_channels, num_active_channels>(
            commands, temp_storage.view(), input.view(), even_outputs, even_num_levels, lower_levels, upper_levels, num_samples);
        histogram.MultiHistogramRange<num_channels, num_active_channels>(
            commands, temp_storage.view(), input.view(), range_outputs, range_levels, range_num_levels, num_samples);

        luisa::vector<uint> initial_3(3u, 77u);
        luisa::vector<uint> initial_2(2u, 77u);
        stream << input.copy_from(luisa::span{samples}) << range_levels_0.copy_from(luisa::span{levels_0})
               << range_levels_1.copy_from(luisa::span{levels_1})
               << even_output_0.copy_from(luisa::span{initial_3})
               << even_output_1.copy_from(luisa::span{initial_2})
               << range_output_0.copy_from(luisa::span{initial_3})
               << range_output_1.copy_from(luisa::span{initial_3}) << commands.commit();

        luisa::vector<uint> even_counts_0(3u);
        luisa::vector<uint> even_counts_1(2u);
        luisa::vector<uint> range_counts_0(3u);
        luisa::vector<uint> range_counts_1(3u);
        stream << even_output_0.copy_to(luisa::span{even_counts_0})
               << even_output_1.copy_to(luisa::span{even_counts_1})
               << range_output_0.copy_to(luisa::span{range_counts_0})
               << range_output_1.copy_to(luisa::span{range_counts_1}) << synchronize();

        std::array<uint, 3u> expected_even_0{2u, 2u, 2u};
        std::array<uint, 2u> expected_even_1{4u, 1u};
        std::array<uint, 3u> expected_range_0{2u, 2u, 2u};
        std::array<uint, 3u> expected_range_1{2u, 2u, 2u};
        for(uint i = 0u; i < 3u; ++i)
        {
            expect(even_counts_0[i] == expected_even_0[i]);
            expect(range_counts_0[i] == expected_range_0[i]);
            expect(range_counts_1[i] == expected_range_1[i]);
        }
        for(uint i = 0u; i < 2u; ++i)
        {
            expect(even_counts_1[i] == expected_even_1[i]);
        }
    };

    "device_histogram_honors_padded_2d_regions_and_distinct_level_types"_test = [&]
    {
        constexpr size_t            row_samples        = 4u;
        constexpr size_t            rows               = 3u;
        constexpr size_t            row_stride_samples = 6u;
        luisa::vector<int>          samples{0, 1, 2, 3, 0, 0, 4, 5, 6, 7, 0, 0, 1, 3, 5, 7, 0, 0};
        luisa::vector<luisa::slong> levels{0, 2, 5, 8};
        auto                        input = device.create_buffer<int>(samples.size());
        auto level_buffer                 = device.create_buffer<luisa::slong>(levels.size());
        auto even_output                  = device.create_buffer<uint>(4u);
        auto range_output                 = device.create_buffer<uint>(3u);

        CommandList commands;
        histogram.HistogramEven(commands,
                                temp_storage.view(),
                                input.view(),
                                even_output.view(),
                                5u,
                                luisa::slong{0},
                                luisa::slong{8},
                                row_samples,
                                rows,
                                row_stride_samples * sizeof(int));
        histogram.HistogramRange(commands,
                                 temp_storage.view(),
                                 input.view(),
                                 range_output.view(),
                                 level_buffer.view(),
                                 levels.size(),
                                 row_samples,
                                 rows,
                                 row_stride_samples * sizeof(int));

        luisa::vector<uint> even_counts(4u);
        luisa::vector<uint> range_counts(3u);
        stream << input.copy_from(luisa::span{samples}) << level_buffer.copy_from(luisa::span{levels})
               << commands.commit() << even_output.copy_to(luisa::span{even_counts})
               << range_output.copy_to(luisa::span{range_counts}) << synchronize();

        std::array<uint, 4u> expected_even{3u, 3u, 3u, 3u};
        std::array<uint, 3u> expected_range{3u, 4u, 5u};
        for(uint bin = 0u; bin < expected_even.size(); ++bin)
        {
            expect(even_counts[bin] == expected_even[bin]);
        }
        for(uint bin = 0u; bin < expected_range.size(); ++bin)
        {
            expect(range_counts[bin] == expected_range[bin]);
        }
    };

    "device_multi_histogram_honors_padded_2d_pixel_rows"_test = [&]
    {
        constexpr size_t   num_channels        = 3u;
        constexpr size_t   num_active_channels = 2u;
        constexpr size_t   row_pixels          = 2u;
        constexpr size_t   rows                = 2u;
        constexpr size_t   row_stride_samples  = 8u;
        luisa::vector<int> samples{0, 10, 99, 2, 12, 99, 0, 0, 4, 14, 99, 6, 16, 99, 0, 0};
        auto               input = device.create_buffer<int>(samples.size());

        auto even_output_0  = device.create_buffer<uint>(4u);
        auto even_output_1  = device.create_buffer<uint>(2u);
        auto range_output_0 = device.create_buffer<uint>(2u);
        auto range_output_1 = device.create_buffer<uint>(2u);
        std::array<BufferView<uint>, num_active_channels> even_outputs{even_output_0.view(),
                                                                       even_output_1.view()};
        std::array<BufferView<uint>, num_active_channels> range_outputs{range_output_0.view(),
                                                                        range_output_1.view()};

        luisa::vector<luisa::slong> levels_0{0, 3, 7};
        luisa::vector<luisa::slong> levels_1{10, 13, 17};
        auto level_buffer_0 = device.create_buffer<luisa::slong>(levels_0.size());
        auto level_buffer_1 = device.create_buffer<luisa::slong>(levels_1.size());
        std::array<BufferView<luisa::slong>, num_active_channels> level_buffers{level_buffer_0.view(),
                                                                                level_buffer_1.view()};
        std::array<uint, num_active_channels>         even_num_levels{5u, 3u};
        std::array<uint, num_active_channels>         range_num_levels{3u, 3u};
        std::array<luisa::slong, num_active_channels> lower_levels{0, 10};
        std::array<luisa::slong, num_active_channels> upper_levels{8, 18};

        CommandList commands;
        histogram.MultiHistogramEven<num_channels, num_active_channels>(commands,
                                                                        temp_storage.view(),
                                                                        input.view(),
                                                                        even_outputs,
                                                                        even_num_levels,
                                                                        lower_levels,
                                                                        upper_levels,
                                                                        row_pixels,
                                                                        rows,
                                                                        row_stride_samples * sizeof(int));
        histogram.MultiHistogramRange<num_channels, num_active_channels>(commands,
                                                                         temp_storage.view(),
                                                                         input.view(),
                                                                         range_outputs,
                                                                         level_buffers,
                                                                         range_num_levels,
                                                                         row_pixels,
                                                                         rows,
                                                                         row_stride_samples * sizeof(int));

        luisa::vector<uint> even_counts_0(4u);
        luisa::vector<uint> even_counts_1(2u);
        luisa::vector<uint> range_counts_0(2u);
        luisa::vector<uint> range_counts_1(2u);
        stream << input.copy_from(luisa::span{samples}) << level_buffer_0.copy_from(luisa::span{levels_0})
               << level_buffer_1.copy_from(luisa::span{levels_1}) << commands.commit()
               << even_output_0.copy_to(luisa::span{even_counts_0})
               << even_output_1.copy_to(luisa::span{even_counts_1})
               << range_output_0.copy_to(luisa::span{range_counts_0})
               << range_output_1.copy_to(luisa::span{range_counts_1}) << synchronize();

        std::array<uint, 4u> expected_even_0{1u, 1u, 1u, 1u};
        std::array<uint, 2u> expected_two_bins{2u, 2u};
        for(uint bin = 0u; bin < expected_even_0.size(); ++bin)
        {
            expect(even_counts_0[bin] == expected_even_0[bin]);
        }
        for(uint bin = 0u; bin < expected_two_bins.size(); ++bin)
        {
            expect(even_counts_1[bin] == expected_two_bins[bin]);
            expect(range_counts_0[bin] == expected_two_bins[bin]);
            expect(range_counts_1[bin] == expected_two_bins[bin]);
        }
    };

    "device_histogram_bins_full_width_unsigned_integers_without_overflow"_test = [&]
    {
        constexpr luisa::ulong      maximum        = std::numeric_limits<luisa::ulong>::max();
        constexpr luisa::ulong      quarter        = maximum / 4ull;
        constexpr luisa::ulong      three_quarters = quarter * 3ull;
        luisa::vector<luisa::ulong> samples{0ull,
                                            quarter,
                                            quarter + 1ull,
                                            maximum / 2ull,
                                            maximum / 2ull + 1ull,
                                            three_quarters,
                                            three_quarters + 3ull,
                                            maximum - 1ull,
                                            maximum};
        auto                        input  = device.create_buffer<luisa::ulong>(samples.size());
        auto                        output = device.create_buffer<uint>(4u);

        CommandList commands;
        histogram.HistogramEven(
            commands, temp_storage.view(), input.view(), output.view(), 5u, 0ull, maximum, samples.size());
        luisa::vector<uint> counts(4u);
        stream << input.copy_from(luisa::span{samples}) << commands.commit()
               << output.copy_to(luisa::span{counts}) << synchronize();

        std::array<uint, 4u> expected{2u, 2u, 2u, 2u};
        for(uint bin = 0u; bin < expected.size(); ++bin)
        {
            expect(counts[bin] == expected[bin]);
        }
    };
}
