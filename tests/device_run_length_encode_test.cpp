// Tests for DeviceRunLengthEncode.
// Covers full encoding, non-trivial runs, and empty input.

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

    DeviceRunLengthEncode<> run_length_encode;
    run_length_encode.create(device);

    "device_run_length_encode_matches_cub_examples"_test = [&]
    {
        std::array<int, 8u> input{0, 2, 2, 9, 5, 5, 5, 8};
        constexpr uint      item_count               = static_cast<uint>(input.size());
        auto                input_buffer             = device.create_buffer<int>(item_count);
        auto                unique_output            = device.create_buffer<int>(item_count);
        auto                counts_output            = device.create_buffer<uint>(item_count);
        auto                run_count_output         = device.create_buffer<uint>(1u);
        auto                offsets_output           = device.create_buffer<uint>(item_count);
        auto                lengths_output           = device.create_buffer<uint>(item_count);
        auto                non_trivial_count_output = device.create_buffer<uint>(1u);
        auto                temp_bytes = DeviceRunLengthEncode<>::GetTempStorageBytes(item_count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        run_length_encode.Encode(commands,
                                 temp_storage.view(),
                                 input_buffer.view(),
                                 unique_output.view(),
                                 counts_output.view(),
                                 run_count_output.view(),
                                 item_count);
        run_length_encode.NonTrivialRuns(commands,
                                         temp_storage.view(),
                                         input_buffer.view(),
                                         offsets_output.view(),
                                         lengths_output.view(),
                                         non_trivial_count_output.view(),
                                         item_count);
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit();

        std::array<int, 8u>  unique_result{};
        std::array<uint, 8u> counts_result{};
        std::array<uint, 8u> offsets_result{};
        std::array<uint, 8u> lengths_result{};
        uint                 run_count         = 0u;
        uint                 non_trivial_count = 0u;
        stream << unique_output.copy_to(luisa::span{unique_result})
               << counts_output.copy_to(luisa::span{counts_result})
               << offsets_output.copy_to(luisa::span{offsets_result})
               << lengths_output.copy_to(luisa::span{lengths_result})
               << run_count_output.copy_to(luisa::span{&run_count, 1u})
               << non_trivial_count_output.copy_to(luisa::span{&non_trivial_count, 1u}) << synchronize();

        std::array<int, 5u>  expected_unique{0, 2, 9, 5, 8};
        std::array<uint, 5u> expected_counts{1u, 2u, 1u, 3u, 1u};
        std::array<uint, 2u> expected_offsets{1u, 4u};
        std::array<uint, 2u> expected_lengths{2u, 3u};
        expect(run_count == expected_unique.size());
        expect(non_trivial_count == expected_offsets.size());
        for(uint i = 0u; i < run_count; ++i)
        {
            expect(unique_result[i] == expected_unique[i]);
            expect(counts_result[i] == expected_counts[i]);
        }
        for(uint i = 0u; i < non_trivial_count; ++i)
        {
            expect(offsets_result[i] == expected_offsets[i]);
            expect(lengths_result[i] == expected_lengths[i]);
        }
    };

    "device_run_length_encode_writes_zero_counts_for_empty_input"_test = [&]
    {
        auto run_count_output         = device.create_buffer<uint>(1u);
        auto non_trivial_count_output = device.create_buffer<uint>(1u);
        uint sentinel                 = 99u;
        stream << run_count_output.copy_from(luisa::span{&sentinel, 1u})
               << non_trivial_count_output.copy_from(luisa::span{&sentinel, 1u});

        CommandList commands;
        run_length_encode.Encode(commands,
                                 BufferView<uint>{},
                                 BufferView<int>{},
                                 BufferView<int>{},
                                 BufferView<uint>{},
                                 run_count_output.view(),
                                 0u);
        run_length_encode.NonTrivialRuns(commands,
                                         BufferView<uint>{},
                                         BufferView<int>{},
                                         BufferView<uint>{},
                                         BufferView<uint>{},
                                         non_trivial_count_output.view(),
                                         0u);
        stream << commands.commit();

        uint run_count         = 99u;
        uint non_trivial_count = 99u;
        stream << run_count_output.copy_to(luisa::span{&run_count, 1u})
               << non_trivial_count_output.copy_to(luisa::span{&non_trivial_count, 1u}) << synchronize();
        expect(run_count == 0u);
        expect(non_trivial_count == 0u);
    };

    "device_run_length_encode_matches_host_reference_across_blocks"_test = [&]
    {
        luisa::vector<int>  input;
        luisa::vector<int>  expected_unique;
        luisa::vector<uint> expected_counts;
        luisa::vector<uint> expected_offsets;
        luisa::vector<uint> expected_lengths;
        for(uint run = 0u; run < 180u; ++run)
        {
            uint length = run % 7u + 1u;
            expected_unique.emplace_back(static_cast<int>(run));
            expected_counts.emplace_back(length);
            if(length > 1u)
            {
                expected_offsets.emplace_back(static_cast<uint>(input.size()));
                expected_lengths.emplace_back(length);
            }
            for(uint item = 0u; item < length; ++item)
            {
                input.emplace_back(static_cast<int>(run));
            }
        }

        auto item_count               = static_cast<uint>(input.size());
        auto input_buffer             = device.create_buffer<int>(item_count);
        auto unique_output            = device.create_buffer<int>(item_count);
        auto counts_output            = device.create_buffer<uint>(item_count);
        auto run_count_output         = device.create_buffer<uint>(1u);
        auto offsets_output           = device.create_buffer<uint>(item_count);
        auto lengths_output           = device.create_buffer<uint>(item_count);
        auto non_trivial_count_output = device.create_buffer<uint>(1u);
        auto temp_bytes               = DeviceRunLengthEncode<>::GetTempStorageBytes(item_count);
        auto temp_storage             = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        run_length_encode.Encode(commands,
                                 temp_storage.view(),
                                 input_buffer.view(),
                                 unique_output.view(),
                                 counts_output.view(),
                                 run_count_output.view(),
                                 item_count);
        run_length_encode.NonTrivialRuns(commands,
                                         temp_storage.view(),
                                         input_buffer.view(),
                                         offsets_output.view(),
                                         lengths_output.view(),
                                         non_trivial_count_output.view(),
                                         item_count);
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit();

        luisa::vector<int>  unique_result(item_count);
        luisa::vector<uint> counts_result(item_count);
        luisa::vector<uint> offsets_result(item_count);
        luisa::vector<uint> lengths_result(item_count);
        uint                run_count         = 0u;
        uint                non_trivial_count = 0u;
        stream << unique_output.copy_to(luisa::span{unique_result})
               << counts_output.copy_to(luisa::span{counts_result})
               << offsets_output.copy_to(luisa::span{offsets_result})
               << lengths_output.copy_to(luisa::span{lengths_result})
               << run_count_output.copy_to(luisa::span{&run_count, 1u})
               << non_trivial_count_output.copy_to(luisa::span{&non_trivial_count, 1u}) << synchronize();

        expect(run_count == expected_unique.size());
        expect(non_trivial_count == expected_offsets.size());
        for(uint i = 0u; i < run_count; ++i)
        {
            expect(unique_result[i] == expected_unique[i]);
            expect(counts_result[i] == expected_counts[i]);
        }
        for(uint i = 0u; i < non_trivial_count; ++i)
        {
            expect(offsets_result[i] == expected_offsets[i]);
            expect(lengths_result[i] == expected_lengths[i]);
        }
    };
}
