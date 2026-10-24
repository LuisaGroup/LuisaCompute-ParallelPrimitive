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

    "device_run_length_encode_preserves_wide_values"_test = [&]
    {
        const std::array<luisa::ulong, 7u> input{luisa::ulong{1} << 40u,
                                                 luisa::ulong{1} << 40u,
                                                 luisa::ulong{1} << 41u,
                                                 luisa::ulong{1} << 42u,
                                                 luisa::ulong{1} << 42u,
                                                 luisa::ulong{1} << 42u,
                                                 luisa::ulong{1} << 43u};
        auto                               in = device.create_buffer<luisa::ulong>(input.size());
        auto unique                           = device.create_buffer<luisa::ulong>(input.size());
        auto lengths                          = device.create_buffer<uint>(input.size());
        auto runs                             = device.create_buffer<uint>(1u);
        auto temp                             = device.create_buffer<uint>(
            bytes_to_uint_count(DeviceRunLengthEncode<>::GetTempStorageBytes(input.size())));
        CommandList commands;
        run_length_encode.Encode(
            commands, temp.view(), in.view(), unique.view(), lengths.view(), runs.view(), input.size());
        std::array<luisa::ulong, 7u> values{};
        std::array<uint, 7u>         result_lengths{};
        uint                         count = 0u;
        stream << in.copy_from(luisa::span{input}) << commands.commit()
               << unique.copy_to(luisa::span{values}) << lengths.copy_to(luisa::span{result_lengths})
               << runs.copy_to(luisa::span{&count, 1u}) << synchronize();
        expect(count == 4u);
        const std::array<uint, 4u> expected_lengths{2u, 1u, 3u, 1u};
        for(uint i = 0u; i < 4u; ++i)
        {
            expect(values[i] == luisa::ulong{1} << (40u + i));
            expect(result_lengths[i] == expected_lengths[i]);
        }
    };

    "device_run_length_encode_long_runs_across_fused_tiles"_test = [&]
    {
        for(uint count : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 10u, 12u,
                          31u, 32u, 33u, 127u, 128u, 129u, 479u, 480u, 481u,
                          3839u, 3840u, 3841u, 5759u, 5760u, 5761u, 230413u, 230414u})
        {
            auto in      = device.create_buffer<int>(count);
            auto unique  = device.create_buffer<int>(count);
            auto lengths = device.create_buffer<uint>(count);
            auto runs    = device.create_buffer<uint>(1u);
            auto temp    = device.create_buffer<uint>(
                bytes_to_uint_count(DeviceRunLengthEncode<>::GetTempStorageBytes(count)));
            luisa::vector<uint> poison(temp.size(), 99u);
            for(uint run_length : {1u, 2u, 17u, 18001u, count})
            {
                luisa::vector<int>  input(count);
                luisa::vector<int>  expected;
                luisa::vector<uint> expected_lengths;
                for(uint i = 0u; i < count; ++i)
                {
                    input[i] = static_cast<int>(i / run_length);
                    if(i % run_length == 0u)
                    {
                        expected.emplace_back(input[i]);
                        expected_lengths.emplace_back(0u);
                    }
                    ++expected_lengths.back();
                }
                CommandList commands;
                run_length_encode.Encode(
                    commands, temp.view(), in.view(), unique.view(), lengths.view(), runs.view(), count);
                stream << temp.copy_from(luisa::span{poison}) << in.copy_from(luisa::span{input})
                       << commands.commit();
                luisa::vector<int>  values(count);
                luisa::vector<uint> result_lengths(count);
                uint                run_count = 0u;
                stream << unique.copy_to(luisa::span{values})
                       << lengths.copy_to(luisa::span{result_lengths})
                       << runs.copy_to(luisa::span{&run_count, 1u}) << synchronize();
                expect(run_count == expected.size());
                values.resize(expected.size());
                result_lengths.resize(expected_lengths.size());
                expect(values == expected);
                expect(result_lengths == expected_lengths);
            }
        }
    };

    "device_run_length_encode_typed_subviews_and_custom_policies"_test = [&]
    {
        auto check_policy = [&]<typename T, typename Encoder>(T, Encoder& encoder)
        {
            constexpr uint count = 23054u;
            constexpr uint padding = 64u;
            luisa::vector<T> input(count + 2u * padding, T{99});
            luisa::vector<T> expected(input.size(), T{99});
            luisa::vector<uint> expected_lengths(input.size(), 99u);
            uint expected_count = 0u;
            for(uint i = 0u; i < count; ++i)
            {
                // Mix dense, sparse and headless tiles in the same dispatch.
                const uint phase = (i / 3840u) % 6u;
                const int key = phase == 0u || phase == 4u ? static_cast<int>(i) :
                                phase == 1u || phase == 2u ? -7 :
                                phase == 3u ? static_cast<int>((i / 17u) % 23u) - 11 : -4;
                input[padding + i] = static_cast<T>(key);
                if(i == 0u || input[padding + i] != input[padding + i - 1u])
                {
                    expected[padding + expected_count] = input[padding + i];
                    expected_lengths[padding + expected_count] = 0u;
                    ++expected_count;
                }
                ++expected_lengths[padding + expected_count - 1u];
            }
            auto in = device.create_buffer<T>(input.size());
            auto unique = device.create_buffer<T>(input.size());
            auto lengths = device.create_buffer<uint>(input.size());
            auto runs = device.create_buffer<uint>(1u);
            auto temp = device.create_buffer<uint>(bytes_to_uint_count(Encoder::GetTempStorageBytes(count)));
            luisa::vector<uint> poison(temp.size(), 99u);
            luisa::vector<T> values(input.size(), T{99});
            luisa::vector<uint> result_lengths(input.size(), 99u);
            CommandList commands;
            encoder.Encode(commands, temp.view(), in.view().subview(padding, count),
                           unique.view().subview(padding, count), lengths.view().subview(padding, count),
                           runs.view(), count);
            uint result_count = 0u;
            stream << in.copy_from(luisa::span{input}) << unique.copy_from(luisa::span{values})
                   << lengths.copy_from(luisa::span{result_lengths}) << temp.copy_from(luisa::span{poison})
                   << commands.commit() << unique.copy_to(luisa::span{values})
                   << lengths.copy_to(luisa::span{result_lengths})
                   << runs.copy_to(luisa::span{&result_count, 1u}) << synchronize();
            expect(result_count == expected_count);
            expect(values == expected);
            expect(result_lengths == expected_lengths);
        };
        check_policy(uint{}, run_length_encode);
        check_policy(float{}, run_length_encode);
        DeviceRunLengthEncode<128u, 32u, 8u> smaller;
        DeviceRunLengthEncode<512u, 32u, 4u> larger;
        smaller.create(device);
        larger.create(device);
        check_policy(int{}, smaller);
        check_policy(int{}, larger);
    };

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
