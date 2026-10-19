// Test for DeviceSegmentReduce.
// This test covers fixed-size and offset segments, empty segments, extrema,
// first-occurrence arg semantics, and relative segment indices.

// /*
//  * @Author: Ligo
//  * @Date: 2025-09-19 16:04:31
//  * @Last Modified by: Ligo
//  * @Last Modified time: 2025-09-22 18:11:54
//  */


#include <cmath>
#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <lcpp/parallel_primitive.h>
#include <numeric>
#include <limits>
#include <random>
#include <vector>
#include <boost/ut.hpp>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;
int main(int argc, char* argv[])
{
    log_level_verbose();

    luisa::string_view ctx_dir;
    luisa::string_view backend_name;
#ifdef _WIN32
    backend_name = "cuda";
#elif __APPLE__
    backend_name = "metal";
#else
    backend_name = "cuda";
#endif

    // Parse command-line arguments: --ctx=./dir --backend=dx
    for(int i = 1; i < argc; ++i)
    {
        luisa::string_view arg(argv[i]);
        if(arg.starts_with("--ctx="))
        {
            ctx_dir = arg.substr(6);
            LUISA_INFO("Use context directory from command-line: {}", ctx_dir);
        }
        else if(arg.starts_with("--backend="))
        {
            backend_name = arg.substr(10);
            LUISA_INFO("Use backend from command-line: {}", backend_name);
        }
    }

    Context     context{ctx_dir.empty() ? argv[0] : ctx_dir};
    Device      device = context.create_device(backend_name);
    Stream      stream = device.create_stream();
    // Epoch B: opt-in vk-cuda interop fast path (env LCPP_INTEROP=1, vk only).
    // The context must outlive every lcpp module that dispatches through it and
    // is destroyed before the vk device it borrows (see interop_dispatch.h).
    auto interop_requested = [] {
        const char* env = std::getenv("LCPP_INTEROP");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    luisa::unique_ptr<InteropContext> interop;
    if (interop_requested && backend_name == "vk") {
        interop = InteropContext::create(context, device);
    }
    CommandList cmdlist;

    DeviceSegmentReduce<> reducer;
    reducer.create(device, &stream, interop.get());
    if (interop_requested && backend_name == "vk" && interop_policy_selects("segment") && !reducer.interop_active()) {
        std::fprintf(stderr, "[FAIL] LCPP_INTEROP selects segment but reducer stayed on the DSL path\n");
        return 3;
    }

    "segment_reduce_ragged_initial_types_and_custom_policies"_test = [&]
    {
        auto check_policy = [&]<typename T, typename Reducer>(T, Reducer& module)
        {
            const luisa::vector<uint> lengths{0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u,
                                               127u, 128u, 255u, 256u, 257u, 1023u,
                                               1024u, 1025u, 4096u, 16384u, 0u};
            const uint count = static_cast<uint>(lengths.size());
            luisa::vector<uint> begins(count), ends(count);
            uint items = 0u;
            for(uint s = 0u; s < count; ++s)
            {
                begins[s] = items;
                items += lengths[s];
                ends[s] = items;
                ++items;
            }
            constexpr uint padding = 64u;
            luisa::vector<T> input(items + 2u * padding, T{});
            for(uint i = 0u; i < items; ++i)
            {
                input[padding + i] = static_cast<T>(static_cast<int>(i % 13u) - 6);
            }
            auto in = device.create_buffer<T>(input.size());
            auto out = device.create_buffer<T>(count + 2u * padding);
            auto begin_buffer = device.create_buffer<uint>(count);
            auto end_buffer = device.create_buffer<uint>(count);
            stream << in.copy_from(luisa::span{input})
                   << begin_buffer.copy_from(luisa::span{begins})
                   << end_buffer.copy_from(luisa::span{ends});
            for(uint mode = 0u; mode < 3u; ++mode)
            {
                luisa::vector<T> result(out.size(), T{111});
                auto expected = result;
                T initial = mode == 0u ? T{} : mode == 1u ? T{7} : T{3};
                for(uint s = 0u; s < count; ++s)
                {
                    T aggregate = initial;
                    for(uint i = begins[s]; i < ends[s]; ++i)
                    {
                        aggregate = mode == 2u ? std::max(aggregate, input[padding + i]) :
                                                static_cast<T>(aggregate + input[padding + i]);
                    }
                    expected[padding + s] = aggregate;
                }
                CommandList commands;
                auto input_view = in.view().subview(padding, items);
                auto output_view = out.view().subview(padding, count);
                if(mode == 0u)
                {
                    module.Sum(commands, BufferView<uint>{}, input_view, output_view,
                               count, begin_buffer.view(), end_buffer.view());
                }
                else if(mode == 1u)
                {
                    module.Reduce(commands, BufferView<uint>{}, input_view, output_view,
                                  count, begin_buffer.view(), end_buffer.view(), SumOp{}, initial);
                }
                else
                {
                    module.Reduce(commands, BufferView<uint>{}, input_view, output_view,
                                  count, begin_buffer.view(), end_buffer.view(), MaxOp{}, initial);
                }
                stream << out.copy_from(luisa::span{result}) << commands.commit()
                       << out.copy_to(luisa::span{result}) << synchronize();
                expect(result == expected);
            }
        };
        check_policy(int{}, reducer);
        check_policy(uint{}, reducer);
        check_policy(float{}, reducer);
        check_policy(luisa::ulong{}, reducer);
        DeviceSegmentReduce<128u, 32u, 8u> smaller;
        DeviceSegmentReduce<512u, 32u, 4u> larger;
        smaller.create(device, &stream);
        larger.create(device, &stream);
        check_policy(int{}, smaller);
        check_policy(int{}, larger);
    };

    "segment_reduce_fixed_full_tiles_and_tails"_test = [&]
    {
        auto check_policy = [&](auto& module)
        {
            for(uint span : {128u, 257u, 4096u})
            {
                constexpr uint count = 13u;
                luisa::vector<int> input(count * span);
                for(uint i = 0u; i < input.size(); ++i)
                {
                    input[i] = static_cast<int>(i % 13u) - 6;
                }
                auto in = device.create_buffer<int>(input.size());
                auto out = device.create_buffer<int>(count);
                luisa::vector<int> expected(count, 7);
                for(uint s = 0u; s < count; ++s)
                {
                    for(uint i = s * span; i < (s + 1u) * span; ++i)
                    {
                        expected[s] += input[i];
                    }
                }
                CommandList commands;
                module.Reduce(commands, BufferView<uint>{}, in.view(), out.view(), count, span, SumOp{}, 7);
                luisa::vector<int> result(count);
                stream << in.copy_from(luisa::span{input}) << commands.commit()
                       << out.copy_to(luisa::span{result}) << synchronize();
                expect(result == expected);
            }
        };
        check_policy(reducer);
        DeviceSegmentReduce<128u, 32u, 8u> smaller;
        DeviceSegmentReduce<512u, 32u, 4u> larger;
        smaller.create(device, &stream);
        larger.create(device, &stream);
        check_policy(smaller);
        check_policy(larger);
    };

    "segment_reduce"_test = [&]
    {
        constexpr int32_t    array_size = 1024;
        luisa::vector<int32> input_data(array_size);
        for(int i = 0; i < array_size; i++)
        {
            input_data[i] = i;
        }
        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);

        constexpr int       items_per_segment = 100;
        auto                in_buffer         = device.create_buffer<int32>(array_size);
        auto                num_segments      = array_size / items_per_segment + 1;
        auto                out_buffer        = device.create_buffer<int32>(num_segments);
        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment] = std::min(i + items_per_segment, array_size);
            }
        }

        auto begin_offsets = device.create_buffer<uint>(num_segments);
        auto end_offsets   = device.create_buffer<uint>(num_segments);

        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();
        stream << begin_offsets.copy_from(luisa::span{begin_offsets_array}) << synchronize();
        stream << end_offsets.copy_from(luisa::span{end_offsets_array}) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceSegmentReduce<>::GetTempStorageBytes<int32>(array_size);
        auto temp_buffer = device.create_buffer<uint>(std::max<size_t>(1, bytes_to_uint_count(temp_bytes)));

        reducer.Sum(cmdlist,
                    temp_buffer.view(),
                    in_buffer.view(),
                    out_buffer.view(),
                    num_segments,
                    begin_offsets.view(),
                    end_offsets.view());
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(luisa::span{result}) << synchronize();
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_sum =
                    std::accumulate(input_data.begin() + i,
                                    input_data.begin() + i + std::min(array_size - i, items_per_segment),
                                    0);
                LUISA_INFO("Segment {}: expected sum = {}, got {}",
                           i / items_per_segment,
                           expected_sum,
                           result[i / items_per_segment]);
                expect(expected_sum == result[i / items_per_segment]);
            }
        }
    };

    "fixed_segment_reduce"_test = [&]
    {
        constexpr int32_t fixed_array       = 1024;
        constexpr int     items_per_segment = 32;
        static_assert(fixed_array % items_per_segment == 0, "fixed_array must be divisible by items_per_segment");
        auto num_segments = fixed_array / items_per_segment;

        auto                in_buffer  = device.create_buffer<int32>(fixed_array);
        auto                out_buffer = device.create_buffer<int32>(num_segments);
        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment]   = i + items_per_segment;
            }
        }

        luisa::vector<int32> input_data(fixed_array);
        for(int i = 0; i < fixed_array; i++)
        {
            input_data[i] = i;
        }
        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();

        // CUB-style: get temp storage size
        size_t temp_bytes = DeviceSegmentReduce<>::GetTempStorageBytes<int32>(fixed_array);
        auto temp_buffer = device.create_buffer<uint>(std::max<size_t>(1, bytes_to_uint_count(temp_bytes)));

        reducer.Sum(cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), num_segments, items_per_segment);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(luisa::span{result}) << synchronize();
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_sum =
                    std::accumulate(input_data.begin() + i, input_data.begin() + i + items_per_segment, 0);
                LUISA_INFO("Segment {}: expected sum = {}, got {}",
                           i / items_per_segment,
                           expected_sum,
                           result[i / items_per_segment]);
                expect(expected_sum == result[i / items_per_segment]);
            }
        }
    };


    "segment_reduce_arg_max"_test = [&]
    {
        constexpr int32_t array_size        = 1024;
        constexpr int     items_per_segment = 100;


        luisa::vector<int32> input_data(array_size);
        for(int i = 0; i < array_size; i++)
        {
            input_data[i] = i;
        }
        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);

        auto in_buffer    = device.create_buffer<int32>(array_size);
        auto num_segments = ceil_div(array_size, items_per_segment);

        luisa::vector<uint> begin_offsets_array(num_segments);
        luisa::vector<uint> end_offsets_array(num_segments);
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                begin_offsets_array[i / items_per_segment] = i;
                end_offsets_array[i / items_per_segment] = std::min(i + items_per_segment, array_size);
            }
        }

        auto begin_offsets = device.create_buffer<uint>(num_segments);
        auto end_offsets   = device.create_buffer<uint>(num_segments);

        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();
        stream << begin_offsets.copy_from(luisa::span{begin_offsets_array}) << synchronize();
        stream << end_offsets.copy_from(luisa::span{end_offsets_array}) << synchronize();

        auto out_buffer       = device.create_buffer<int>(num_segments);
        auto index_out_buffer = device.create_buffer<uint>(num_segments);

        // CUB-style: get temp storage size for ArgMax
        size_t temp_bytes = DeviceSegmentReduce<>::GetArgTempStorageBytes<int32>(array_size, num_segments);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        reducer.ArgMax(cmdlist,
                       temp_buffer.view(),
                       in_buffer.view(),
                       out_buffer.view(),
                       index_out_buffer.view(),
                       num_segments,
                       begin_offsets.view(),
                       end_offsets.view());
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(luisa::span{result}) << synchronize();
        for(auto i = 0; i < array_size; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_max =
                    std::max_element(input_data.begin() + i,
                                     input_data.begin() + i + std::min(array_size - i, items_per_segment));
                LUISA_INFO("Segment {}: expected max = {}, got {}",
                           i / items_per_segment,
                           *expected_max,
                           result[i / items_per_segment]);
                expect(*expected_max == result[i / items_per_segment]);
            }
        }
    };


    "fixed_segment_reduce_arg_min"_test = [&]
    {
        constexpr int32_t fixed_array       = 1024;
        constexpr int     items_per_segment = 32;
        static_assert(fixed_array % items_per_segment == 0, "fixed_array must be divisible by items_per_segment");
        auto num_segments = fixed_array / items_per_segment;

        auto in_buffer        = device.create_buffer<int32>(fixed_array);
        auto out_buffer       = device.create_buffer<int32>(num_segments);
        auto index_out_buffer = device.create_buffer<uint>(num_segments);

        luisa::vector<int32> input_data(fixed_array);
        for(int i = 0; i < fixed_array; i++)
        {
            input_data[i] = i;
        }

        std::mt19937 rng(114521);  // 固定种子
        std::shuffle(input_data.begin(), input_data.end(), rng);
        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();

        // CUB-style: get temp storage size for ArgMin
        size_t temp_bytes = DeviceSegmentReduce<>::GetArgTempStorageBytes<int32>(fixed_array, num_segments);
        auto temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        reducer.ArgMin(
            cmdlist, temp_buffer.view(), in_buffer.view(), out_buffer.view(), index_out_buffer.view(), num_segments, items_per_segment);
        stream << cmdlist.commit() << synchronize();

        luisa::vector<int32> result(num_segments);
        stream << out_buffer.copy_to(luisa::span{result}) << synchronize();
        for(auto i = 0; i < fixed_array; i++)
        {
            if(i % items_per_segment == 0)
            {
                auto expected_min =
                    std::min_element(input_data.begin() + i, input_data.begin() + i + items_per_segment);
                LUISA_INFO("Segment {}: expected arg min = {}, got {}",
                           i / items_per_segment,
                           *expected_min,
                           result[i / items_per_segment]);
                expect(*expected_min == result[i / items_per_segment]);
            }
        }
    };

    "segment_reduce_empty_segments_use_initial_value"_test = [&]
    {
        luisa::vector<int32_t> input{4, 6, 8, 10};
        luisa::vector<uint>    begin_offsets{0u, 2u, 2u};
        luisa::vector<uint>    end_offsets{2u, 2u, 4u};
        constexpr uint         num_segments = 3u;

        auto input_buffer  = device.create_buffer<int32_t>(input.size());
        auto begin_buffer  = device.create_buffer<uint>(num_segments);
        auto end_buffer    = device.create_buffer<uint>(num_segments);
        auto output_buffer = device.create_buffer<int32_t>(num_segments);
        auto temp_buffer   = device.create_buffer<uint>(1u);

        CommandList commands;
        reducer.Sum(commands,
                    temp_buffer.view(),
                    input_buffer.view(),
                    output_buffer.view(),
                    num_segments,
                    begin_buffer.view(),
                    end_buffer.view());

        luisa::vector<int32_t> output(num_segments);
        stream << input_buffer.copy_from(luisa::span{input})
               << begin_buffer.copy_from(luisa::span{begin_offsets})
               << end_buffer.copy_from(luisa::span{end_offsets}) << commands.commit()
               << output_buffer.copy_to(luisa::span{output}) << synchronize();

        expect(output[0] == 10);
        expect(output[1] == 0);
        expect(output[2] == 18);
    };

    "segment_argmax_handles_negative_ties_and_empty_segments"_test = [&]
    {
        luisa::vector<float> input{-9.0f, -2.0f, -2.0f, -8.0f, -5.0f};
        luisa::vector<uint>  begin_offsets{0u, 1u, 4u};
        luisa::vector<uint>  end_offsets{0u, 4u, 4u};
        constexpr uint       num_segments = 3u;

        auto input_buffer = device.create_buffer<float>(input.size());
        auto begin_buffer = device.create_buffer<uint>(num_segments);
        auto end_buffer   = device.create_buffer<uint>(num_segments);
        auto value_buffer = device.create_buffer<float>(num_segments);
        auto index_buffer = device.create_buffer<uint>(num_segments);
        auto temp_buffer  = device.create_buffer<uint>(bytes_to_uint_count(
            DeviceSegmentReduce<>::GetArgTempStorageBytes<float>(input.size(), num_segments)));

        CommandList commands;
        reducer.ArgMax(commands,
                       temp_buffer.view(),
                       input_buffer.view(),
                       value_buffer.view(),
                       index_buffer.view(),
                       num_segments,
                       begin_buffer.view(),
                       end_buffer.view());

        luisa::vector<float> values(num_segments);
        luisa::vector<uint>  indices(num_segments);
        stream << input_buffer.copy_from(luisa::span{input})
               << begin_buffer.copy_from(luisa::span{begin_offsets})
               << end_buffer.copy_from(luisa::span{end_offsets}) << commands.commit()
               << value_buffer.copy_to(luisa::span{values})
               << index_buffer.copy_to(luisa::span{indices}) << synchronize();

        expect(values[0] == std::numeric_limits<float>::lowest());
        expect(indices[0] == 1u);
        expect(std::abs(values[1] - (-2.0f)) < 1e-6f);
        expect(indices[1] == 0u);
        expect(values[2] == std::numeric_limits<float>::lowest());
        expect(indices[2] == 1u);
    };

    "segment_argmin_offset_overload_returns_first_relative_index"_test = [&]
    {
        luisa::vector<int32_t> input{7, 1, 1, 9, 4, 2};
        luisa::vector<uint>    begin_offsets{0u, 3u};
        luisa::vector<uint>    end_offsets{3u, 6u};
        constexpr uint         num_segments = 2u;

        auto input_buffer = device.create_buffer<int32_t>(input.size());
        auto begin_buffer = device.create_buffer<uint>(num_segments);
        auto end_buffer   = device.create_buffer<uint>(num_segments);
        auto value_buffer = device.create_buffer<int32_t>(num_segments);
        auto index_buffer = device.create_buffer<uint>(num_segments);
        auto temp_buffer  = device.create_buffer<uint>(bytes_to_uint_count(
            DeviceSegmentReduce<>::GetArgTempStorageBytes<int32_t>(input.size(), num_segments)));

        CommandList commands;
        reducer.ArgMin(commands,
                       temp_buffer.view(),
                       input_buffer.view(),
                       value_buffer.view(),
                       index_buffer.view(),
                       num_segments,
                       begin_buffer.view(),
                       end_buffer.view());

        luisa::vector<int32_t> values(num_segments);
        luisa::vector<uint>    indices(num_segments);
        stream << input_buffer.copy_from(luisa::span{input})
               << begin_buffer.copy_from(luisa::span{begin_offsets})
               << end_buffer.copy_from(luisa::span{end_offsets}) << commands.commit()
               << value_buffer.copy_to(luisa::span{values})
               << index_buffer.copy_to(luisa::span{indices}) << synchronize();

        expect(values[0] == 1);
        expect(indices[0] == 1u);
        expect(values[1] == 2);
        expect(indices[1] == 2u);
    };
}
