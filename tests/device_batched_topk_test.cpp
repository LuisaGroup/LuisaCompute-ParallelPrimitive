// Tests for the portable flat-buffer DeviceBatchedTopK adapter.
// Covers variable segments, empty segments, capped K, and key/value pairs.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <utility>

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

    DeviceBatchedTopK<> batched_topk;
    batched_topk.create(device);

    "device_batched_topk_selects_each_segment_independently"_test = [&]
    {
        constexpr size_t          num_segments = 3u;
        constexpr size_t          k            = 2u;
        const luisa::vector<int>  keys{9, 1, 7, 4, 6, 2, 8, 3, 5};
        const luisa::vector<int>  values{90, 10, 70, 40, 60, 20, 80, 30, 50};
        const luisa::vector<uint> begins{0u, 4u, 4u};
        const luisa::vector<uint> ends{4u, 4u, 9u};
        auto                      keys_input        = device.create_buffer<int>(keys.size());
        auto                      values_input      = device.create_buffer<int>(values.size());
        auto                      begin_offsets     = device.create_buffer<uint>(begins.size());
        auto                      end_offsets       = device.create_buffer<uint>(ends.size());
        auto                      min_keys_output   = device.create_buffer<int>(num_segments * k);
        auto                      max_keys_output   = device.create_buffer<int>(num_segments * k);
        auto                      max_values_output = device.create_buffer<int>(num_segments * k);
        const luisa::vector<int>  sentinel(num_segments * k, -1);

        CommandList commands;
        batched_topk.MinKeys(commands,
                             BufferView<uint>{},
                             keys_input.view(),
                             min_keys_output.view(),
                             begin_offsets.view(),
                             end_offsets.view(),
                             num_segments,
                             k);
        batched_topk.MaxPairs(commands,
                              BufferView<uint>{},
                              keys_input.view(),
                              max_keys_output.view(),
                              values_input.view(),
                              max_values_output.view(),
                              begin_offsets.view(),
                              end_offsets.view(),
                              num_segments,
                              k);

        luisa::vector<int> min_result(num_segments * k);
        luisa::vector<int> max_key_result(num_segments * k);
        luisa::vector<int> max_value_result(num_segments * k);
        stream << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
               << begin_offsets.copy_from(luisa::span{begins}) << end_offsets.copy_from(luisa::span{ends})
               << min_keys_output.copy_from(luisa::span{sentinel})
               << max_keys_output.copy_from(luisa::span{sentinel})
               << max_values_output.copy_from(luisa::span{sentinel}) << commands.commit()
               << min_keys_output.copy_to(luisa::span{min_result})
               << max_keys_output.copy_to(luisa::span{max_key_result})
               << max_values_output.copy_to(luisa::span{max_value_result}) << synchronize();

        std::sort(min_result.begin(), min_result.begin() + k);
        std::sort(min_result.begin() + 2u * k, min_result.end());
        expect(min_result[0] == 1);
        expect(min_result[1] == 4);
        expect(min_result[2] == -1);
        expect(min_result[3] == -1);
        expect(min_result[4] == 2);
        expect(min_result[5] == 3);

        std::array<std::pair<int, int>, k> segment_zero{};
        std::array<std::pair<int, int>, k> segment_two{};
        for(size_t index = 0u; index < k; ++index)
        {
            segment_zero[index] = {max_key_result[index], max_value_result[index]};
            segment_two[index] = {max_key_result[2u * k + index], max_value_result[2u * k + index]};
        }
        std::sort(segment_zero.begin(), segment_zero.end());
        std::sort(segment_two.begin(), segment_two.end());
        expect(segment_zero[0] == std::pair{7, 70});
        expect(segment_zero[1] == std::pair{9, 90});
        expect(max_key_result[2] == -1);
        expect(max_value_result[2] == -1);
        expect(max_key_result[3] == -1);
        expect(max_value_result[3] == -1);
        expect(segment_two[0] == std::pair{6, 60});
        expect(segment_two[1] == std::pair{8, 80});
    };

    "device_batched_topk_caps_k_and_accepts_empty_work"_test = [&]
    {
        constexpr size_t num_segments = 2u;
        constexpr size_t k            = 3u;
        expect(DeviceBatchedTopK<>::GetTempStorageBytes<int>(num_segments, k) == 0u);
        expect(DeviceBatchedTopK<>::GetTempStorageBytes<int, int>(num_segments, k) == 0u);

        const luisa::vector<int>  keys{5, 1};
        const luisa::vector<uint> begins{0u, 0u};
        const luisa::vector<uint> ends{0u, 2u};
        auto                      keys_input    = device.create_buffer<int>(keys.size());
        auto                      begin_offsets = device.create_buffer<uint>(begins.size());
        auto                      end_offsets   = device.create_buffer<uint>(ends.size());
        auto                      output        = device.create_buffer<int>(num_segments * k);
        const luisa::vector<int>  sentinel(num_segments * k, -1);

        CommandList commands;
        batched_topk.MinKeys(commands,
                             BufferView<uint>{},
                             keys_input.view(),
                             output.view(),
                             begin_offsets.view(),
                             end_offsets.view(),
                             num_segments,
                             k);
        batched_topk.MaxKeys(commands,
                             BufferView<uint>{},
                             BufferView<int>{},
                             BufferView<int>{},
                             BufferView<uint>{},
                             BufferView<uint>{},
                             0u,
                             k);

        luisa::vector<int> result(num_segments * k);
        stream << keys_input.copy_from(luisa::span{keys}) << begin_offsets.copy_from(luisa::span{begins})
               << end_offsets.copy_from(luisa::span{ends}) << output.copy_from(luisa::span{sentinel})
               << commands.commit() << output.copy_to(luisa::span{result}) << synchronize();
        expect(result[0] == -1);
        expect(result[1] == -1);
        expect(result[2] == -1);
        std::sort(result.begin() + k, result.end());
        expect(result[3] == -1);
        expect(result[4] == 1);
        expect(result[5] == 5);
    };
}
