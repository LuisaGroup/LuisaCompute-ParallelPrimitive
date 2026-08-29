
// Tests for warp-level reduce and scan collectives.
// Covers logical warps, segmented reductions, broadcasts, and algorithm policies.

#include "lcpp/block/block_reduce.h"
#include "lcpp/block/block_scan.h"
#include "lcpp/runtime/core.h"
#include "lcpp/warp/warp_reduce.h"
#include "lcpp/warp/warp_scan.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/var.h"
#include "luisa/runtime/shader.h"
#include "luisa/vstl/config.h"
#include <cmath>
#include <cstddef>
#include <lcpp/parallel_primitive.h>
#include <boost/ut.hpp>
#include <numeric>
#include <vector>
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
    CommandList cmdlist;
    Stream      stream = device.create_stream();


    constexpr size_t WARP_SIZE  = 32;
    constexpr size_t array_size = 512;
    constexpr size_t BLOCK_SIZE = 256;

    auto in_buffer         = device.create_buffer<int32>(array_size);
    auto reduce_out_buffer = device.create_buffer<int32>(array_size / WARP_SIZE);

    luisa::vector<int32> input_data(array_size);
    for(int i = 0; i < array_size; i++)
    {
        input_data[i] = i;
    }

    "test_warp_reduce"_test = [&]
    {
        luisa::vector<int32> result(array_size / WARP_SIZE);
        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();

        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_reduce_test_shader = nullptr;
        lazy_compile(device,
                     warp_reduce_test_shader,
                     [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         Int thid        = Int(block_size().x * block_id().x + thread_id().x);
                         Int thread_data = def(0);
                         $if(thid < n)
                         {
                             thread_data = arr_in.read(thid);
                         };
                         Int result = WarpReduce<int>().Reduce(thread_data,
                                                               [](const Var<int>& a, const Var<int>& b) noexcept
                                                               { return a + b; });
                         $if(compute::warp_lane_id() == 0)
                         {
                             arr_out.write(thid / UInt(WARP_SIZE), result);
                         };
                     });

        stream << (*warp_reduce_test_shader)(in_buffer.view(), reduce_out_buffer.view(), array_size).dispatch(array_size);
        stream << reduce_out_buffer.copy_to(luisa::span{result}) << synchronize();  // 输出结果

        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            auto index_result =
                std::accumulate(input_data.begin() + i * WARP_SIZE, input_data.begin() + (i + 1) * WARP_SIZE, 0);
            LUISA_INFO("index: {}, index_result: {}, warp_reduce_result: {}", i, index_result, result[i]);
            expect(result[i] == index_result);
        }
    };


    "test_warp_segment_reduce"_test = [&]
    {
        luisa::vector<int32> input_keys(array_size);
        constexpr int        ITEMS_PER_SEGMENT = 10;
        for(auto i = 0; i < array_size; i += ITEMS_PER_SEGMENT)
        {
            for(auto j = 0; j < ITEMS_PER_SEGMENT - 1 && i + j < array_size; j++)
            {
                input_keys[i + j] = 0;
            }
            if(i + ITEMS_PER_SEGMENT - 1 < array_size)
            {
                input_keys[i + ITEMS_PER_SEGMENT - 1] = 1;
            }
            // input_keys[i] = 1;
            // for(auto j = 1; j < ITEMS_PER_SEGMENT && i + j < array_size; j++)
            // {
            //     input_keys[i + j] = 0;
            // }
        }

        auto seg_result_num = std::ceil(array_size / ITEMS_PER_SEGMENT + array_size / WARP_SIZE
                                        - (array_size * 2) / (ITEMS_PER_SEGMENT * WARP_SIZE));
        LUISA_INFO("Expected segments: {}", seg_result_num);

        auto key_buffer                = device.create_buffer<int32>(array_size);
        auto segment_reduce_out_buffer = device.create_buffer<int32>(array_size);

        stream << key_buffer.copy_from(luisa::span{input_keys}) << synchronize();
        stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();

        luisa::vector<int32> seg_result(array_size);

        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, Buffer<int>, int>> warp_segment_reduce_shader = nullptr;
        lazy_compile(device,
                     warp_segment_reduce_shader,
                     [&](BufferVar<int> arr_in, BufferVar<int> key_in, BufferVar<int> arr_out, Int n) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         Int thid        = Int(block_size().x * block_id().x + thread_id().x);
                         Int thread_data = def(0);
                         $if(thid < n)
                         {
                             thread_data = arr_in.read(thid);
                         };
                         Int key_data = def(0);
                         $if(thid < n)
                         {
                             key_data = key_in.read(thid);
                         };
                         Int result = WarpReduce<int>().TailSegmentedSum(thread_data, key_data);
                         arr_out.write(thid, result);
                     });

        stream << (*warp_segment_reduce_shader)(
                      in_buffer.view(), key_buffer.view(), segment_reduce_out_buffer.view(), array_size)
                      .dispatch(array_size);
        stream << segment_reduce_out_buffer.copy_to(luisa::span{seg_result}) << synchronize();  // 输出结果

        for(auto i = 0; i < array_size;)
        {
            for(int j = 1; j < array_size; ++j)
            {
                auto curr_index = i + j;
                if(curr_index % ITEMS_PER_SEGMENT == 0 || curr_index % WARP_SIZE == 0)
                {
                    auto index = i / ITEMS_PER_SEGMENT + i / WARP_SIZE - (i * 2) / (ITEMS_PER_SEGMENT * WARP_SIZE);
                    auto index_result =
                        std::accumulate(input_data.begin() + i, input_data.begin() + curr_index, 0);
                    LUISA_INFO("segment start index: {}, start index: {}, index: {}, index_result: {}, warp_segment_reduce_result: {}",
                               i,
                               i,
                               index,
                               index_result,
                               seg_result[i]);
                    expect(seg_result[i] == index_result);
                    i += j;
                    break;
                }
            }
        };
    };


    "test_warp_ex_scan"_test = [&]
    {
        auto                 scan_out_buffer = device.create_buffer<int32>(array_size);
        luisa::vector<int32> scan_result(array_size);
        luisa::unique_ptr<Shader<1, Buffer<int>, Buffer<int>, int>> warp_ex_scan_test_shader = nullptr;
        lazy_compile(device,
                     warp_ex_scan_test_shader,
                     [&](BufferVar<int> arr_in, BufferVar<int> arr_out, Int n) noexcept
                     {
                         luisa::compute::set_block_size(BLOCK_SIZE);
                         luisa::compute::set_warp_size(WARP_SIZE);
                         Int thid        = Int(block_size().x * block_id().x + thread_id().x);
                         Int thread_data = def(0);
                         $if(thid < n)
                         {
                             thread_data = arr_in.read(thid);
                         };
                         Int output_block_scan;
                         Int warp_aggregate;
                         output_block_scan = warp_prefix_sum(thread_data);
                         WarpScan<int>().ExclusiveSum(thread_data, output_block_scan, warp_aggregate);
                         $if(thid < n)
                         {
                             arr_out.write(thid, output_block_scan);
                         };
                     });

        stream << (*warp_ex_scan_test_shader)(in_buffer.view(), scan_out_buffer.view(), array_size).dispatch(array_size);
        stream << scan_out_buffer.copy_to(luisa::span{scan_result}) << synchronize();  // 输出结果
        for(auto i = 0; i < array_size / WARP_SIZE; ++i)
        {
            luisa::vector<int> exclusive_scan_result(WARP_SIZE);
            std::exclusive_scan(input_data.begin() + i * WARP_SIZE,
                                input_data.begin() + (i + 1) * WARP_SIZE,
                                exclusive_scan_result.begin(),
                                0);
            for(auto j = 0; j < WARP_SIZE; ++j)
            {
                LUISA_INFO("index: {}, index_result: {}, warp_exclusive_scan_result: {}",
                           i * WARP_SIZE + j,
                           exclusive_scan_result[j],
                           scan_result[i * WARP_SIZE + j]);
                expect(exclusive_scan_result[j] == scan_result[i * WARP_SIZE + j]);
            }
        };
    };

    "warp_scan_broadcasts_from_a_lane_in_each_logical_warp"_test = [&]
    {
        constexpr uint       logical_warp_size = 8u;
        constexpr uint       source_lane       = 3u;
        auto                 broadcast_buffer  = device.create_buffer<int32>(array_size);
        luisa::vector<int32> broadcast_result(array_size);

        Kernel1D kernel = [](BufferVar<int> arr_in, BufferVar<int> arr_out, UInt count) noexcept
        {
            set_block_size(BLOCK_SIZE);
            UInt index = dispatch_id().x;
            $if(index < count)
            {
                Int value     = arr_in.read(index);
                Int broadcast = WarpScan<int, logical_warp_size>().Broadcast(value, 3u);
                arr_out.write(index, broadcast);
            };
        };

        auto shader = device.compile(kernel);
        stream << in_buffer.copy_from(luisa::span{input_data})
               << shader(in_buffer, broadcast_buffer, array_size).dispatch(array_size)
               << broadcast_buffer.copy_to(luisa::span{broadcast_result}) << synchronize();

        for(uint i = 0u; i < array_size; ++i)
        {
            uint logical_warp_begin = (i / logical_warp_size) * logical_warp_size;
            expect(broadcast_result[i] == input_data[logical_warp_begin + source_lane]);
        }
    };

    "warp_inclusive_scan_without_an_initial_value_uses_every_lane"_test = [&]
    {
        auto                 inclusive_buffer = device.create_buffer<int32>(array_size);
        luisa::vector<int32> inclusive_result(array_size);

        Kernel1D kernel = [](BufferVar<int> arr_in, BufferVar<int> arr_out, UInt count) noexcept
        {
            set_block_size(BLOCK_SIZE);
            UInt index = dispatch_id().x;
            $if(index < count)
            {
                Int input = arr_in.read(index);
                Int output;
                WarpScan<int>().InclusiveScan(input, output, SumOp{});
                arr_out.write(index, output);
            };
        };

        auto shader = device.compile(kernel);
        stream << in_buffer.copy_from(luisa::span{input_data})
               << shader(in_buffer, inclusive_buffer, array_size).dispatch(array_size)
               << inclusive_buffer.copy_to(luisa::span{inclusive_result}) << synchronize();

        for(uint warp = 0u; warp < array_size / WARP_SIZE; ++warp)
        {
            int expected = 0;
            for(uint lane = 0u; lane < WARP_SIZE; ++lane)
            {
                uint index = warp * WARP_SIZE + lane;
                expected += input_data[index];
                expect(inclusive_result[index] == expected);
            }
        }
    };

    "logical_warp_collectives_stay_isolated_and_implement_every_algorithm_branch"_test = [&]
    {
        constexpr uint logical_warp_size  = 8u;
        constexpr uint logical_warp_count = array_size / logical_warp_size;

        auto reduce_buffer        = device.create_buffer<int32>(logical_warp_count);
        auto shared_reduce_buffer = device.create_buffer<int32>(logical_warp_count);
        auto inclusive_buffer     = device.create_buffer<int32>(array_size);
        auto exclusive_buffer     = device.create_buffer<int32>(array_size);
        auto aggregate_buffer     = device.create_buffer<int32>(array_size);
        auto shared_scan_buffer   = device.create_buffer<int32>(array_size);
        auto head_segment_buffer  = device.create_buffer<int32>(array_size);
        auto tail_segment_buffer  = device.create_buffer<int32>(array_size);

        luisa::vector<int32> reduce_result(logical_warp_count);
        luisa::vector<int32> shared_reduce_result(logical_warp_count);
        luisa::vector<int32> inclusive_result(array_size);
        luisa::vector<int32> exclusive_result(array_size);
        luisa::vector<int32> aggregate_result(array_size);
        luisa::vector<int32> shared_scan_result(array_size);
        luisa::vector<int32> head_segment_result(array_size);
        luisa::vector<int32> tail_segment_result(array_size);

        Kernel1D kernel = [](BufferVar<int> reduce_out,
                             BufferVar<int> shared_reduce_out,
                             BufferVar<int> inclusive_out,
                             BufferVar<int> exclusive_out,
                             BufferVar<int> aggregate_out,
                             BufferVar<int> shared_scan_out,
                             BufferVar<int> head_segment_out,
                             BufferVar<int> tail_segment_out) noexcept
        {
            set_block_size(BLOCK_SIZE);
            UInt index = dispatch_id().x;
            UInt lane  = warp_lane_id() % 8u;
            Int  value = cast<int>(lane + 1u);

            Int reduced = WarpReduce<int, 8u>().Sum(value, 5u);
            Int shared_reduced = WarpReduce<int, 8u, WarpReduceAlgorithm::WARP_SHARED_MEMORY>().Sum(value);
            $if(lane == 0u)
            {
                reduce_out.write(index / 8u, reduced);
                shared_reduce_out.write(index / 8u, shared_reduced);
            };

            Int inclusive;
            Int exclusive;
            Int aggregate;
            Int shared_inclusive;
            WarpScan<int, 8u>().InclusiveSum(value, inclusive);
            WarpScan<int, 8u>().ExclusiveSum(value, exclusive, aggregate);
            WarpScan<int, 8u, WarpScanAlgorithm::WARP_SHARED_MEMORY>().InclusiveSum(value, shared_inclusive);

            Int head_flag    = cast<int>(lane == 0u | lane == 3u | lane == 6u);
            Int tail_flag    = cast<int>(lane == 2u | lane == 5u | lane == 7u);
            Int head_segment = WarpReduce<int, 8u>().HeadSegmentedSum(value, head_flag);
            Int tail_segment = WarpReduce<int, 8u>().TailSegmentedSum(value, tail_flag);

            inclusive_out.write(index, inclusive);
            exclusive_out.write(index, exclusive);
            aggregate_out.write(index, aggregate);
            shared_scan_out.write(index, shared_inclusive);
            head_segment_out.write(index, head_segment);
            tail_segment_out.write(index, tail_segment);
        };

        auto shader = device.compile(kernel);
        stream << shader(reduce_buffer,
                         shared_reduce_buffer,
                         inclusive_buffer,
                         exclusive_buffer,
                         aggregate_buffer,
                         shared_scan_buffer,
                         head_segment_buffer,
                         tail_segment_buffer)
                      .dispatch(array_size)
               << reduce_buffer.copy_to(luisa::span{reduce_result})
               << shared_reduce_buffer.copy_to(luisa::span{shared_reduce_result})
               << inclusive_buffer.copy_to(luisa::span{inclusive_result})
               << exclusive_buffer.copy_to(luisa::span{exclusive_result})
               << aggregate_buffer.copy_to(luisa::span{aggregate_result})
               << shared_scan_buffer.copy_to(luisa::span{shared_scan_result})
               << head_segment_buffer.copy_to(luisa::span{head_segment_result})
               << tail_segment_buffer.copy_to(luisa::span{tail_segment_result}) << synchronize();

        for(uint logical_warp = 0u; logical_warp < logical_warp_count; ++logical_warp)
        {
            expect(reduce_result[logical_warp] == 15);
            expect(shared_reduce_result[logical_warp] == 36);
            int prefix = 0;
            for(uint lane = 0u; lane < logical_warp_size; ++lane)
            {
                uint index = logical_warp * logical_warp_size + lane;
                expect(exclusive_result[index] == prefix);
                prefix += static_cast<int>(lane + 1u);
                expect(inclusive_result[index] == prefix);
                expect(shared_scan_result[index] == prefix);
                expect(aggregate_result[index] == 36);
            }

            uint base = logical_warp * logical_warp_size;
            expect(head_segment_result[base] == 6);
            expect(head_segment_result[base + 3u] == 15);
            expect(head_segment_result[base + 6u] == 15);
            expect(tail_segment_result[base] == 6);
            expect(tail_segment_result[base + 3u] == 15);
            expect(tail_segment_result[base + 6u] == 15);
        }
    };
};
