// Test for shared-memory WarpReduce and WarpScan implementations.
// Covers logical subwarps, valid-item reduction, segmented reduction, scans,
// aggregates, initial values, combined scan, and broadcast.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint block_threads     = 32u;
constexpr uint logical_warp_size = 8u;
constexpr uint logical_warps     = block_threads / logical_warp_size;
}  // namespace

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

    "warp_shared_collectives_cover_every_public_operation_family"_test = [&]
    {
        auto reduced_buffer        = device.create_buffer<int>(logical_warps);
        auto head_buffer           = device.create_buffer<int>(block_threads);
        auto tail_buffer           = device.create_buffer<int>(block_threads);
        auto inclusive_buffer      = device.create_buffer<int>(block_threads);
        auto exclusive_buffer      = device.create_buffer<int>(block_threads);
        auto aggregate_buffer      = device.create_buffer<int>(block_threads);
        auto broadcast_buffer      = device.create_buffer<int>(block_threads);
        auto scan_inclusive_buffer = device.create_buffer<int>(block_threads);
        auto scan_exclusive_buffer = device.create_buffer<int>(block_threads);

        Kernel1D kernel = [](BufferVar<int> reduced_out,
                             BufferVar<int> head_out,
                             BufferVar<int> tail_out,
                             BufferVar<int> inclusive_out,
                             BufferVar<int> exclusive_out,
                             BufferVar<int> aggregate_out,
                             BufferVar<int> broadcast_out,
                             BufferVar<int> scan_inclusive_out,
                             BufferVar<int> scan_exclusive_out) noexcept
        {
            set_block_size(block_threads);
            UInt tid   = thread_id().x;
            UInt lane  = tid % logical_warp_size;
            Int  value = cast<int>(lane + 1u);

            WarpReduce<int, logical_warp_size, WarpReduceAlgorithm::WARP_SHARED_MEMORY> reduce;
            Int reduced   = reduce.Sum(value, 5u);
            Int head_flag = cast<int>(lane == 0u | lane == 3u | lane == 6u);
            Int tail_flag = cast<int>(lane == 2u | lane == 5u | lane == 7u);
            Int head      = reduce.HeadSegmentedSum(value, head_flag);
            Int tail      = reduce.TailSegmentedSum(value, tail_flag);

            WarpScan<int, logical_warp_size, WarpScanAlgorithm::WARP_SHARED_MEMORY> scan;
            Int                                                                     inclusive;
            Int                                                                     exclusive;
            Int                                                                     aggregate;
            Int                                                                     scan_inclusive;
            Int                                                                     scan_exclusive;
            scan.InclusiveScan(value, inclusive, aggregate, SumOp{}, Int{10});
            scan.ExclusiveScan(value, exclusive, SumOp{}, Int{10});
            Int broadcast = scan.Broadcast(value, 3u);
            scan.Scan(value, scan_inclusive, scan_exclusive, SumOp{}, Int{10});

            $if(lane == 0u)
            {
                reduced_out.write(tid / logical_warp_size, reduced);
            };
            head_out.write(tid, head);
            tail_out.write(tid, tail);
            inclusive_out.write(tid, inclusive);
            exclusive_out.write(tid, exclusive);
            aggregate_out.write(tid, aggregate);
            broadcast_out.write(tid, broadcast);
            scan_inclusive_out.write(tid, scan_inclusive);
            scan_exclusive_out.write(tid, scan_exclusive);
        };

        auto               shader = device.compile(kernel);
        luisa::vector<int> reduced(logical_warps);
        luisa::vector<int> head(block_threads);
        luisa::vector<int> tail(block_threads);
        luisa::vector<int> inclusive(block_threads);
        luisa::vector<int> exclusive(block_threads);
        luisa::vector<int> aggregate(block_threads);
        luisa::vector<int> broadcast(block_threads);
        luisa::vector<int> scan_inclusive(block_threads);
        luisa::vector<int> scan_exclusive(block_threads);
        stream << shader(reduced_buffer,
                         head_buffer,
                         tail_buffer,
                         inclusive_buffer,
                         exclusive_buffer,
                         aggregate_buffer,
                         broadcast_buffer,
                         scan_inclusive_buffer,
                         scan_exclusive_buffer)
                      .dispatch(block_threads)
               << reduced_buffer.copy_to(luisa::span{reduced})
               << head_buffer.copy_to(luisa::span{head}) << tail_buffer.copy_to(luisa::span{tail})
               << inclusive_buffer.copy_to(luisa::span{inclusive})
               << exclusive_buffer.copy_to(luisa::span{exclusive})
               << aggregate_buffer.copy_to(luisa::span{aggregate})
               << broadcast_buffer.copy_to(luisa::span{broadcast})
               << scan_inclusive_buffer.copy_to(luisa::span{scan_inclusive})
               << scan_exclusive_buffer.copy_to(luisa::span{scan_exclusive}) << synchronize();

        for(uint logical_warp = 0u; logical_warp < logical_warps; ++logical_warp)
        {
            expect(reduced[logical_warp] == 15);
            for(uint lane = 0u; lane < logical_warp_size; ++lane)
            {
                uint index            = logical_warp * logical_warp_size + lane;
                int  prefix           = static_cast<int>((lane + 1u) * (lane + 2u) / 2u);
                int  exclusive_prefix = prefix - static_cast<int>(lane + 1u);
                uint segment_end      = lane < 3u ? 2u : (lane < 6u ? 5u : 7u);
                int  segment_sum      = 0;
                for(uint source = lane; source <= segment_end; ++source)
                {
                    segment_sum += static_cast<int>(source + 1u);
                }
                expect(head[index] == segment_sum);
                expect(tail[index] == segment_sum);
                expect(inclusive[index] == 10 + prefix);
                expect(exclusive[index] == 10 + exclusive_prefix);
                expect(aggregate[index] == 36);
                expect(broadcast[index] == 4);
                expect(scan_inclusive[index] == 10 + prefix);
                expect(scan_exclusive[index] == 10 + exclusive_prefix);
            }
        }
    };
}
