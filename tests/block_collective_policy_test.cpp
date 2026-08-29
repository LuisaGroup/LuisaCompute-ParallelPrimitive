// Tests for shared-memory BlockReduce and BlockScan policies.
// Covers scalar, array, aggregate, partial-tile, and prefix callback paths.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/dsl/builtin.h>
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
constexpr uint block_threads    = 32u;
constexpr uint items_per_thread = 2u;
constexpr uint block_items      = block_threads * items_per_thread;
constexpr uint partial_threads  = 19u;
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

    "block_shared_reduce_and_scan_policies_implement_every_public_path"_test = [&]
    {
        auto reductions       = device.create_buffer<uint>(3u);
        auto scalar_exclusive = device.create_buffer<uint>(block_threads);
        auto scalar_inclusive = device.create_buffer<uint>(block_threads);
        auto array_exclusive  = device.create_buffer<uint>(block_items);
        auto array_inclusive  = device.create_buffer<uint>(block_items);
        auto callback_output  = device.create_buffer<uint>(block_threads);
        auto aggregates       = device.create_buffer<uint>(3u);

        luisa::vector<uint> host_reductions(3u);
        luisa::vector<uint> host_scalar_exclusive(block_threads);
        luisa::vector<uint> host_scalar_inclusive(block_threads);
        luisa::vector<uint> host_array_exclusive(block_items);
        luisa::vector<uint> host_array_inclusive(block_items);
        luisa::vector<uint> host_callback(block_threads);
        luisa::vector<uint> host_aggregates(3u);

        Kernel1D kernel = [](BufferVar<uint> reductions_buffer,
                             BufferVar<uint> scalar_exclusive_buffer,
                             BufferVar<uint> scalar_inclusive_buffer,
                             BufferVar<uint> array_exclusive_buffer,
                             BufferVar<uint> array_inclusive_buffer,
                             BufferVar<uint> callback_buffer,
                             BufferVar<uint> aggregates_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid    = thread_id().x;
            UInt scalar = tid + 1u;
            auto sum    = [](const UInt& lhs, const UInt& rhs) noexcept { return lhs + rhs; };

            BlockReduce<uint, block_threads, items_per_thread, 32u, BlockReduceAlgorithm::SHARED_MEMORY> shared_reduce;
            UInt full_reduction    = shared_reduce.Sum(scalar);
            UInt partial_reduction = shared_reduce.Sum(scalar, partial_threads);

            ArrayVar<uint, items_per_thread> items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                items[item] = tid * items_per_thread + item + 1u;
            }
            UInt array_reduction = shared_reduce.Sum(items);
            $if(tid == 0u)
            {
                reductions_buffer.write(0u, full_reduction);
                reductions_buffer.write(1u, partial_reduction);
                reductions_buffer.write(2u, array_reduction);
            };

            UInt scalar_exclusive;
            UInt scalar_inclusive;
            UInt scalar_exclusive_aggregate;
            UInt scalar_inclusive_aggregate;
            BlockScan<uint, block_threads, items_per_thread, 32u, BlockScanAlgorithm::SHARED_MEMORY> shared_scan;
            shared_scan.ExclusiveSum(scalar, scalar_exclusive, scalar_exclusive_aggregate);
            shared_scan.InclusiveSum(scalar, scalar_inclusive, scalar_inclusive_aggregate);
            scalar_exclusive_buffer.write(tid, scalar_exclusive);
            scalar_inclusive_buffer.write(tid, scalar_inclusive);

            ArrayVar<uint, items_per_thread> array_exclusive;
            ArrayVar<uint, items_per_thread> array_inclusive;
            UInt                             array_aggregate;
            shared_scan.ExclusiveSum(items, array_exclusive, array_aggregate);
            shared_scan.InclusiveSum(items, array_inclusive);
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                array_exclusive_buffer.write(index, array_exclusive[item]);
                array_inclusive_buffer.write(index, array_inclusive[item]);
            }

            UInt callback_scan;
            auto prefix = [](const UInt&) noexcept { return UInt(100u); };
            shared_scan.ExclusiveScan(scalar, callback_scan, sum, prefix);
            callback_buffer.write(tid, callback_scan);

            $if(tid == 0u)
            {
                aggregates_buffer.write(0u, scalar_exclusive_aggregate);
                aggregates_buffer.write(1u, scalar_inclusive_aggregate);
                aggregates_buffer.write(2u, array_aggregate);
            };
        };

        auto shader = device.compile(kernel);
        stream << shader(reductions, scalar_exclusive, scalar_inclusive, array_exclusive, array_inclusive, callback_output, aggregates)
                      .dispatch(block_threads)
               << reductions.copy_to(luisa::span{host_reductions})
               << scalar_exclusive.copy_to(luisa::span{host_scalar_exclusive})
               << scalar_inclusive.copy_to(luisa::span{host_scalar_inclusive})
               << array_exclusive.copy_to(luisa::span{host_array_exclusive})
               << array_inclusive.copy_to(luisa::span{host_array_inclusive})
               << callback_output.copy_to(luisa::span{host_callback})
               << aggregates.copy_to(luisa::span{host_aggregates}) << synchronize();

        expect(host_reductions[0] == block_threads * (block_threads + 1u) / 2u);
        expect(host_reductions[1] == partial_threads * (partial_threads + 1u) / 2u);
        expect(host_reductions[2] == block_items * (block_items + 1u) / 2u);
        expect(host_aggregates[0] == host_reductions[0]);
        expect(host_aggregates[1] == host_reductions[0]);
        expect(host_aggregates[2] == host_reductions[2]);

        for(uint tid = 0u; tid < block_threads; ++tid)
        {
            uint exclusive = tid * (tid + 1u) / 2u;
            uint inclusive = (tid + 1u) * (tid + 2u) / 2u;
            expect(host_scalar_exclusive[tid] == exclusive);
            expect(host_scalar_inclusive[tid] == inclusive);
            expect(host_callback[tid] == 100u + exclusive);
        }
        for(uint index = 0u; index < block_items; ++index)
        {
            uint exclusive = index * (index + 1u) / 2u;
            uint inclusive = (index + 1u) * (index + 2u) / 2u;
            expect(host_array_exclusive[index] == exclusive);
            expect(host_array_inclusive[index] == inclusive);
        }
    };

    "block_warp_shuffle_handles_logical_warps"_test = [&]
    {
        constexpr uint logical_block_threads = 64u;
        constexpr uint logical_warp_threads  = 16u;
        constexpr uint logical_valid_threads = 19u;

        auto logical_reduction = device.create_buffer<uint>(2u);
        auto logical_scan      = device.create_buffer<uint>(logical_block_threads);

        Kernel1D logical_kernel = [](BufferVar<uint> reduction, BufferVar<uint> scan) noexcept
        {
            set_block_size(logical_block_threads);
            UInt tid   = thread_id().x;
            UInt value = tid + 1u;

            UInt aggregate =
                BlockReduce<uint, logical_block_threads, 1u, logical_warp_threads, BlockReduceAlgorithm::WARP_SHUFFLE>()
                    .Sum(value);
            UInt partial_aggregate =
                BlockReduce<uint, logical_block_threads, 1u, logical_warp_threads, BlockReduceAlgorithm::WARP_SHUFFLE>()
                    .Sum(value, 19u);
            UInt prefix;
            BlockScan<uint, logical_block_threads, 1u, logical_warp_threads, BlockScanAlgorithm::WARP_SHUFFLE>()
                .InclusiveSum(value, prefix);
            $if(tid == 0u)
            {
                reduction.write(0u, aggregate);
                reduction.write(1u, partial_aggregate);
            };
            scan.write(tid, prefix);
        };

        auto                logical_shader = device.compile(logical_kernel);
        luisa::vector<uint> host_logical_reduction(2u);
        luisa::vector<uint> host_logical_scan(logical_block_threads);

        stream << logical_shader(logical_reduction, logical_scan).dispatch(logical_block_threads)
               << logical_reduction.copy_to(luisa::span{host_logical_reduction})
               << logical_scan.copy_to(luisa::span{host_logical_scan}) << synchronize();

        expect(host_logical_reduction[0] == logical_block_threads * (logical_block_threads + 1u) / 2u);
        expect(host_logical_reduction[1] == logical_valid_threads * (logical_valid_threads + 1u) / 2u);
        for(uint index = 0u; index < logical_block_threads; ++index)
        {
            expect(host_logical_scan[index] == (index + 1u) * (index + 2u) / 2u);
        }
    };
}
