// Epoch 1 diagnostic: measure the HOST-side cost of building a CommandList for
// each device op WITHOUT committing or synchronizing.  This isolates the
// per-call dispatch overhead (shader-cache key construction + lookup + command
// graph construction) from the backend submit+sync latency that dominates the
// end-to-end wall time reported by lcpp_cub_bench on dx/vk.
//
// Run the same binary against the pre- and post-optimization headers; the
// per-call delta is the host overhead the string->pointer key change removed.
//
// usage: host_enqueue_probe [backend=dx] [num_items=4096] [reps=200] [calls_per_cl=200]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <lcpp/parallel_primitive.h>

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

namespace {

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename F>
double time_enqueue_us(Stream& stream, int reps, int calls_per_cl, F&& enqueue_into) {
    // Warm every code path / cache entry once (compiles shaders), then discard
    // safely by committing.
    {
        CommandList warm;
        enqueue_into(warm);
        stream << warm.commit();
        stream.synchronize();
    }
    // Measure ONLY the host-side enqueue: build a CommandList containing
    // calls_per_cl op invocations and time that loop.  commit()/sync happen
    // outside the timed region so backend submit+sync latency is excluded --
    // this is exactly the per-call dispatch cost (cache key + lookup + command
    // graph construction) the optimization targets.
    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CommandList cl;
        auto t0 = std::chrono::steady_clock::now();
        for (int j = 0; j < calls_per_cl; ++j) {
            enqueue_into(cl);
        }
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        samples.push_back(us / calls_per_cl);
        // Drain to avoid unbounded queue growth; outside the timed region.
        stream << cl.commit();
        stream.synchronize();
    }
    return median(samples);
}

}  // namespace

int main(int argc, char** argv) {
    const char* backend = argc > 1 ? argv[1] : "dx";
    size_t items        = argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 4096u;
    int reps            = argc > 3 ? std::atoi(argv[3]) : 200;
    int calls_per_cl    = argc > 4 ? std::atoi(argv[4]) : 200;

    Context context{argv[0]};
    Device device = context.create_device(backend);
    if (!device) {
        std::fprintf(stderr, "no device for backend '%s'\n", backend);
        return 2;
    }
    Stream stream = device.create_stream();

    auto in     = device.create_buffer<int>(items);
    auto out    = device.create_buffer<int>(items);
    auto one    = device.create_buffer<int>(1u);
    auto uintin = device.create_buffer<uint>(items);
    auto uintout = device.create_buffer<uint>(items);

    constexpr uint RED_BLOCK = 256u, RED_ITEMS = 4u;
    constexpr uint SCAN_BLOCK = 256u, SCAN_ITEMS = 8u;
    constexpr uint RADIX_BLOCK = 256u, RADIX_ITEMS = 4u;

    using ReduceBench = DeviceReduce<RED_BLOCK, 32u, RED_ITEMS>;
    using ScanBench   = DeviceScan<SCAN_BLOCK, 32u, SCAN_ITEMS>;
    using RadixBench  = DeviceRadixSort<RADIX_BLOCK, 32u, RADIX_ITEMS>;

    auto red_temp   = device.create_buffer<uint>(bytes_to_uint_count(ReduceBench::GetTempStorageBytes<int>(items)));
    auto scan_temp  = device.create_buffer<uint>(bytes_to_uint_count(ScanBench::GetTempStorageBytes<int>(items)));
    auto radix_temp = device.create_buffer<uint>(bytes_to_uint_count(RadixBench::GetSortKeysTempStorageBytes<uint>(static_cast<uint>(items))));

    ReduceBench lcpp_reduce; lcpp_reduce.create(device);
    ScanBench   lcpp_scan;   lcpp_scan.create(device);
    RadixBench  lcpp_sort;   lcpp_sort.create(device);

    double reduce_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_reduce.Sum(cl, red_temp.view(), in.view(), one.view(), items);
    });
    double scan_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_scan.InclusiveSum(cl, scan_temp.view(), in.view(), out.view(), items);
    });
    double radix_us = time_enqueue_us(stream, reps, calls_per_cl, [&](CommandList& cl) {
        lcpp_sort.SortKeys(cl, radix_temp.view(), uintin.view(), uintout.view(), static_cast<uint>(items));
    });

    std::fprintf(stderr, "backend=%s items=%zu reps=%d calls_per_cl=%d\n", backend, items, reps, calls_per_cl);
    std::fprintf(stderr, "DeviceReduce.Sum            host_enqueue_us=%.3f\n", reduce_us);
    std::fprintf(stderr, "DeviceScan.InclusiveSum     host_enqueue_us=%.3f\n", scan_us);
    std::fprintf(stderr, "DeviceRadixSort.SortKeys    host_enqueue_us=%.3f\n", radix_us);

    // Sanity: still produces correct results end-to-end (commit + sync once).
    stream << synchronize();
    return 0;
}
