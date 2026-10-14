#include <lcpp/parallel_primitive.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>
#include <unordered_map>

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

// Portable defaults: keep per-block shared memory under the 32 KiB floor of
// the HLSL/DX codegen path ((BLOCK*ITEMS + BLOCK*ITEMS/32) * 4B <= 32 KiB).
// Override via -DLCPP_SCAN_BLOCK/-DLCPP_SCAN_ITEMS for backend-specific tuning.
#ifndef LCPP_SCAN_BLOCK
#define LCPP_SCAN_BLOCK 256u
#endif
#ifndef LCPP_SCAN_ITEMS
#define LCPP_SCAN_ITEMS 15u
#endif
#ifndef LCPP_SCAN_PACKED_TILE_STATE
#define LCPP_SCAN_PACKED_TILE_STATE 1
#endif
#ifndef LCPP_REDUCE_BLOCK
#define LCPP_REDUCE_BLOCK 256u
#endif
#ifndef LCPP_REDUCE_ITEMS
#define LCPP_REDUCE_ITEMS 8u
#endif
#ifndef LCPP_REDUCE_WARP_REDUX
#define LCPP_REDUCE_WARP_REDUX false
#endif
#ifndef LCPP_RADIX_BLOCK
#define LCPP_RADIX_BLOCK 256u
#endif
#ifndef LCPP_RADIX_ITEMS
#define LCPP_RADIX_ITEMS 16u
#endif
#ifndef LCPP_RADIX_CUB_POLICY
#define LCPP_RADIX_CUB_POLICY true
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

extern "C" {
int cub_bench_version();
cudaError_t cub_reduce_query(const int *, int *, int, cudaStream_t, size_t *);
cudaError_t cub_reduce_run(void *, size_t, const int *, int *, int, cudaStream_t);
cudaError_t cub_scan_query(const int *, int *, int, cudaStream_t, size_t *);
cudaError_t cub_scan_run(void *, size_t, const int *, int *, int, cudaStream_t);
cudaError_t cub_radix_query(const unsigned int *, unsigned int *, int, int, int, cudaStream_t, size_t *);
cudaError_t cub_radix_run(void *, size_t, const unsigned int *, unsigned int *, int, int, int, cudaStream_t);
}

namespace {

void check_cuda(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

void check_cu(CUresult error, const char *what) {
    if (error != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *text = nullptr;
        cuGetErrorName(error, &name);
        cuGetErrorString(error, &text);
        std::fprintf(stderr, "%s: %s (%s)\n", what, text ? text : "unknown", name ? name : "unknown");
        std::exit(2);
    }
}

template<typename F>
void with_context(CUcontext context, F &&f) {
    CUcontext previous = nullptr;
    check_cu(cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
    const bool pushed = previous != context;
    if (pushed) { check_cu(cuCtxPushCurrent(context), "cuCtxPushCurrent"); }
    f();
    if (pushed) {
        CUcontext popped = nullptr;
        check_cu(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
        if (popped != context) {
            std::fprintf(stderr, "context stack mismatch\n");
            std::exit(2);
        }
    }
}

template<typename T>
T *raw(BufferView<T> view) {
    return reinterpret_cast<T *>(view.native_handle()) + view.offset();
}

struct Stats {
    std::vector<float> values;
    [[nodiscard]] float median() const {
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2u];
    }
    [[nodiscard]] float minimum() const { return *std::min_element(values.begin(), values.end()); }
    [[nodiscard]] float maximum() const { return *std::max_element(values.begin(), values.end()); }
};

void print_stats(std::string_view name, const Stats &lcpp, const Stats &cub, size_t num_items) {
    const auto lcpp_median = lcpp.median();
    const auto cub_median  = cub.median();
    std::printf("%s,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4fx\n",
                name.data(),
                num_items,
                lcpp_median,
                lcpp.minimum(),
                lcpp.maximum(),
                cub_median,
                cub.minimum(),
                cub.maximum(),
                lcpp_median / cub_median);
}

// ---------------------------------------------------------------------------
// Timers for the lcpp side
// ---------------------------------------------------------------------------

struct LcppTimer {
    virtual ~LcppTimer() = default;
    virtual float measure(const char *name, const std::function<void()> &enqueue) = 0;
};

// CUDA backend: events recorded on the LC stream itself; optional graph mode.
struct CudaStreamTimer final : LcppTimer {
    CUstream stream{};
    CUcontext context{};
    CUevent start{};
    CUevent stop{};
    bool graph_mode = std::getenv("LCPP_BENCH_GRAPH") != nullptr;
    int graph_batch = std::getenv("LCPP_BENCH_BATCH") ? std::max(1, std::atoi(std::getenv("LCPP_BENCH_BATCH"))) : 1;
    std::unordered_map<std::string, CUgraphExec> graphs;

    CudaStreamTimer(CUstream s, CUcontext c) : stream{s}, context{c} {
        with_context(context, [&] {
            check_cu(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
            check_cu(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
        });
    }
    ~CudaStreamTimer() override {
        with_context(context, [&] {
            for (auto [name, graph] : graphs) { cuGraphExecDestroy(graph); }
            cuEventDestroy(start);
            cuEventDestroy(stop);
        });
    }
    float measure(const char *name, const std::function<void()> &enqueue) override {
        CUgraphExec graph_exec{};
        if (graph_mode) {
            auto it = graphs.find(name);
            if (it == graphs.end()) {
                with_context(context, [&] {
                    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize(capture)");
                    check_cu(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL), "cuStreamBeginCapture");
                    for (int batch = 0; batch < graph_batch; ++batch) { enqueue(); }
                    CUgraph graph{};
                    check_cu(cuStreamEndCapture(stream, &graph), "cuStreamEndCapture");
                    check_cu(cuGraphInstantiate(&graph_exec, graph, 0), "cuGraphInstantiate");
                    check_cu(cuGraphDestroy(graph), "cuGraphDestroy");
                    check_cu(cuGraphLaunch(graph_exec, stream), "cuGraphLaunch(warmup)");
                    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize(graph warmup)");
                });
                it = graphs.emplace(name, graph_exec).first;
            }
            graph_exec = it->second;
        }
        with_context(context, [&] { check_cu(cuEventRecord(start, stream), "cuEventRecord(start)"); });
        if (graph_mode) {
            with_context(context, [&] { check_cu(cuGraphLaunch(graph_exec, stream), "cuGraphLaunch"); });
        } else {
            enqueue();
        }
        float elapsed_ms = 0.0f;
        with_context(context, [&] {
            check_cu(cuEventRecord(stop, stream), "cuEventRecord(stop)");
            check_cu(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
            check_cu(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime");
        });
        return elapsed_ms / (graph_mode ? graph_batch : 1);
    }
};

// Non-CUDA backends (dx/vk/...): wall-clock timing around enqueue + synchronize.
// A full stream synchronize() on these backends costs ~100+ us, which would
// drown the GPU time of small ops, so we amortize: `batch` enqueues are
// submitted back-to-back and bracketed by a single synchronize pair, and the
// elapsed wall time is divided by the batch (per-op average throughput,
// comparable to how the CUB side is timed with GPU events on its own stream).
// Epoch-0/1 numbers used batch=1 (LCPP_BENCH_BATCH=1 reproduces them).
struct HostSyncTimer final : LcppTimer {
    Stream *stream;
    int batch = [] {
        const char *env = std::getenv("LCPP_BENCH_BATCH");
        return env ? std::max(1, std::atoi(env)) : 10;
    }();
    explicit HostSyncTimer(Stream &s) : stream(&s) {}
    float measure(const char *, const std::function<void()> &enqueue) override {
        stream->synchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < batch; ++i) { enqueue(); }
        stream->synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count() / static_cast<float>(batch);
    }
};

// ---------------------------------------------------------------------------
// Standalone CUB baseline (runtime API only) used when the lcpp backend is
// not CUDA: CUB still runs on the same physical GPU through its own stream.
// ---------------------------------------------------------------------------

struct CubStandalone {
    cudaStream_t stream{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    bool ok = false;

    CubStandalone() {
        ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreate(&start) == cudaSuccess &&
             cudaEventCreate(&stop) == cudaSuccess;
        if (!ok) { std::fprintf(stderr, "failed to create standalone CUDA stream/events for CUB baseline\n"); }
    }
    ~CubStandalone() {
        if (start) { cudaEventDestroy(start); }
        if (stop) { cudaEventDestroy(stop); }
        if (stream) { cudaStreamDestroy(stream); }
    }

    float measure(const std::function<void(cudaStream_t)> &enqueue) {
        cudaEventRecord(start, stream);
        enqueue(stream);
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms;
    }
};

struct CubBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    CubBuffer() = default;
    ~CubBuffer() { reset(); }
    void reset() {
        if (ptr) { cudaFree(ptr); }
        ptr = nullptr;
        bytes = 0;
    }
    void allocate(size_t b) {
        if (b <= bytes) { return; }
        reset();
        if (cudaMalloc(&ptr, b) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(%zu) failed\n", b);
            ptr = nullptr;
            bytes = 0;
            std::exit(2);
        }
        bytes = b;
    }
};

template<typename T>
void cub_upload(CubBuffer &buf, const std::vector<T> &host, cudaStream_t stream) {
    buf.allocate(host.size() * sizeof(T));
    check_cuda(cudaMemcpyAsync(buf.ptr, host.data(), host.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(H2D)");
}

template<typename T>
void cub_download(const CubBuffer &buf, std::vector<T> &host, cudaStream_t stream) {
    host.resize(buf.bytes / sizeof(T));
    check_cuda(cudaMemcpyAsync(host.data(), buf.ptr, buf.bytes, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(D2H)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(download)");
}

bool is_backend_name(std::string_view s) {
    return s == "cuda" || s == "dx" || s == "vk" || s == "metal" || s == "fallback" || s == "cpu";
}

}  // namespace

int main(int argc, char **argv) {
    // usage: lcpp_cub_bench [backend] [num_items<=UINT_MAX] [repetitions>=5]
    // The legacy numeric-only form "lcpp_cub_bench [num_items] [repetitions]"
    // (implicit cuda backend) is still accepted.
    const char *backend_name = "cuda";
    int argi = 1;
    if (argc > argi && is_backend_name(argv[argi])) {
        backend_name = argv[argi];
        ++argi;
    }
    size_t num_items = 1u << 22u;
    int repetitions  = 10;
    if (argc > argi) { num_items = static_cast<size_t>(std::strtoull(argv[argi], nullptr, 10)); }
    if (argc > argi + 1) { repetitions = std::atoi(argv[argi + 1]); }
    if (num_items == 0u || num_items > static_cast<size_t>(std::numeric_limits<uint>::max()) || repetitions < 5) {
        std::fprintf(stderr, "usage: %s [backend] [num_items<=UINT_MAX] [repetitions>=5]\n", argv[0]);
        return 1;
    }

    Context context{argv[0]};
    // GPU name for the report. On the vk backend, backend_device_names() must
    // NOT be called before device creation (volk global dispatch tables) nor
    // while a device is live, so vk simply reports "unknown".
    std::string gpu_name = "unknown";
    if (std::string_view{backend_name} != "vk") {
        const auto gpu_names = context.backend_device_names(backend_name);
        if (!gpu_names.empty()) { gpu_name = gpu_names.front(); }
    }
    Device device = context.create_device(backend_name);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s'\n", backend_name);
        return 2;
    }
    Stream stream = device.create_stream();
    const bool is_cuda = std::string_view{backend_name} == "cuda";

    CUstream raw_stream = nullptr;
    CUcontext context_handle = nullptr;
    if (is_cuda) {
        raw_stream = reinterpret_cast<CUstream>(stream.native_handle());
        context_handle = reinterpret_cast<CUcontext>(device.native_handle());
        CUcontext previous_context = nullptr;
        check_cu(cuCtxGetCurrent(&previous_context), "cuCtxGetCurrent");
        (void)previous_context;
    }

    std::printf("gpu=%s backend=%s cub=%d items=%zu reps=%d\n",
                gpu_name.c_str(),
                device.backend_name().data(), cub_bench_version(), num_items, repetitions);
    std::printf("operation,items,lcpp_median_ms,lcpp_min_ms,lcpp_max_ms,cub_median_ms,cub_min_ms,cub_max_ms,lcpp_over_cub\n");

    // lcpp-side timer
    std::unique_ptr<LcppTimer> lcpp_timer;
    if (is_cuda) {
        lcpp_timer = std::make_unique<CudaStreamTimer>(raw_stream, context_handle);
    } else {
        lcpp_timer = std::make_unique<HostSyncTimer>(stream);
    }
    // CUB baseline: standalone CUDA stream when lcpp runs on another backend.
    CubStandalone cub;
    if (!is_cuda && !cub.ok) { return 2; }

    // =========================== DeviceReduce::Sum ===========================
    {
        auto input       = device.create_buffer<int>(num_items);
        auto lcpp_output = device.create_buffer<int>(1u);
        auto cub_output  = device.create_buffer<int>(1u);
        using ReduceBench = DeviceReduce<LCPP_REDUCE_BLOCK, 32u, LCPP_REDUCE_ITEMS>;
        auto lcpp_temp   = device.create_buffer<uint>(bytes_to_uint_count(ReduceBench::GetTempStorageBytes<int>(num_items)));
        std::vector<int> host_input(num_items);
        int expected_reduce = 0;
        for (size_t i = 0; i < num_items; ++i) {
            host_input[i] = static_cast<int>(i % 13u) - 6;
            expected_reduce += host_input[i];
        }
        stream << input.copy_from(span{host_input}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_query(raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB reduce query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(reduce temp)"); });
        } else {
            cub_upload(cub_in, host_input, cub.stream);
            cub_out.allocate(sizeof(int));
            check_cuda(cub_reduce_query(static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream, &cub_temp_bytes), "CUB reduce query");
            cub_temp.allocate(cub_temp_bytes);
        }

        ReduceBench lcpp_reduce;
        lcpp_reduce.create(device);
        CommandList warmup;
        lcpp_reduce.Sum(warmup, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB reduce warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_reduce_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream), "CUB reduce warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(reduce warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_reduce_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB reduce");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_reduce_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), s), "CUB reduce");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("reduce_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("reduce_lcpp", [&] {
                CommandList commands;
                lcpp_reduce.Sum(commands, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        int lcpp_result = 0;
        int cub_result  = 0;
        stream << lcpp_output.copy_to(span{&lcpp_result, 1u}) << synchronize();
        if (is_cuda) {
            stream << cub_output.copy_to(span{&cub_result, 1u}) << synchronize();
        } else {
            std::vector<int> tmp;
            cub_download(cub_out, tmp, cub.stream);
            cub_result = tmp[0];
        }
        if (lcpp_result != expected_reduce || cub_result != expected_reduce) {
            std::fprintf(stderr, "reduce verification failed: lcpp=%d cub=%d expected=%d\n", lcpp_result, cub_result, expected_reduce);
            return 3;
        }
        print_stats("DeviceReduce.Sum", lcpp_stats, cub_stats, num_items);
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(reduce temp)"); });
        }
    }

    // ========================= DeviceScan::InclusiveSum =========================
    {
        auto input       = device.create_buffer<int>(num_items);
        auto lcpp_output = device.create_buffer<int>(num_items);
        auto cub_output  = device.create_buffer<int>(num_items);
        using ScanBench = DeviceScan<LCPP_SCAN_BLOCK, 32u, LCPP_SCAN_ITEMS>;
        auto lcpp_temp   = device.create_buffer<uint>(bytes_to_uint_count(ScanBench::GetTempStorageBytes<int>(num_items)));
        std::vector<int> host_input(num_items);
        std::vector<int> expected(num_items);
        int running_sum = 0;
        for (size_t i = 0u; i < num_items; ++i) {
            host_input[i] = static_cast<int>(i % 7u) - 3;
            running_sum += host_input[i];
            expected[i] = running_sum;
        }
        stream << input.copy_from(span{host_input}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_query(raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB scan query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(scan temp)"); });
        } else {
            cub_upload(cub_in, host_input, cub.stream);
            cub_out.allocate(num_items * sizeof(int));
            check_cuda(cub_scan_query(static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream, &cub_temp_bytes), "CUB scan query");
            cub_temp.allocate(cub_temp_bytes);
        }

        ScanBench lcpp_scan;
        lcpp_scan.create(device);
        CommandList warmup;
        lcpp_scan.InclusiveSum(warmup, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB scan warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_scan_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), cub.stream), "CUB scan warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(scan warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_scan_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(input.view()), raw(cub_output.view()), static_cast<int>(num_items), reinterpret_cast<cudaStream_t>(raw_stream)), "CUB scan");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_scan_run(cub_temp.ptr, cub_temp_bytes, static_cast<const int *>(cub_in.ptr), static_cast<int *>(cub_out.ptr), static_cast<int>(num_items), s), "CUB scan");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("scan_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("scan_lcpp", [&] {
                CommandList commands;
                lcpp_scan.InclusiveSum(commands, lcpp_temp.view(), input.view(), lcpp_output.view(), num_items);
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        std::vector<int> lcpp_result(num_items);
        std::vector<int> cub_result(num_items);
        stream << lcpp_output.copy_to(span{lcpp_result}) << synchronize();
        if (is_cuda) {
            stream << cub_output.copy_to(span{cub_result}) << synchronize();
        } else {
            cub_download(cub_out, cub_result, cub.stream);
        }
        const bool lcpp_scan_ok = std::equal(lcpp_result.begin(), lcpp_result.end(), expected.begin());
        const bool cub_scan_ok = std::equal(cub_result.begin(), cub_result.end(), expected.begin());
        if (!lcpp_scan_ok || !cub_scan_ok) {
            std::fprintf(stderr, "scan verification failed\n");
            return 3;
        }
        print_stats("DeviceScan.InclusiveSum", lcpp_stats, cub_stats, num_items);
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(scan temp)"); });
        }
    }

    // ========================= DeviceRadixSort::SortKeys =========================
    {
        auto keys_in       = device.create_buffer<uint>(num_items);
        auto keys_out      = device.create_buffer<uint>(num_items);
        auto lcpp_keys_out = device.create_buffer<uint>(num_items);
        using RadixBench = DeviceRadixSort<LCPP_RADIX_BLOCK, 32u, LCPP_RADIX_ITEMS>;
        auto lcpp_temp     = device.create_buffer<uint>(bytes_to_uint_count(RadixBench::GetSortKeysTempStorageBytes<uint>(static_cast<uint>(num_items))));
        std::vector<uint> host_keys(num_items);
        for (size_t i = 0; i < num_items; ++i) { host_keys[i] = static_cast<uint>((i * 2654435761ull) ^ (i >> 7u)); }
        stream << keys_in.copy_from(span{host_keys}) << synchronize();

        size_t cub_temp_bytes = 0u;
        CUdeviceptr cub_temp_ptr = 0u;
        CubBuffer cub_in, cub_out, cub_temp;
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_query(raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream), &cub_temp_bytes), "CUB radix query");
            });
            with_context(context_handle, [&] { check_cu(cuMemAlloc(&cub_temp_ptr, cub_temp_bytes), "cuMemAlloc(radix temp)"); });
        } else {
            cub_upload(cub_in, host_keys, cub.stream);
            cub_out.allocate(num_items * sizeof(uint));
            check_cuda(cub_radix_query(static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, cub.stream, &cub_temp_bytes), "CUB radix query");
            cub_temp.allocate(cub_temp_bytes);
        }

        RadixBench lcpp_sort;
        lcpp_sort.create(device);
        CommandList warmup;
        lcpp_sort.SortKeys(warmup, lcpp_temp.view(), keys_in.view(), lcpp_keys_out.view(), static_cast<uint>(num_items));
        stream << warmup.commit() << synchronize();
        if (is_cuda) {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream)), "CUB radix warmup");
            });
            stream.synchronize();
        } else {
            check_cuda(cub_radix_run(cub_temp.ptr, cub_temp_bytes, static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, cub.stream), "CUB radix warmup");
            check_cuda(cudaStreamSynchronize(cub.stream), "cudaStreamSynchronize(radix warmup)");
        }

        auto cub_once_cuda = [&] {
            with_context(context_handle, [&] {
                check_cuda(cub_radix_run(reinterpret_cast<void *>(cub_temp_ptr), cub_temp_bytes, raw(keys_in.view()), raw(keys_out.view()), static_cast<int>(num_items), 0, 32, reinterpret_cast<cudaStream_t>(raw_stream)), "CUB radix");
            });
        };
        auto cub_once_standalone = [&](cudaStream_t s) {
            check_cuda(cub_radix_run(cub_temp.ptr, cub_temp_bytes, static_cast<const uint *>(cub_in.ptr), static_cast<uint *>(cub_out.ptr), static_cast<int>(num_items), 0, 32, s), "CUB radix");
        };
        auto measure_cub = [&]() -> float {
            if (is_cuda) { return lcpp_timer->measure("radix_cub", cub_once_cuda); }
            return cub.measure(cub_once_standalone);
        };
        auto measure_lcpp = [&]() -> float {
            return lcpp_timer->measure("radix_lcpp", [&] {
                CommandList commands;
                lcpp_sort.SortKeys(commands, lcpp_temp.view(), keys_in.view(), lcpp_keys_out.view(), static_cast<uint>(num_items));
                stream << commands.commit();
            });
        };

        Stats lcpp_stats;
        Stats cub_stats;
        lcpp_stats.values.reserve(repetitions);
        cub_stats.values.reserve(repetitions);
        for (int i = 0; i < repetitions; ++i) {
            if ((i & 1) == 0) {
                lcpp_stats.values.emplace_back(measure_lcpp());
                cub_stats.values.emplace_back(measure_cub());
            } else {
                cub_stats.values.emplace_back(measure_cub());
                lcpp_stats.values.emplace_back(measure_lcpp());
            }
        }
        stream.synchronize();
        std::vector<uint> lcpp_result(num_items);
        std::vector<uint> cub_result(num_items);
        stream << lcpp_keys_out.copy_to(span{lcpp_result}) << synchronize();
        if (is_cuda) {
            stream << keys_out.copy_to(span{cub_result}) << synchronize();
        } else {
            cub_download(cub_out, cub_result, cub.stream);
        }
        auto expected = host_keys;
        std::sort(expected.begin(), expected.end());
        if (!std::is_sorted(lcpp_result.begin(), lcpp_result.end()) || !std::is_sorted(cub_result.begin(), cub_result.end()) ||
            lcpp_result != cub_result || lcpp_result != expected) {
            std::fprintf(stderr, "radix verification failed\n");
            return 3;
        }
        print_stats("DeviceRadixSort.SortKeys", lcpp_stats, cub_stats, num_items);
        if (is_cuda) {
            with_context(context_handle, [&] { check_cu(cuMemFree(cub_temp_ptr), "cuMemFree(radix temp)"); });
        }
    }

    return 0;
}
