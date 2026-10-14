#include "cub_misc.h"

#include <lcpp/parallel_primitive.h>
#include <cuda.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

namespace {
[[noreturn]] void fail(const std::string &message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    std::exit(2);
}
void check(CUresult error, const char *operation) {
    if (error != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(error, &message);
        fail(std::string{operation} + ": " + (message ? message : "CUDA driver error"));
    }
}

struct CudaContext {
    CUcontext handle;
    template <typename F>
    void run(F &&function) const {
        CUcontext previous = nullptr;
        check(cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
        const bool pushed = previous != handle;
        if (pushed) { check(cuCtxPushCurrent(handle), "cuCtxPushCurrent"); }
        function();
        if (pushed) {
            CUcontext popped = nullptr;
            check(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
            if (popped != handle) { fail("CUDA context stack mismatch"); }
        }
    }
};

template <typename T>
T *raw(BufferView<T> view) {
    return reinterpret_cast<T *>(view.native_handle()) + view.offset();
}

struct GpuTimer {
    struct Graph {
        CUgraphExec executable = nullptr;
        size_t kernels = 0u;
        size_t nodes = 0u;
    };
    CudaContext context;
    CUstream stream;
    CUevent start = nullptr;
    CUevent stop = nullptr;
    bool graph_mode;
    unsigned batch;
    Graph graphs[2];

    GpuTimer(CudaContext c, CUstream s, bool use_graph, unsigned graph_batch)
        : context{c}, stream{s}, graph_mode{use_graph}, batch{use_graph ? graph_batch : 1u} {
        context.run([&] {
            check(cuEventCreate(&start, CU_EVENT_DEFAULT), "event start");
            check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "event stop");
        });
    }
    ~GpuTimer() {
        context.run([&] {
            for (auto &graph : graphs) {
                if (graph.executable) { cuGraphExecDestroy(graph.executable); }
            }
            cuEventDestroy(start);
            cuEventDestroy(stop);
        });
    }

    void prepare(unsigned side, const std::function<void()> &enqueue) {
        if (!graph_mode) { return; }
        context.run([&] {
            check(cuStreamSynchronize(stream), "capture synchronization");
            check(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL), "capture begin");
            for (unsigned i = 0u; i < batch; ++i) { enqueue(); }
            CUgraph graph = nullptr;
            check(cuStreamEndCapture(stream, &graph), "capture end");
            check(cuGraphGetNodes(graph, nullptr, &graphs[side].nodes), "graph node count");
            std::vector<CUgraphNode> nodes(graphs[side].nodes);
            check(cuGraphGetNodes(graph, nodes.data(), &graphs[side].nodes), "graph nodes");
            std::vector<CUfunction> reported;
            for (auto node : nodes) {
                CUgraphNodeType type;
                check(cuGraphNodeGetType(node, &type), "graph node type");
                graphs[side].kernels += type == CU_GRAPH_NODE_TYPE_KERNEL;
                if (type == CU_GRAPH_NODE_TYPE_KERNEL && std::getenv("LCPP_BENCH_LAUNCH")) {
                    CUDA_KERNEL_NODE_PARAMS parameters{};
                    check(cuGraphKernelNodeGetParams(node, &parameters), "kernel node parameters");
                    if (std::find(reported.begin(), reported.end(), parameters.func) == reported.end()) {
                        reported.push_back(parameters.func);
                        int registers = 0, shared = 0, active_blocks = 0;
                        const char *name = nullptr;
                        check(cuFuncGetName(&name, parameters.func), "kernel name");
                        check(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, parameters.func), "kernel registers");
                        check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, parameters.func), "kernel shared memory");
                        const unsigned threads = parameters.blockDimX * parameters.blockDimY * parameters.blockDimZ;
                        check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, parameters.func,
                                                                         threads, parameters.sharedMemBytes), "kernel occupancy limit");
                        std::fprintf(stderr, "# launch side=%s name=%s block=%u grid=%u registers=%d static_shared_bytes=%d dynamic_shared_bytes=%u max_blocks_per_sm=%d source=CUDA_driver\n",
                                     side == 0u ? "LCPP" : "CUB", name, threads, parameters.gridDimX * parameters.gridDimY * parameters.gridDimZ,
                                     registers, shared, parameters.sharedMemBytes, active_blocks);
                    }
                }
            }
            check(cuGraphInstantiate(&graphs[side].executable, graph, 0), "graph instantiate");
            check(cuGraphDestroy(graph), "graph destroy");
            for (unsigned i = 0u; i < 16u; ++i) {
                check(cuGraphLaunch(graphs[side].executable, stream), "graph warmup");
            }
            check(cuStreamSynchronize(stream), "graph warmup synchronization");
        });
    }

    float measure(unsigned side, const std::function<void()> &enqueue) {
        float milliseconds = 0.0f;
        context.run([&] {
            check(cuEventRecord(start, stream), "event record start");
            if (graph_mode) {
                check(cuGraphLaunch(graphs[side].executable, stream), "graph launch");
            } else {
                enqueue();
            }
            check(cuEventRecord(stop, stream), "event record stop");
            check(cuEventSynchronize(stop), "event synchronize");
            check(cuEventElapsedTime(&milliseconds, start, stop), "event elapsed");
        });
        return milliseconds / static_cast<float>(batch);
    }
};

struct Stats {
    std::vector<float> samples;
    float median() const {
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2u];
    }
    float minimum() const { return *std::min_element(samples.begin(), samples.end()); }
    float maximum() const { return *std::max_element(samples.begin(), samples.end()); }
};

struct CaseData {
    std::vector<int> input;
    std::vector<int> second;
    std::vector<unsigned> flags;
    std::vector<unsigned> offsets;
    std::vector<int> expected;
    std::vector<unsigned> expected_aux;
    std::optional<unsigned> expected_count;
    size_t temp_bytes = 0u;
};

struct Views {
    BufferView<int> input;
    BufferView<int> second;
    BufferView<unsigned> flags;
    BufferView<unsigned> offsets;
    BufferView<int> output;
    BufferView<unsigned> aux;
    BufferView<unsigned> count;
    BufferView<unsigned> temp;
};

struct Benchmark {
    Device &device;
    Stream &stream;
    CudaContext context;
    CUstream native_stream;
    size_t items;
    unsigned repetitions;
    bool graph_mode;
    unsigned batch;
    std::string filter;

    void run(const char *name, CubMiscOp operation, const CaseData &data,
             const std::function<void(CommandList &, const Views &)> &encode) {
        if (!filter.empty() && std::string{name}.find(filter) == std::string::npos) { return; }
        std::fprintf(stderr, "# case=%s starting\n", name);
        const auto input_size = data.input.size();
        const bool full_capacity = operation == CubMiscOp::select || operation == CubMiscOp::unique ||
                                   operation == CubMiscOp::run_length_encode;
        const auto output_size = std::max<size_t>(full_capacity ? items : 1u, data.expected.size());
        const auto aux_size = std::max<size_t>(operation == CubMiscOp::run_length_encode ? items : 1u,
                                               data.expected_aux.size());
        auto input = device.create_buffer<int>(std::max<size_t>(1u, input_size));
        auto second = device.create_buffer<int>(std::max<size_t>(1u, data.second.size()));
        auto flags = device.create_buffer<unsigned>(std::max<size_t>(1u, data.flags.size()));
        auto offsets = device.create_buffer<unsigned>(std::max<size_t>(1u, data.offsets.size()));
        auto lcpp_output = device.create_buffer<int>(output_size);
        auto cub_output = device.create_buffer<int>(output_size);
        auto lcpp_aux = device.create_buffer<unsigned>(aux_size);
        auto cub_aux = device.create_buffer<unsigned>(aux_size);
        auto lcpp_count = device.create_buffer<unsigned>(1u);
        auto cub_count = device.create_buffer<unsigned>(1u);
        auto temp = device.create_buffer<unsigned>(std::max<size_t>(1u, bytes_to_uint_count(data.temp_bytes)));
        // Unused output entries stay observable through a sentinel before warmup.
        std::vector<int> initial_output(output_size, -123456);
        std::vector<unsigned> initial_aux(aux_size, 0xdeadbeefu);
        const unsigned initial_count = 0xdeadbeefu;
        stream << input.copy_from(luisa::span{data.input})
               << lcpp_output.copy_from(luisa::span{initial_output})
               << cub_output.copy_from(luisa::span{initial_output})
               << lcpp_aux.copy_from(luisa::span{initial_aux})
               << cub_aux.copy_from(luisa::span{initial_aux})
               << lcpp_count.copy_from(luisa::span{&initial_count, 1u})
               << cub_count.copy_from(luisa::span{&initial_count, 1u});
        if (!data.second.empty()) { stream << second.copy_from(luisa::span{data.second}); }
        if (!data.flags.empty()) { stream << flags.copy_from(luisa::span{data.flags}); }
        if (!data.offsets.empty()) { stream << offsets.copy_from(luisa::span{data.offsets}); }
        stream.synchronize();

        Views views{input.view(), second.view(), flags.view(), offsets.view(), lcpp_output.view(),
                    lcpp_aux.view(), lcpp_count.view(),
                    temp.view().subview(0u, bytes_to_uint_count(data.temp_bytes))};
        CubMiscArgs args{
            raw(input.view()), raw(second.view()), raw(flags.view()), raw(offsets.view()),
            raw(cub_output.view()), raw(cub_count.view()), raw(cub_aux.view()),
            static_cast<int>(items),
            data.offsets.empty() ? 0 : static_cast<int>(data.offsets.size() - 1u)};
        size_t cub_bytes = 0u;
        auto dispatch_cub = [&](void *storage) {
            context.run([&] {
                const auto error = cub_misc_dispatch(operation, storage, &cub_bytes, &args,
                                                      reinterpret_cast<cudaStream_t>(native_stream));
                if (error != cudaSuccess) { fail(std::string{name} + ": " + cudaGetErrorString(error)); }
            });
        };
        dispatch_cub(nullptr);
        CUdeviceptr cub_temp = 0u;
        context.run([&] { check(cuMemAlloc(&cub_temp, std::max<size_t>(1u, cub_bytes)), "CUB temp allocation"); });
        auto enqueue_lcpp = [&] {
            CommandList commands;
            encode(commands, views);
            stream << commands.commit();
        };
        auto enqueue_cub = [&] { dispatch_cub(reinterpret_cast<void *>(cub_temp)); };

        auto verify = [&] {
            std::vector<int> lcpp_result(output_size), cub_result(output_size);
            std::vector<unsigned> lcpp_extra(initial_aux.size()), cub_extra(initial_aux.size());
            unsigned lcpp_size = initial_count, cub_size = initial_count;
            stream << lcpp_output.copy_to(luisa::span{lcpp_result})
                   << cub_output.copy_to(luisa::span{cub_result})
                   << lcpp_aux.copy_to(luisa::span{lcpp_extra})
                   << cub_aux.copy_to(luisa::span{cub_extra})
                   << lcpp_count.copy_to(luisa::span{&lcpp_size, 1u})
                   << cub_count.copy_to(luisa::span{&cub_size, 1u}) << synchronize();
            if (data.expected_count && (lcpp_size != *data.expected_count || cub_size != *data.expected_count)) {
                fail(std::string{name} + ": count mismatch LCPP=" + std::to_string(lcpp_size) +
                     " CUB=" + std::to_string(cub_size) + " expected=" + std::to_string(*data.expected_count));
            }
            for (size_t i = 0u; i < data.expected.size(); ++i) {
                if (lcpp_result[i] != data.expected[i] || cub_result[i] != data.expected[i]) {
                    fail(std::string{name} + ": output mismatch at " + std::to_string(i));
                }
            }
            for (size_t i = 0u; i < data.expected_aux.size(); ++i) {
                if (lcpp_extra[i] != data.expected_aux[i] || cub_extra[i] != data.expected_aux[i]) {
                    fail(std::string{name} + ": auxiliary output mismatch at " + std::to_string(i));
                }
            }
        };

        enqueue_lcpp();
        enqueue_cub();
        verify();
        {
            // Capture and graph warmup finish before any event sample is recorded.
            GpuTimer timer{context, native_stream, graph_mode, batch};
            timer.prepare(0u, enqueue_lcpp);
            timer.prepare(1u, enqueue_cub);
            Stats lcpp, cub;
            for (unsigned i = 0u; i < repetitions; ++i) {
                if ((i & 1u) == 0u) {
                    lcpp.samples.push_back(timer.measure(0u, enqueue_lcpp));
                    cub.samples.push_back(timer.measure(1u, enqueue_cub));
                } else {
                    cub.samples.push_back(timer.measure(1u, enqueue_cub));
                    lcpp.samples.push_back(timer.measure(0u, enqueue_lcpp));
                }
            }
            verify();
            const auto lcpp_median = lcpp.median();
            const auto cub_median = cub.median();
            std::printf("%s,%zu,%u,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%zu,%zu,%zu,%zu\n",
                        name, items, repetitions, lcpp_median, lcpp.minimum(), lcpp.maximum(),
                        cub_median, cub.minimum(), cub.maximum(), lcpp_median / cub_median,
                        timer.graphs[0].kernels / timer.batch, timer.graphs[1].kernels / timer.batch,
                        data.temp_bytes, cub_bytes);
            std::fflush(stdout);
        }
        context.run([&] { check(cuMemFree(cub_temp), "CUB temp free"); });
        std::fprintf(stderr, "# case=%s CPU/LCPP/CUB verification=pass\n", name);
    }
};

size_t parse_size(const char *text, const char *name) {
    errno = 0;
    char *end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end || text[0] == '-') { fail(std::string{"Invalid "} + name); }
    return static_cast<size_t>(value);
}
}

int main(int argc, char **argv) {
    const size_t n = argc > 1 ? parse_size(argv[1], "item count") : 4194304u;
    const size_t reps = argc > 2 ? parse_size(argv[2], "repetition count") : 60u;
    if (n < 2u || n > static_cast<size_t>(std::numeric_limits<int>::max()) || reps < 5u || reps > 10000u) {
        fail("Usage: cuda_cub_misc [2<=items<=INT_MAX] [5<=repetitions<=10000] [case substring]");
    }
    unsigned batch = 4u;
    if (const auto text = std::getenv("LCPP_BENCH_BATCH")) {
        const auto parsed = parse_size(text, "batch size");
        if (parsed < 1u || parsed > 1024u) { fail("Batch size must be between 1 and 1024."); }
        batch = static_cast<unsigned>(parsed);
    }
    const auto mode = std::getenv("LCPP_BENCH_GRAPH");
    const bool use_graph = !mode || std::string_view{mode} != "0";
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto native_context = reinterpret_cast<CUcontext>(device.native_handle());
    auto native_stream = reinterpret_cast<CUstream>(stream.native_handle());
    CudaContext cuda_context{native_context};
    cuda_context.run([&] {
        CUdevice gpu;
        char name[256]{};
        int driver = 0, major = 0, minor = 0;
        check(cuCtxGetDevice(&gpu), "device");
        check(cuDeviceGetName(name, sizeof(name), gpu), "device name");
        check(cuDriverGetVersion(&driver), "driver version");
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gpu), "capability");
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gpu), "capability");
        std::printf("# gpu=%s capability=%d.%d driver=%d cub=%d items=%zu reps=%zu graph=%u batch=%u\n",
                    name, major, minor, driver, cub_misc_version(), n, reps, use_graph, use_graph ? batch : 1u);
    });
    std::puts("operation,items,reps,lcpp_median_ms,lcpp_min_ms,lcpp_max_ms,cub_median_ms,cub_min_ms,cub_max_ms,lcpp_over_cub,lcpp_kernel_nodes,cub_kernel_nodes,lcpp_temp_bytes,cub_temp_bytes");
    Benchmark bench{device, stream, cuda_context, native_stream, n, static_cast<unsigned>(reps),
                    use_graph, batch, argc > 3 ? argv[3] : ""};

    {
        CaseData data;
        data.input.resize(n);
        data.expected.resize(n);
        for (size_t i = 0u; i < n; ++i) {
            data.input[i] = static_cast<int>(i % 1000u) - 500;
            data.expected[i] = data.input[i] * 3 + 1;
        }
        DeviceTransform<> transform;
        transform.create(device);
        auto function = [](const Int &value) noexcept { return value * 3 + 1; };
        bench.run("DeviceTransform.Transform", CubMiscOp::transform, data, [&](CommandList &cmd, const Views &v) {
            transform.Transform(cmd, BufferView<uint>{}, v.input, v.output, n, function);
        });
        DeviceAdjacentDifference<> difference;
        difference.create(device);
        for (size_t i = 0u; i < n; ++i) {
            data.expected[i] = i == 0u ? data.input[i] : data.input[i] - data.input[i - 1u];
        }
        bench.run("DeviceAdjacentDifference.SubtractLeftCopy", CubMiscOp::adjacent_difference, data,
                  [&](CommandList &cmd, const Views &v) {
                      difference.SubtractLeftCopy(cmd, BufferView<uint>{}, v.input, v.output, n);
                  });
    }
    {
        DeviceFind<> find;
        find.create(device);
        auto predicate = [](const Int &value) noexcept { return value == 777777; };
        for (int scenario = 0; scenario < 3; ++scenario) {
            CaseData data;
            data.input.assign(n, 0);
            data.expected_count = scenario == 0 ? 0u : scenario == 1 ? static_cast<unsigned>(n - 1u) : static_cast<unsigned>(n);
            if (scenario != 2) { data.input[*data.expected_count] = 777777; }
            const char *names[]{"DeviceFind.FindIf.head", "DeviceFind.FindIf.tail", "DeviceFind.FindIf.miss"};
            bench.run(names[scenario], CubMiscOp::find, data, [&](CommandList &cmd, const Views &v) {
                find.FindIf(cmd, BufferView<uint>{}, v.input, v.count, predicate, n);
            });
        }
    }
    {
        CaseData data;
        data.input.resize(n);
        data.flags.resize(n);
        std::iota(data.input.begin(), data.input.end(), 0);
        for (size_t i = 0u; i < n; ++i) {
            if (i % 2u == 0u) { data.expected.push_back(data.input[i]); }
            data.flags[i] = i % 4u == 0u;
        }
        data.expected_count = static_cast<unsigned>(data.expected.size());
        data.temp_bytes = DeviceSelect<>::GetTempStorageBytes(n);
        DeviceSelect<> selector;
        selector.create(device);
        Callable<bool(int)> even = [](Int value) noexcept { return value % 2 == 0; };
        bench.run("DeviceSelect.If.50pct", CubMiscOp::select, data, [&](CommandList &cmd, const Views &v) {
            selector.If(cmd, v.temp, v.input, v.output, v.count, n, even);
        });

        data.expected.clear();
        for (size_t i = 0u; i < n; ++i) {
            if (data.flags[i]) { data.expected.push_back(data.input[i]); }
        }
        data.expected_count = static_cast<unsigned>(data.expected.size());
        for (size_t i = n; i-- > 0u;) {
            if (!data.flags[i]) { data.expected.push_back(data.input[i]); }
        }
        data.temp_bytes = DevicePartition<>::GetTempStorageBytes(n);
        DevicePartition<> partition;
        partition.create(device);
        bench.run("DevicePartition.Flagged.25pct", CubMiscOp::partition, data, [&](CommandList &cmd, const Views &v) {
            partition.Flagged(cmd, v.temp, v.input, v.flags, v.output, v.count, n);
        });
    }
    {
        CaseData data;
        data.input.resize(n / 2u);
        data.second.resize(n - data.input.size());
        const auto key = [](size_t i) { return static_cast<int>((i * 2654435761ull ^ (i >> 7u)) & 0x7fffffffu); };
        for (size_t i = 0u; i < data.input.size(); ++i) { data.input[i] = key(i); }
        for (size_t i = 0u; i < data.second.size(); ++i) { data.second[i] = key(i + data.input.size()); }
        std::sort(data.input.begin(), data.input.end());
        std::sort(data.second.begin(), data.second.end());
        data.expected.resize(n);
        std::merge(data.input.begin(), data.input.end(), data.second.begin(), data.second.end(), data.expected.begin());
        DeviceMerge<> merge;
        merge.create(device);
        bench.run("DeviceMerge.MergeKeys", CubMiscOp::merge, data, [&](CommandList &cmd, const Views &v) {
            merge.MergeKeys(cmd, BufferView<uint>{}, v.input, data.input.size(), v.second, data.second.size(), v.output);
        });
    }
    {
        CaseData data;
        data.input.resize(n);
        data.expected_aux.assign(256u, 0u);
        for (size_t i = 0u; i < n; ++i) {
            data.input[i] = static_cast<int>(((i * 2654435761ull) ^ (i >> 7u)) & 255u);
            ++data.expected_aux[data.input[i]];
        }
        DeviceHistogram<> histogram;
        histogram.create(device);
        bench.run("DeviceHistogram.HistogramEven.256bins", CubMiscOp::histogram, data, [&](CommandList &cmd, const Views &v) {
            histogram.HistogramEven(cmd, BufferView<uint>{}, v.input, v.aux, 257u, 0, 256, n);
        });
    }
    {
        CaseData data;
        data.input.resize(n);
        for (size_t i = 0u; i < n; ++i) { data.input[i] = static_cast<int>(i / 16u); }
        const auto runs = (n + 15u) / 16u;
        data.expected.resize(runs);
        std::iota(data.expected.begin(), data.expected.end(), 0);
        data.expected_count = static_cast<unsigned>(runs);
        data.temp_bytes = DeviceSelect<>::GetTempStorageBytes(n);
        DeviceSelect<> unique;
        unique.create(device);
        bench.run("DeviceSelect.Unique.run16", CubMiscOp::unique, data, [&](CommandList &cmd, const Views &v) {
            unique.Unique(cmd, v.temp, v.input, v.output, v.count, n);
        });
        data.expected_aux.assign(runs, 16u);
        data.expected_aux.back() = static_cast<unsigned>(n - (runs - 1u) * 16u);
        data.temp_bytes = DeviceRunLengthEncode<>::GetTempStorageBytes(n);
        DeviceRunLengthEncode<> rle;
        rle.create(device);
        bench.run("DeviceRunLengthEncode.Encode.run16", CubMiscOp::run_length_encode, data,
                  [&](CommandList &cmd, const Views &v) {
                      rle.Encode(cmd, v.temp, v.input, v.output, v.aux, v.count, n);
                  });
    }
    for (unsigned segment_size : {256u, 4096u}) {
        CaseData data;
        data.input.resize(n);
        for (size_t i = 0u; i < n; ++i) { data.input[i] = static_cast<int>(i % 13u) - 6; }
        const auto segments = (n + segment_size - 1u) / segment_size;
        data.offsets.resize(segments + 1u);
        data.expected.assign(segments, 0);
        for (size_t s = 0u; s <= segments; ++s) {
            data.offsets[s] = static_cast<unsigned>(std::min(n, s * segment_size));
        }
        for (size_t s = 0u; s < segments; ++s) {
            data.expected[s] = std::accumulate(data.input.begin() + data.offsets[s],
                                               data.input.begin() + data.offsets[s + 1u], 0);
        }
        DeviceSegmentReduce<> reduce;
        reduce.create(device);
        const char *name = segment_size == 256u ? "DeviceSegmentReduce.Sum.seg256" : "DeviceSegmentReduce.Sum.seg4096";
        bench.run(name, CubMiscOp::segment_sum, data, [&](CommandList &cmd, const Views &v) {
            reduce.Sum(cmd, BufferView<uint>{}, v.input, v.output, static_cast<unsigned>(segments),
                       v.offsets.subview(0u, segments), v.offsets.subview(1u, segments));
        });
    }
    return 0;
}
