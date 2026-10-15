// Boundary, scratch, in-place, and fallback checks for the default scan policies.
// Run the ordinary case on CUDA, Vulkan, and required Vulkan CUDA interop;
// segmented scans use the native DSL implementations on CUDA and Vulkan.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <lcpp/parallel_primitive.h>
#include <lcpp/runtime/interop_dispatch.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

namespace
{
size_t         checks   = 0u;
size_t         failures = 0u;
constexpr uint canary   = 0xa5c3f17du;

void check(bool ok, const std::string& label)
{
    ++checks;
    failures += !ok;
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", label.c_str());
}

template <typename T>
void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& label)
{
    auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if(mismatch.first != actual.end())
    {
        std::fprintf(stderr,
                     "  index=%zu actual=%.17g expected=%.17g\n",
                     static_cast<size_t>(mismatch.first - actual.begin()),
                     static_cast<double>(*mismatch.first),
                     static_cast<double>(*mismatch.second));
    }
    check(mismatch.first == actual.end(), label);
}

template <typename T>
T value(size_t i)
{
    // Exact binary fractions keep parallel floating-point sums exactly comparable.
    if constexpr(std::is_floating_point_v<T>)
    {
        return static_cast<T>(i % 7u + 1u) * T{0.25};
    }
    else
    {
        return static_cast<T>(i % 7u + 1u);
    }
}

template <typename T>
constexpr const char* type_name()
{
    if constexpr(std::is_same_v<T, int>)
        return "int";
    if constexpr(std::is_same_v<T, uint>)
        return "uint";
    if constexpr(std::is_same_v<T, float>)
        return "float";
    return "double";
}

struct GuardedTemp
{
    static constexpr size_t guard_words = 256u;
    size_t                  words;
    Buffer<uint>            backing;

    GuardedTemp(Device& device, Stream& stream, size_t bytes)
        : words{std::max<size_t>(1u, bytes_to_uint_count(bytes))}
        , backing{device.create_buffer<uint>(words + 2u * guard_words)}
    {
        std::vector<uint> initial(backing.size(), canary);
        stream << backing.copy_from(luisa::span{initial}) << synchronize();
    }

    [[nodiscard]] BufferView<uint> view() const { return backing.view(guard_words, words); }

    void verify(Stream& stream, const std::string& label) const
    {
        std::vector<uint> data(backing.size());
        stream << backing.copy_to(luisa::span{data}) << synchronize();
        auto unchanged = [](uint x) { return x == canary; };
        check(std::all_of(data.begin(), data.begin() + guard_words, unchanged)
                  && std::all_of(data.begin() + guard_words + words, data.end(), unchanged),
              label + ".scratch_guards");
    }
};

struct Runner
{
    Device&         device;
    Stream&         stream;
    InteropContext* interop;
    bool            require_interop;

    void active(const DeviceScan<>& scan, const std::string& label)
    {
        if(require_interop)
            check(scan.interop_active(), label + ".interop_active");
    }
};

template <typename T>
void ordinary(Runner& r)
{
    DeviceScan<> scan;
    scan.create(r.device, &r.stream, r.interop);
    const std::string prefix = std::string{"ordinary."} + type_name<T>();
    r.active(scan, prefix + ".create");
    // Cover both portable 256x15 and CUDA 320x25 tile boundaries.
    for(size_t n :
        {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 3839u, 3840u, 3841u, 4095u, 4096u, 4097u, 7999u, 8000u, 8001u, 16003u, 262147u})
    {
        std::vector<T> host(n + 8u, static_cast<T>(-777));
        for(size_t i = 0u; i < n; ++i)
            host[i] = value<T>(i);
        auto input  = r.device.create_buffer<T>(host.size());
        auto output = r.device.create_buffer<T>(host.size());
        r.stream << input.copy_from(luisa::span{host}) << synchronize();
        GuardedTemp temp{r.device, r.stream, DeviceScan<>::GetTempStorageBytes<T>(n)};
        for(bool inclusive : {false, true})
        {
            std::vector<T> expected(host.size(), static_cast<T>(-777));
            T              sum{};
            for(size_t i = 0u; i < n; ++i)
            {
                if(inclusive)
                    sum += host[i];
                expected[i] = sum;
                if(!inclusive)
                    sum += host[i];
            }
            std::vector<T> sentinel(host.size(), static_cast<T>(-777));
            r.stream << output.copy_from(luisa::span{sentinel}) << synchronize();
            CommandList commands;
            if(inclusive)
                scan.InclusiveSum(commands, temp.view(), input.view(), output.view(), n);
            else
                scan.ExclusiveSum(commands, temp.view(), input.view(), output.view(), n);
            std::vector<T> actual(host.size());
            r.stream << commands.commit() << output.copy_to(luisa::span{actual}) << synchronize();
            auto label = prefix + ".n=" + std::to_string(n) + (inclusive ? ".inclusive" : ".exclusive");
            expect_equal(actual, expected, label);
            temp.verify(r.stream, label);
            r.active(scan, label);
        }
        if(n == 16003u)
        {
            // Initial values and a noncommutative associative operator also use
            // the default ordinary policy, including multiple complete tiles.
            for(uint mode = 0u; mode < 4u; ++mode)
            {
                bool inclusive = mode % 2u != 0u;
                bool inplace   = mode < 2u;
                T    sum       = T{3};
                auto expected  = inplace ? host : std::vector<T>(host.size(), static_cast<T>(-777));
                for(size_t i = 0u; i < n; ++i)
                {
                    if(inclusive)
                        sum += host[i];
                    expected[i] = sum;
                    if(!inclusive)
                        sum += host[i];
                }
                std::vector<T> sentinel(host.size(), static_cast<T>(-777));
                r.stream << input.copy_from(luisa::span{host})
                         << output.copy_from(luisa::span{sentinel}) << synchronize();
                auto        target = inplace ? input.view() : output.view();
                CommandList commands;
                if(inclusive)
                    scan.InclusiveScanInit(commands, temp.view(), input.view(), target, n, SumOp{}, T{3});
                else
                    scan.ExclusiveScan(commands, temp.view(), input.view(), target, n, SumOp{}, T{3});
                std::vector<T> actual(host.size());
                r.stream << commands.commit() << target.copy_to(luisa::span{actual}) << synchronize();
                auto label = prefix + (inplace ? ".inplace" : ".out_of_place")
                             + (inclusive ? ".inclusive_init" : ".exclusive_init");
                expect_equal(actual, expected, label);
                temp.verify(r.stream, label);
                r.active(scan, label);
            }
            r.stream << input.copy_from(luisa::span{host}) << synchronize();
            auto        left = [](const Var<T>& a, const Var<T>&) noexcept { return a; };
            CommandList commands;
            scan.InclusiveScan(commands, temp.view(), input.view(), output.view(), n, left);
            std::vector<T> actual(host.size());
            r.stream << commands.commit() << output.copy_to(luisa::span{actual}) << synchronize();
            std::vector<T> expected(host.size(), static_cast<T>(-777));
            std::fill_n(expected.begin(), n, host.front());
            expect_equal(actual, expected, prefix + ".left_projection");
            temp.verify(r.stream, prefix + ".left_projection");
            r.active(scan, prefix + ".left_projection");
        }
    }
}

template <typename T, size_t Block = details::BLOCK_SIZE>
void segmented(Runner& r, bool short_segments = false)
{
    std::vector<uint> lengths;
    if(short_segments)
    {
        // A low average can contain a long outlier. Forty-one segments also
        // leave a partial final block in the warp policy.
        lengths = {0u, 1u, 31u, 32u, 33u, 127u, 128u, 129u, 255u, 256u, 257u, 4097u};
        lengths.resize(41u, 3u);
    }
    else
    {
        lengths = {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 8193u};
    }
    std::vector<uint> begins, ends, packed;
    uint              input_size  = 3u;
    uint              output_size = 5u;
    for(auto length : lengths)
    {
        begins.push_back(input_size);
        ends.push_back(input_size + length);
        packed.push_back(output_size);
        input_size += length + 3u;
        output_size += length + 5u;
    }
    const size_t   size = std::max(input_size, output_size) + 8u;
    std::vector<T> host(size, static_cast<T>(-777));
    for(size_t s = 0u; s < lengths.size(); ++s)
        for(uint i = begins[s]; i < ends[s]; ++i)
            host[i] = value<T>(i);
    auto input          = r.device.create_buffer<T>(size);
    auto output         = r.device.create_buffer<T>(size);
    auto begin_buffer   = r.device.create_buffer<uint>(lengths.size());
    auto end_buffer     = r.device.create_buffer<uint>(lengths.size());
    auto output_offsets = r.device.create_buffer<uint>(lengths.size());
    r.stream << begin_buffer.copy_from(luisa::span{begins}) << end_buffer.copy_from(luisa::span{ends})
             << output_offsets.copy_from(luisa::span{packed}) << synchronize();
    DeviceSegmentedScan<Block> scan;
    scan.create(r.device);
    const auto prefix = std::string{"segmented."} + type_name<T>() + ".block=" + std::to_string(Block)
                        + (short_segments ? ".short_ragged" : ".long_ragged");
    check(scan.template GetTempStorageBytes<T>(size, lengths.size()) == 0u, prefix + ".zero_temp");
    // Modes include the built-in initial-value SumOp entry points.
    for(uint mode = 0u; mode < 4u; ++mode)
    {
        for(uint layout = 0u; layout < 3u; ++layout)
        {
            bool           inclusive    = mode % 2u == 0u;
            bool           with_initial = mode >= 2u;
            bool           inplace      = layout == 2u;
            const auto&    offsets      = layout == 1u ? packed : begins;
            std::vector<T> expected = inplace ? host : std::vector<T>(size, static_cast<T>(-777));
            for(size_t s = 0u; s < lengths.size(); ++s)
            {
                T sum = with_initial ? T{3} : T{};
                for(uint i = 0u; i < lengths[s]; ++i)
                {
                    auto x = host[begins[s] + i];
                    if(inclusive)
                        sum += x;
                    expected[offsets[s] + i] = sum;
                    if(!inclusive)
                        sum += x;
                }
            }
            std::vector<T> sentinel(size, static_cast<T>(-777));
            r.stream << input.copy_from(luisa::span{host})
                     << output.copy_from(luisa::span{sentinel}) << synchronize();
            auto        target        = inplace ? input.view() : output.view();
            auto        output_begins = layout == 1u ? output_offsets.view() : begin_buffer.view();
            CommandList commands;
            if(mode == 0u)
                scan.InclusiveSegmentedSum(commands,
                                           {},
                                           input.view(),
                                           target,
                                           begin_buffer.view(),
                                           end_buffer.view(),
                                           output_begins,
                                           lengths.size());
            if(mode == 1u)
                scan.ExclusiveSegmentedSum(commands,
                                           {},
                                           input.view(),
                                           target,
                                           begin_buffer.view(),
                                           end_buffer.view(),
                                           output_begins,
                                           lengths.size());
            if(mode == 2u)
                scan.InclusiveSegmentedScanInit(commands,
                                                {},
                                                input.view(),
                                                target,
                                                begin_buffer.view(),
                                                end_buffer.view(),
                                                output_begins,
                                                lengths.size(),
                                                SumOp{},
                                                T{3});
            if(mode == 3u)
                scan.ExclusiveSegmentedScan(commands,
                                            {},
                                            input.view(),
                                            target,
                                            begin_buffer.view(),
                                            end_buffer.view(),
                                            output_begins,
                                            lengths.size(),
                                            SumOp{},
                                            T{3});
            std::vector<T> actual(size);
            r.stream << commands.commit() << target.copy_to(luisa::span{actual}) << synchronize();
            expect_equal(actual,
                         expected,
                         prefix + ".mode=" + std::to_string(mode) + ".layout=" + std::to_string(layout));
        }
    }
}

void block_initial(Runner& r)
{
    constexpr uint block  = 128u;
    Kernel1D       kernel = [=](BufferVar<int> output)
    {
        set_block_size(block);
        UInt                      thread = thread_id().x;
        Int                       item   = cast<int>(thread % 7u + 1u);
        Int                       exclusive, inclusive, aggregate;
        BlockScan<int, block, 1u> scan;
        scan.ExclusiveScan(item, exclusive, aggregate, SumOp{}, Int{3});
        output.write(thread, exclusive);
        output.write(block + thread, aggregate);
        sync_block();
        scan.InclusiveScan(item, inclusive, aggregate, SumOp{}, Int{3});
        output.write(2u * block + thread, inclusive);
        output.write(3u * block + thread, aggregate);
    };
    auto             shader = r.device.compile(kernel);
    auto             output = r.device.create_buffer<int>(4u * block);
    std::vector<int> actual(output.size()), expected(output.size());
    int              sum     = 3;
    int              raw_sum = 0;
    for(uint i = 0u; i < block; ++i)
        raw_sum += value<int>(i);
    for(uint i = 0u; i < block; ++i)
    {
        expected[i] = sum;
        sum += value<int>(i);
        expected[2u * block + i] = sum;
        expected[block + i] = expected[3u * block + i] = raw_sum;
    }
    r.stream << shader(output).dispatch(block) << output.copy_to(luisa::span{actual}) << synchronize();
    expect_equal(actual, expected, "block.scalar.initial_and_raw_aggregate");
}

void segmented_many(Runner& r, bool block_policy)
{
    const uint        segments_per_block = block_policy ? 1u : 256u / r.device.compute_warp_size();
    const uint        boundary           = 65535u * segments_per_block;
    const uint        segments           = boundary + 2u;
    const uint        n                  = 2u * segments;
    std::vector<int>  host(n + 8u, 1), sentinel(n + 8u, -777);
    std::vector<uint> begins(segments), ends(segments);
    for(uint s = 0u; s < segments; ++s)
    {
        begins[s] = 2u * s;
        ends[s]   = begins[s] + 2u;
    }
    // Exercise an empty segment next to the split as well as a nonempty one
    // on either side. Offset buffers retain their original aligned views.
    ends[boundary - 1u] = begins[boundary - 1u];
    // A larger input view selects the block heuristic without making all
    // segments long. Unused capacity is never read by either policy.
    const size_t input_capacity =
        block_policy ? (size_t{r.device.compute_warp_size()} * 8u + 1u) * segments : host.size();
    auto input        = r.device.create_buffer<int>(input_capacity);
    auto output       = r.device.create_buffer<int>(host.size());
    auto begin_buffer = r.device.create_buffer<uint>(segments);
    auto end_buffer   = r.device.create_buffer<uint>(segments);
    r.stream << input.view(0u, host.size()).copy_from(luisa::span{host})
             << begin_buffer.copy_from(luisa::span{begins})
             << end_buffer.copy_from(luisa::span{ends}) << synchronize();
    DeviceSegmentedScan<> scan;
    scan.create(r.device);
    for(bool inclusive : {false, true})
    {
        auto expected = sentinel;
        for(uint s = 0u; s < segments; ++s)
            if(ends[s] != begins[s])
            {
                expected[2u * s]      = inclusive ? 1 : 0;
                expected[2u * s + 1u] = inclusive ? 2 : 1;
            }
        r.stream << output.copy_from(luisa::span{sentinel}) << synchronize();
        CommandList commands;
        if(inclusive)
            scan.InclusiveSegmentedSum(
                commands, {}, input.view(), output.view(), begin_buffer.view(), end_buffer.view(), segments);
        else
            scan.ExclusiveSegmentedSum(
                commands, {}, input.view(), output.view(), begin_buffer.view(), end_buffer.view(), segments);
        std::vector<int> actual(output.size());
        r.stream << commands.commit() << output.copy_to(luisa::span{actual}) << synchronize();
        auto label = std::string{"segmented.chunk."} + (block_policy ? "block" : "warp")
                     + (inclusive ? ".inclusive" : ".exclusive");
        expect_equal(actual, expected, label);
    }
}

void segmented_nonassociative(Runner& r)
{
    constexpr size_t n = 4097u;
    std::vector<int> host(n);
    for(size_t i = 0u; i < n; ++i)
        host[i] = value<int>(i);
    auto input  = r.device.create_buffer<int>(n);
    auto output = r.device.create_buffer<int>(n);
    auto begins = r.device.create_buffer<uint>(1u);
    auto ends   = r.device.create_buffer<uint>(1u);
    uint begin = 0u, end = n;
    r.stream << input.copy_from(luisa::span{host}) << begins.copy_from(&begin)
             << ends.copy_from(&end) << synchronize();
    DeviceSegmentedScan<> scan;
    scan.create(r.device);
    auto subtract = [](const Int& lhs, const Int& rhs) noexcept { return lhs - rhs; };
    for(bool inclusive : {false, true})
    {
        std::vector<int> expected(n), actual(n);
        int              sum = 99;
        for(size_t i = 0u; i < n; ++i)
        {
            if(inclusive)
                sum -= host[i];
            expected[i] = sum;
            if(!inclusive)
                sum -= host[i];
        }
        CommandList commands;
        if(inclusive)
            scan.InclusiveSegmentedScanInit(
                commands, {}, input.view(), output.view(), begins.view(), ends.view(), 1u, subtract, 99);
        else
            scan.ExclusiveSegmentedScan(
                commands, {}, input.view(), output.view(), begins.view(), ends.view(), 1u, subtract, 99);
        r.stream << commands.commit() << output.copy_to(luisa::span{actual}) << synchronize();
        expect_equal(actual, expected, inclusive ? "segmented.subtract.inclusive_init" : "segmented.subtract.exclusive");
    }
}
}  // namespace

int main(int argc, char* argv[])
{
    luisa::string_view directory;
    luisa::string_view backend         = "cuda";
    std::string_view   selected        = "all";
    bool               require_interop = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg{argv[i]};
        if(arg.starts_with("--ctx="))
            directory = argv[i] + 6;
        else if(arg.starts_with("--backend="))
            backend = argv[i] + 10;
        else if(arg.starts_with("--case="))
            selected = arg.substr(7u);
        else if(arg == "--require-interop")
            require_interop = true;
        else
            return 2;
    }
    if(selected != "all" && selected != "ordinary" && selected != "segmented")
        return 2;
    if(require_interop && (backend != "vk" || selected != "ordinary"))
        return 2;
    if(require_interop)
    {
#ifdef _WIN32
        _putenv_s("LCPP_INTEROP", "all");
#else
        setenv("LCPP_INTEROP", "all", 1);
#endif
    }
    Context context{directory.empty() ? argv[0] : directory};
    Device  device  = context.create_device(backend);
    Stream  stream  = device.create_stream();
    auto    interop = require_interop ? InteropContext::create(context, device) : nullptr;
    if(require_interop && !interop)
    {
        std::fprintf(stderr, "Required Vulkan CUDA interop is unavailable\n");
        return 3;
    }
    Runner runner{device, stream, interop.get(), require_interop};
    if(selected != "segmented")
    {
        ordinary<int>(runner);
        ordinary<uint>(runner);
        ordinary<float>(runner);
        ordinary<double>(runner);
        if(!require_interop)
            block_initial(runner);
    }
    if(selected != "ordinary")
    {
        segmented<int>(runner);
        segmented<uint>(runner);
        segmented<float>(runner);
        segmented<double>(runner);
        segmented<int, 128u>(runner);
        segmented<int, 32u>(runner);
        segmented<int>(runner, true);
        segmented<uint>(runner, true);
        segmented<float>(runner, true);
        segmented<double>(runner, true);
        segmented<int, 128u>(runner, true);
        segmented<int, 32u>(runner, true);
        segmented_many(runner, true);
        segmented_many(runner, false);
        segmented_nonassociative(runner);
    }
    std::printf("Scan optimization result: %zu checks, %zu failures\n", checks, failures);
    return failures == 0u ? 0 : 1;
}
