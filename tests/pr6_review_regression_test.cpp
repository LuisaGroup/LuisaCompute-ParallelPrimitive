// Focused GPU regressions for PR #6. Run each case in a separate process when
// testing the pre-fix sources or using compute-sanitizer.
//
// pr6_review_regression_test --ctx=... --backend=cuda --case=all
// LCPP_INTEROP=all pr6_review_regression_test --ctx=... --backend=vk --require-interop
// Cases: cache, segment, prefix, large, policy. No unavailable-interop skips.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
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

size_t         failures = 0u;
size_t         checks   = 0u;
constexpr uint canary   = 0xa5c3f17du;

void check(bool ok, const std::string& label)
{
    ++checks;
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", label.c_str());
    failures += !ok;
}

template <typename T>
void equal(T actual, T expected, const std::string& label)
{
    if(actual != expected)
    {
        std::fprintf(stderr, "  actual=%.17g expected=%.17g\n", static_cast<double>(actual), static_cast<double>(expected));
    }
    check(actual == expected, label);
}

template <typename T>
void equal(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& label)
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

// The API receives exactly the requested scratch subview. The larger backing
// allocation absorbs the known pre-fix overruns, which are detected as changed
// canaries without allowing those writes to damage unrelated allocations.
struct GuardedTemp
{
    size_t       words;
    Buffer<uint> backing;

    GuardedTemp(Device& device, Stream& stream, size_t bytes, size_t guard_words = 4096u)
        : words{std::max<size_t>(1u, bytes_to_uint_count(bytes))}
        , backing{device.create_buffer<uint>(words + guard_words)}
    {
        std::vector<uint> initial(backing.size(), canary);
        stream << backing.copy_from(luisa::span{initial}) << synchronize();
    }

    [[nodiscard]] BufferView<uint> view() const { return backing.view(0u, words); }

    void verify(Stream& stream, const std::string& label) const
    {
        std::vector<uint> tail(backing.size() - words);
        stream << backing.view(words, tail.size()).copy_to(luisa::span{tail}) << synchronize();
        auto changed = std::find_if(tail.begin(), tail.end(), [](uint v) { return v != canary; });
        if(changed != tail.end())
        {
            std::fprintf(stderr,
                         "  scratch overrun at word %zu: 0x%08x\n",
                         words + static_cast<size_t>(changed - tail.begin()),
                         *changed);
        }
        check(changed == tail.end(), label + ".scratch_canary");
    }
};

struct Runner
{
    Device&         device;
    Stream&         stream;
    InteropContext* interop;
    bool            require_interop;

    template <typename Module>
    void create(Module& module, const char* name, bool dsl = false)
    {
        module.create(device, &stream, dsl ? nullptr : interop);
        if(!dsl && require_interop)
        {
            check(module.interop_active(), std::string{name} + ".interop_active_after_create");
        }
    }

    template <typename Module>
    void active(const Module& module, const std::string& label, bool dsl = false)
    {
        if(!dsl && require_interop)
        {
            check(module.interop_active(), label + ".interop_active_after_dispatch");
        }
    }
};

auto captured_binary(bool maximum)
{
    return [maximum](const Int& a, const Int& b) noexcept
    {
        if(maximum)
        {
            return compute::max(a, b);
        }
        return a + b;
    };
}

auto captured_unary(int bias)
{
    return [bias](const Int& x) noexcept { return x + bias; };
}

void cache_case(Runner& r)
{
    constexpr size_t n = 4097u;
    std::vector<int> host(n);
    for(size_t i = 0u; i < n; ++i)
    {
        host[i] = static_cast<int>(i % 7u) + 1;
    }
    const int sum    = std::accumulate(host.begin(), host.end(), 0);
    auto      input  = r.device.create_buffer<int>(n);
    auto      output = r.device.create_buffer<int>(n);
    r.stream << input.copy_from(luisa::span{host}) << synchronize();
    DeviceReduce<> reducer;
    DeviceScan<>   scanner;
    r.create(reducer, "cache.reduce");
    r.create(scanner, "cache.scan");
    GuardedTemp reduce_temp{r.device, r.stream, DeviceReduce<>::GetTempStorageBytes<int>(n)};
    GuardedTemp scan_temp{r.device, r.stream, DeviceScan<>::GetTempStorageBytes<int>(n)};

    Callable<int(int, int)> callable_sum = [](Int a, Int b) { return a + b; };
    Callable<int(int, int)> callable_max = [](Int a, Int b) { return compute::max(a, b); };
    static_assert(std::is_same_v<decltype(callable_sum), decltype(callable_max)>);
    auto reduce = [&](auto op, int expected, const std::string& label)
    {
        CommandList commands;
        reducer.Reduce(commands, reduce_temp.view(), input.view(), output.view(0u, 1u), n, op, 0);
        int actual = 0;
        r.stream << commands.commit() << output.view(0u, 1u).copy_to(&actual) << synchronize();
        equal(actual, expected, label);
        r.active(reducer, label);
    };
    reduce(callable_sum, sum, "cache.reduce.callable_sum");
    reduce(callable_max, 7, "cache.reduce.callable_max");
    reduce(callable_sum, sum, "cache.reduce.callable_sum_reused");
    reduce(captured_binary(false), sum, "cache.reduce.capture_sum");
    reduce(captured_binary(true), 7, "cache.reduce.capture_max");

    auto transform = [&](auto op, int expected, const std::string& label)
    {
        CommandList commands;
        reducer.TransformReduce(commands, reduce_temp.view(), input.view(), output.view(0u, 1u), n, SumOp(), op, 0);
        int actual = 0;
        r.stream << commands.commit() << output.view(0u, 1u).copy_to(&actual) << synchronize();
        equal(actual, expected, label);
        r.active(reducer, label);
    };
    transform(captured_unary(1), sum + static_cast<int>(n), "cache.transform.capture_1");
    transform(captured_unary(7), sum + 7 * static_cast<int>(n), "cache.transform.capture_7");
    Callable<int(int)> callable_double = [](Int x) { return 2 * x; };
    Callable<int(int)> callable_square = [](Int x) { return x * x; };
    int                sum_square      = 0;
    for(auto x : host)
    {
        sum_square += x * x;
    }
    transform(callable_double, 2 * sum, "cache.transform.callable_double");
    transform(callable_square, sum_square, "cache.transform.callable_square");

    auto equivalent_identity = [](const Int& x) -> Int { return x; };
    check(get_unary_op_hash<int>(IdentityOp()) == get_unary_op_hash<int>(equivalent_identity),
          "cache.transform.equivalent_identity_ast_hash");
    // Equal transform ASTs can select different interop tile policies. The
    // implementation must include that policy in shader identity; these calls
    // verify the public result in both cache insertion orders.
    for(bool lambda_first : {false, true})
    {
        DeviceReduce<> equivalent_reducer;
        r.create(equivalent_reducer, "cache.equivalent_identity");
        for(size_t count : {2048u, 8193u})
        {
            std::vector<int> identity_host(count);
            for(size_t i = 0u; i < count; ++i)
            {
                identity_host[i] = static_cast<int>(i % 7u) + 1;
            }
            const int identity_sum = std::accumulate(identity_host.begin(), identity_host.end(), 0);
            auto      identity_input = r.device.create_buffer<int>(count);
            GuardedTemp identity_temp{r.device, r.stream, DeviceReduce<>::GetTempStorageBytes<int>(count)};
            r.stream << identity_input.copy_from(luisa::span{identity_host}) << synchronize();
            auto identity_transform = [&](auto op, const std::string& suffix)
            {
                const auto label = std::string{"cache.equivalent_identity."}
                                   + (lambda_first ? "lambda_first." : "identity_first.")
                                   + std::to_string(count) + "." + suffix;
                CommandList commands;
                equivalent_reducer.TransformReduce(
                    commands, identity_temp.view(), identity_input.view(), output.view(0u, 1u), count, SumOp(), op, 0);
                int actual = 0;
                r.stream << commands.commit() << output.view(0u, 1u).copy_to(&actual) << synchronize();
                equal(actual, identity_sum, label);
                identity_temp.verify(r.stream, label);
                r.active(equivalent_reducer, label);
            };
            if(lambda_first)
            {
                identity_transform(equivalent_identity, "lambda");
                identity_transform(IdentityOp(), "identity");
                identity_transform(equivalent_identity, "lambda_reused");
            }
            else
            {
                identity_transform(IdentityOp(), "identity");
                identity_transform(equivalent_identity, "lambda");
                identity_transform(IdentityOp(), "identity_reused");
            }
        }
    }

    auto scan = [&](auto op, bool maximum, const std::string& label)
    {
        std::vector<int> expected(n), actual(n);
        int              aggregate = 0;
        for(size_t i = 0u; i < n; ++i)
        {
            expected[i] = aggregate;
            aggregate   = maximum ? std::max(aggregate, host[i]) : aggregate + host[i];
        }
        CommandList commands;
        scanner.ExclusiveScan(commands, scan_temp.view(), input.view(), output.view(), n, op, 0);
        r.stream << commands.commit() << output.copy_to(luisa::span{actual}) << synchronize();
        equal(actual, expected, label);
        r.active(scanner, label);
    };
    scan(callable_sum, false, "cache.scan.callable_sum");
    scan(callable_max, true, "cache.scan.callable_max");
    scan(callable_sum, false, "cache.scan.callable_sum_reused");
    scan(captured_binary(false), false, "cache.scan.capture_sum");
    scan(captured_binary(true), true, "cache.scan.capture_max");
    reduce_temp.verify(r.stream, "cache.reduce");
    scan_temp.verify(r.stream, "cache.scan");
}

void segment_case(Runner& r)
{
    constexpr uint   segments     = 4097u;
    constexpr uint   segment_size = 1025u;
    constexpr size_t n            = static_cast<size_t>(segments) * segment_size;
    std::vector<int> host(n);
    for(uint s = 0u; s < segments; ++s)
    {
        for(uint i = 0u; i < segment_size; ++i)
        {
            host[static_cast<size_t>(s) * segment_size + i] = static_cast<int>((i * 13u + s) % 997u);
        }
        host[static_cast<size_t>(s) * segment_size + 7u] = -100000 - static_cast<int>(s);
        host[static_cast<size_t>(s) * segment_size + segment_size - 9u] = 100000 + static_cast<int>(s);
    }
    // Reverse physical segments for offset calls. Early blocks then overwrite
    // input belonging to blocks that have not started when the two scratch
    // regions alias, making the old overlap observable in the returned values.
    std::vector<uint> begins(segments), ends(segments);
    for(uint s = 0u; s < segments; ++s)
    {
        begins[s] = (segments - 1u - s) * segment_size;
        ends[s]   = begins[s] + segment_size;
    }
    auto input        = r.device.create_buffer<int>(n);
    auto begin_buffer = r.device.create_buffer<uint>(segments);
    auto end_buffer   = r.device.create_buffer<uint>(segments);
    auto values       = r.device.create_buffer<int>(segments);
    auto indices      = r.device.create_buffer<uint>(segments);
    r.stream << input.copy_from(luisa::span{host}) << begin_buffer.copy_from(luisa::span{begins})
             << end_buffer.copy_from(luisa::span{ends}) << synchronize();
    DeviceSegmentReduce<> reducer;
    r.create(reducer, "segment");
    for(bool fixed : {true, false})
    {
        for(bool minimum : {true, false})
        {
            std::string label = std::string{"segment."} + (fixed ? "fixed." : "offset.")
                                + (minimum ? "argmin" : "argmax");
            GuardedTemp temp{r.device, r.stream, DeviceSegmentReduce<>::GetArgTempStorageBytes<int>(n, segments)};
            CommandList commands;
            if(fixed)
            {
                if(minimum)
                {
                    reducer.ArgMin(commands, temp.view(), input.view(), values.view(), indices.view(), segments, segment_size);
                }
                else
                {
                    reducer.ArgMax(commands, temp.view(), input.view(), values.view(), indices.view(), segments, segment_size);
                }
            }
            else
            {
                if(minimum)
                {
                    reducer.ArgMin(commands,
                                   temp.view(),
                                   input.view(),
                                   values.view(),
                                   indices.view(),
                                   segments,
                                   begin_buffer.view(),
                                   end_buffer.view());
                }
                else
                {
                    reducer.ArgMax(commands,
                                   temp.view(),
                                   input.view(),
                                   values.view(),
                                   indices.view(),
                                   segments,
                                   begin_buffer.view(),
                                   end_buffer.view());
                }
            }
            std::vector<int>  actual_values(segments), expected_values(segments);
            std::vector<uint> actual_indices(segments), expected_indices(segments);
            for(uint s = 0u; s < segments; ++s)
            {
                const uint begin = fixed ? s * segment_size : begins[s];
                const auto first = host.begin() + begin;
                const auto last  = first + segment_size;
                const auto extremum = minimum ? std::min_element(first, last) : std::max_element(first, last);
                expected_values[s]  = *extremum;
                expected_indices[s] = static_cast<uint>(extremum - first);
            }
            r.stream << commands.commit() << values.copy_to(luisa::span{actual_values})
                     << indices.copy_to(luisa::span{actual_indices}) << synchronize();
            equal(actual_values, expected_values, label + ".values");
            equal(actual_indices, expected_indices, label + ".relative_indices");
            temp.verify(r.stream, label);
            r.active(reducer, label);
        }
    }
}

void prefix_case(Runner& r)
{
    constexpr size_t prefix = 257u;
    constexpr size_t n      = 8193u;
    std::vector<int> host(n, 500000);
    for(size_t i = 0u; i < prefix; ++i)
    {
        host[i] = static_cast<int>(100u + i);
    }
    host[17u]         = -9001;
    host[prefix - 1u] = 9001;
    for(size_t i = prefix; i < n; ++i)
    {
        host[i] = i % 2u ? -500000 : 500000;
    }
    auto input   = r.device.create_buffer<int>(n);
    auto values  = r.device.create_buffer<int>(33u);
    auto indices = r.device.create_buffer<uint>(33u);
    r.stream << input.copy_from(luisa::span{host}) << synchronize();
    DeviceReduce<> reducer;
    r.create(reducer, "prefix");
    for(bool minimum : {true, false})
    {
        std::string label = minimum ? "prefix.argmin" : "prefix.argmax";
        GuardedTemp temp{r.device, r.stream, DeviceReduce<>::GetArgTempStorageBytes<int>(prefix), 2u * n + 4096u};
        std::vector<int>  actual_values(33u, 1234567);
        std::vector<uint> actual_indices(33u, canary);
        r.stream << values.copy_from(luisa::span{actual_values})
                 << indices.copy_from(luisa::span{actual_indices}) << synchronize();
        CommandList commands;
        if(minimum)
        {
            reducer.ArgMin(commands, temp.view(), input.view(), values.view(0u, 1u), indices.view(0u, 1u), prefix);
        }
        else
        {
            reducer.ArgMax(commands, temp.view(), input.view(), values.view(0u, 1u), indices.view(0u, 1u), prefix);
        }
        r.stream << commands.commit() << values.copy_to(luisa::span{actual_values})
                 << indices.copy_to(luisa::span{actual_indices}) << synchronize();
        equal(actual_values[0], minimum ? -9001 : 9001, label + ".value");
        equal(actual_indices[0], minimum ? 17u : static_cast<uint>(prefix - 1u), label + ".index");
        check(std::all_of(
                  actual_values.begin() + 1u, actual_values.end(), [](int v) { return v == 1234567; })
                  && std::all_of(actual_indices.begin() + 1u,
                                 actual_indices.end(),
                                 [](uint v) { return v == canary; }),
              label + ".output_canaries");
        temp.verify(r.stream, label);
        r.active(reducer, label);
    }
}

void large_case(Runner& r)
{
    constexpr size_t  n          = (1u << 24u) + 1u;
    constexpr uint    tail_value = 123456789u;
    std::vector<uint> host(n, 0u);
    host.back() = tail_value;
    auto input  = r.device.create_buffer<uint>(n);
    auto output = r.device.create_buffer<uint>(1u);
    r.stream << input.copy_from(luisa::span{host}) << synchronize();
    DeviceReduce<> reducer;
    r.create(reducer, "large");
    GuardedTemp temp{r.device, r.stream, DeviceReduce<>::GetTempStorageBytes<uint>(n)};
    for(bool maximum : {false, true})
    {
        std::string label = maximum ? "large.max_tail" : "large.sum_tail";
        CommandList commands;
        if(maximum)
        {
            reducer.Max(commands, temp.view(), input.view(), output.view(), n);
        }
        else
        {
            reducer.Sum(commands, temp.view(), input.view(), output.view(), n);
        }
        uint actual = 0u;
        r.stream << commands.commit() << output.copy_to(&actual) << synchronize();
        equal(actual, tail_value, label);
        r.active(reducer, label);
    }
    temp.verify(r.stream, "large");
}

template <size_t Items, typename T, bool Transform>
void policy_type(Runner& r, bool dsl)
{
    using Reducer = DeviceReduce<256u, 32u, Items>;
    Reducer reducer;
    r.create(reducer, "policy", dsl);
    constexpr size_t tile = 256u * Items;
    for(size_t n : {tile - 3u, tile, tile + 1u, tile * 257u + 1u})
    {
        std::string label = std::string{"policy."} + (dsl ? "dsl." : "interop.")
                            + std::to_string(Items) + "." + (std::is_same_v<T, int> ? "int" : "float")
                            + (Transform ? ".transform." : ".identity.") + std::to_string(n);
        std::vector<T> host(n);
        T              expected = 0;
        for(size_t i = 0u; i < n; ++i)
        {
            const int x = static_cast<int>(i % 7u) - 3;
            host[i]     = std::is_same_v<T, int> ? static_cast<T>(x) : static_cast<T>(x) * T(0.25);
            expected += Transform ? T(2) * host[i] + T(0.5) : host[i];
        }
        auto        input  = r.device.create_buffer<T>(n);
        auto        output = r.device.create_buffer<T>(1u);
        GuardedTemp temp{r.device, r.stream, Reducer::template GetTempStorageBytes<T>(n)};
        r.stream << input.copy_from(luisa::span{host}) << synchronize();
        CommandList commands;
        if constexpr(Transform)
        {
            reducer.TransformReduce(
                commands,
                temp.view(),
                input.view(),
                output.view(),
                n,
                SumOp(),
                [](const Var<T>& x) { return T(2) * x + T(0.5); },
                T(0));
        }
        else
        {
            reducer.Sum(commands, temp.view(), input.view(), output.view(), n);
        }
        T actual = 0;
        r.stream << commands.commit() << output.copy_to(&actual) << synchronize();
        equal(actual, expected, label);
        temp.verify(r.stream, label);
        r.active(reducer, label, dsl);
    }
}

template <size_t Items>
void policy_items(Runner& r, bool dsl)
{
    policy_type<Items, int, false>(r, dsl);
    policy_type<Items, float, false>(r, dsl);
    policy_type<Items, float, true>(r, dsl);
}

void policy_case(Runner& r)
{
    // Always test the portable policy, including on a Vulkan interop run.
    for(bool dsl : {true, false})
    {
        if(!dsl && r.interop == nullptr)
        {
            continue;
        }
        policy_items<8u>(r, dsl);
        policy_items<16u>(r, dsl);
        policy_items<32u>(r, dsl);
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    luisa::string_view ctx_dir;
    luisa::string_view backend         = "cuda";
    std::string_view   selected        = "all";
    bool               require_interop = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg{argv[i]};
        if(arg.starts_with("--ctx="))
        {
            ctx_dir = argv[i] + 6;
        }
        else if(arg.starts_with("--backend="))
        {
            backend = argv[i] + 10;
        }
        else if(arg.starts_with("--case="))
        {
            selected = arg.substr(7u);
        }
        else if(arg == "--require-interop")
        {
            require_interop = true;
        }
        else
        {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if(selected != "all" && selected != "cache" && selected != "segment" && selected != "prefix"
       && selected != "large" && selected != "policy")
    {
        std::fprintf(stderr, "Unknown case: %.*s\n", static_cast<int>(selected.size()), selected.data());
        return 2;
    }
    if(require_interop)
    {
#ifdef _WIN32
        _putenv_s("LCPP_INTEROP", "all");
#else
        setenv("LCPP_INTEROP", "all", 1);
#endif
    }
    const char*            env    = std::getenv("LCPP_INTEROP");
    const std::string_view policy = env == nullptr ? "" : env;
    require_interop |=
        backend == "vk" && (policy == "all" || policy == "1" || policy == "on" || policy == "true");
    if(require_interop && backend != "vk")
    {
        std::fprintf(stderr, "--require-interop requires --backend=vk\n");
        return 2;
    }
    Context                           context{ctx_dir.empty() ? argv[0] : ctx_dir};
    Device                            device = context.create_device(backend);
    Stream                            stream = device.create_stream();
    luisa::unique_ptr<InteropContext> interop;
    if(backend == "vk" && !policy.empty() && policy != "0" && policy != "off" && policy != "none")
    {
        interop = InteropContext::create(context, device);
    }
    if(require_interop && interop == nullptr)
    {
        std::fprintf(stderr, "[FAIL] Vulkan CUDA interop was required but is unavailable\n");
        return 3;
    }
    std::printf("PR6 regressions: backend=%.*s case=%.*s interop=%s required=%s\n",
                static_cast<int>(backend.size()),
                backend.data(),
                static_cast<int>(selected.size()),
                selected.data(),
                interop ? "available" : "disabled",
                require_interop ? "yes" : "no");
    Runner runner{device, stream, interop.get(), require_interop};
    if(selected == "all" || selected == "cache")
    {
        cache_case(runner);
    }
    if(selected == "all" || selected == "segment")
    {
        segment_case(runner);
    }
    if(selected == "all" || selected == "prefix")
    {
        prefix_case(runner);
    }
    if(selected == "all" || selected == "large")
    {
        large_case(runner);
    }
    if(selected == "all" || selected == "policy")
    {
        policy_case(runner);
    }
    std::printf("PR6 result: %zu checks, %zu failures\n", checks, failures);
    return failures == 0u ? 0 : 1;
}
