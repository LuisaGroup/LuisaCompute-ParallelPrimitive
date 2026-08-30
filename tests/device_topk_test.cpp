// Tests for DeviceTopK.
// Covers min/max keys and pairs, decomposed keys, capped K, ties, and empty work.

#include <lcpp/parallel_primitive.h>

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <tuple>
#include <utility>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

struct CompositeKey
{
    uint primary;
    int  secondary;
};

LUISA_STRUCT(CompositeKey, primary, secondary){};

namespace
{
[[nodiscard]] bool composite_less(const CompositeKey& lhs, const CompositeKey& rhs) noexcept
{
    return std::tie(lhs.primary, lhs.secondary) < std::tie(rhs.primary, rhs.secondary);
}

[[nodiscard]] bool composite_equal(const CompositeKey& lhs, const CompositeKey& rhs) noexcept
{
    return lhs.primary == rhs.primary && lhs.secondary == rhs.secondary;
}
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

    DeviceTopK<> topk;
    topk.create(device);

    "device_topk_selects_numeric_min_and_max_keys_without_order_assumptions"_test = [&]
    {
        constexpr size_t          k = 4u;
        const std::array<int, 8u> input{7, 1, 9, 3, 3, 8, 0, 5};
        const std::array<int, k>  expected_min{0, 1, 3, 3};
        const std::array<int, k>  expected_max{5, 7, 8, 9};
        auto                      input_buffer = device.create_buffer<int>(input.size());
        auto                      min_output   = device.create_buffer<int>(k);
        auto                      max_output   = device.create_buffer<int>(k);

        CommandList commands;
        topk.MinKeys(commands, BufferView<uint>{}, input_buffer.view(), min_output.view(), input.size(), k);
        topk.MaxKeys(commands, BufferView<uint>{}, input_buffer.view(), max_output.view(), input.size(), k);

        std::array<int, k> min_result{};
        std::array<int, k> max_result{};
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit()
               << min_output.copy_to(luisa::span{min_result})
               << max_output.copy_to(luisa::span{max_result}) << synchronize();
        std::sort(min_result.begin(), min_result.end());
        std::sort(max_result.begin(), max_result.end());
        for(size_t index = 0u; index < k; ++index)
        {
            expect(min_result[index] == expected_min[index]);
            expect(max_result[index] == expected_max[index]);
        }
    };

    "device_topk_numeric_pairs_preserve_key_value_association"_test = [&]
    {
        constexpr size_t                         k = 3u;
        const std::array<int, 6u>                keys{4, 1, 9, 2, 7, 3};
        const std::array<int, 6u>                values{40, 10, 90, 20, 70, 30};
        const std::array<std::pair<int, int>, k> expected_min{{{1, 10}, {2, 20}, {3, 30}}};
        const std::array<std::pair<int, int>, k> expected_max{{{7, 70}, {9, 90}, {4, 40}}};
        auto keys_input        = device.create_buffer<int>(keys.size());
        auto values_input      = device.create_buffer<int>(values.size());
        auto min_keys_output   = device.create_buffer<int>(k);
        auto min_values_output = device.create_buffer<int>(k);
        auto max_keys_output   = device.create_buffer<int>(k);
        auto max_values_output = device.create_buffer<int>(k);

        CommandList commands;
        topk.MinPairs(commands,
                      BufferView<uint>{},
                      keys_input.view(),
                      min_keys_output.view(),
                      values_input.view(),
                      min_values_output.view(),
                      keys.size(),
                      k);
        topk.MaxPairs(commands,
                      BufferView<uint>{},
                      keys_input.view(),
                      max_keys_output.view(),
                      values_input.view(),
                      max_values_output.view(),
                      keys.size(),
                      k);

        std::array<int, k> min_keys{};
        std::array<int, k> min_values{};
        std::array<int, k> max_keys{};
        std::array<int, k> max_values{};
        stream << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
               << commands.commit() << min_keys_output.copy_to(luisa::span{min_keys})
               << min_values_output.copy_to(luisa::span{min_values})
               << max_keys_output.copy_to(luisa::span{max_keys})
               << max_values_output.copy_to(luisa::span{max_values}) << synchronize();

        std::array<std::pair<int, int>, k> min_result{};
        std::array<std::pair<int, int>, k> max_result{};
        for(size_t index = 0u; index < k; ++index)
        {
            min_result[index] = {min_keys[index], min_values[index]};
            max_result[index] = {max_keys[index], max_values[index]};
        }
        std::sort(min_result.begin(), min_result.end());
        std::sort(max_result.begin(), max_result.end());
        auto sorted_expected_max = expected_max;
        std::sort(sorted_expected_max.begin(), sorted_expected_max.end());
        for(size_t index = 0u; index < k; ++index)
        {
            expect(min_result[index] == expected_min[index]);
            expect(max_result[index] == sorted_expected_max[index]);
        }
    };

    "device_topk_supports_decomposed_keys_for_all_method_groups"_test = [&]
    {
        constexpr size_t                   k = 2u;
        const std::array<CompositeKey, 5u> keys{{{2u, 5}, {1u, 9}, {2u, -1}, {0u, 7}, {3u, 0}}};
        const std::array<int, 5u>          values{25, 19, 21, 7, 30};
        auto keys_input      = device.create_buffer<CompositeKey>(keys.size());
        auto values_input    = device.create_buffer<int>(values.size());
        auto min_keys_output = device.create_buffer<CompositeKey>(k);
        auto max_keys_output = device.create_buffer<CompositeKey>(k);
        auto min_pair_keys   = device.create_buffer<CompositeKey>(k);
        auto max_pair_keys   = device.create_buffer<CompositeKey>(k);
        auto min_pair_values = device.create_buffer<int>(k);
        auto max_pair_values = device.create_buffer<int>(k);
        auto decomposer      = [](const Var<CompositeKey>& key) noexcept
        { return std::tuple{key.primary, key.secondary}; };

        CommandList commands;
        topk.MinKeys(commands, BufferView<uint>{}, keys_input.view(), min_keys_output.view(), keys.size(), k, decomposer);
        topk.MaxKeys(commands, BufferView<uint>{}, keys_input.view(), max_keys_output.view(), keys.size(), k, decomposer);
        topk.MinPairs(commands,
                      BufferView<uint>{},
                      keys_input.view(),
                      min_pair_keys.view(),
                      values_input.view(),
                      min_pair_values.view(),
                      keys.size(),
                      k,
                      decomposer);
        topk.MaxPairs(commands,
                      BufferView<uint>{},
                      keys_input.view(),
                      max_pair_keys.view(),
                      values_input.view(),
                      max_pair_values.view(),
                      keys.size(),
                      k,
                      decomposer);

        std::array<CompositeKey, k> min_keys{};
        std::array<CompositeKey, k> max_keys{};
        std::array<CompositeKey, k> min_pair_key_result{};
        std::array<CompositeKey, k> max_pair_key_result{};
        std::array<int, k>          min_pair_value_result{};
        std::array<int, k>          max_pair_value_result{};
        stream << keys_input.copy_from(luisa::span{keys}) << values_input.copy_from(luisa::span{values})
               << commands.commit() << min_keys_output.copy_to(luisa::span{min_keys})
               << max_keys_output.copy_to(luisa::span{max_keys})
               << min_pair_keys.copy_to(luisa::span{min_pair_key_result})
               << max_pair_keys.copy_to(luisa::span{max_pair_key_result})
               << min_pair_values.copy_to(luisa::span{min_pair_value_result})
               << max_pair_values.copy_to(luisa::span{max_pair_value_result}) << synchronize();

        std::sort(min_keys.begin(), min_keys.end(), composite_less);
        std::sort(max_keys.begin(), max_keys.end(), composite_less);
        std::array<CompositeKey, k> expected_min{{{0u, 7}, {1u, 9}}};
        std::array<CompositeKey, k> expected_max{{{2u, 5}, {3u, 0}}};
        for(size_t index = 0u; index < k; ++index)
        {
            expect(composite_equal(min_keys[index], expected_min[index]));
            expect(composite_equal(max_keys[index], expected_max[index]));
        }

        std::array<std::pair<std::pair<uint, int>, int>, k> min_pairs{};
        std::array<std::pair<std::pair<uint, int>, int>, k> max_pairs{};
        for(size_t index = 0u; index < k; ++index)
        {
            min_pairs[index] = {{min_pair_key_result[index].primary, min_pair_key_result[index].secondary},
                                min_pair_value_result[index]};
            max_pairs[index] = {{max_pair_key_result[index].primary, max_pair_key_result[index].secondary},
                                max_pair_value_result[index]};
        }
        std::sort(min_pairs.begin(), min_pairs.end());
        std::sort(max_pairs.begin(), max_pairs.end());
        const std::array<std::pair<std::pair<uint, int>, int>, k> expected_min_pairs{
            {{{0u, 7}, 7}, {{1u, 9}, 19}}};
        const std::array<std::pair<std::pair<uint, int>, int>, k> expected_max_pairs{
            {{{2u, 5}, 25}, {{3u, 0}, 30}}};
        for(size_t index = 0u; index < k; ++index)
        {
            expect(min_pairs[index] == expected_min_pairs[index]);
            expect(max_pairs[index] == expected_max_pairs[index]);
        }
    };

    "device_topk_caps_k_and_accepts_empty_work"_test = [&]
    {
        expect(DeviceTopK<>::GetTempStorageBytes<int>(3u, 7u) == 0u);
        expect(DeviceTopK<>::GetTempStorageBytes<CompositeKey, int>(3u, 7u) == 0u);
        const std::array<int, 3u> input{4, 2, 8};
        auto                      input_buffer = device.create_buffer<int>(input.size());
        auto                      output       = device.create_buffer<int>(input.size());
        CommandList               commands;
        topk.MaxKeys(commands, BufferView<uint>{}, input_buffer.view(), output.view(), input.size(), 7u);
        topk.MinKeys(commands, BufferView<uint>{}, BufferView<int>{}, BufferView<int>{}, 0u, 5u);
        topk.MaxKeys(commands, BufferView<uint>{}, BufferView<int>{}, BufferView<int>{}, 5u, 0u);

        std::array<int, 3u> result{};
        stream << input_buffer.copy_from(luisa::span{input}) << commands.commit()
               << output.copy_to(luisa::span{result}) << synchronize();
        std::sort(result.begin(), result.end());
        std::array<int, 3u> expected{2, 4, 8};
        for(size_t index = 0u; index < expected.size(); ++index)
        {
            expect(result[index] == expected[index]);
        }
    };
}
