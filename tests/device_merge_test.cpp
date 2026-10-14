// Tests for DeviceMerge.
// Covers key and pair merging, duplicate keys, custom ordering, and empty inputs.

#include <lcpp/parallel_primitive.h>
#include <algorithm>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

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

    DeviceMerge<> merge;
    merge.create(device);

    "device_merge_keys_supports_default_and_custom_order"_test = [&]
    {
        const luisa::vector<int> ascending_a{1, 3, 3, 8};
        const luisa::vector<int> ascending_b{2, 3, 7};
        const luisa::vector<int> expected_ascending{1, 2, 3, 3, 3, 7, 8};
        const luisa::vector<int> descending_a{9, 7, 4};
        const luisa::vector<int> descending_b{8, 7, 1};
        const luisa::vector<int> expected_descending{9, 8, 7, 7, 4, 1};

        auto keys_a            = device.create_buffer<int>(ascending_a.size());
        auto keys_b            = device.create_buffer<int>(ascending_b.size());
        auto ascending_output  = device.create_buffer<int>(expected_ascending.size());
        auto descending_keys_a = device.create_buffer<int>(descending_a.size());
        auto descending_keys_b = device.create_buffer<int>(descending_b.size());
        auto descending_output = device.create_buffer<int>(expected_descending.size());
        stream << keys_a.copy_from(luisa::span{ascending_a}) << keys_b.copy_from(luisa::span{ascending_b})
               << descending_keys_a.copy_from(luisa::span{descending_a})
               << descending_keys_b.copy_from(luisa::span{descending_b});

        auto        greater = [](const Int& lhs, const Int& rhs) noexcept { return lhs > rhs; };
        CommandList commands;
        merge.MergeKeys(commands,
                        BufferView<uint>{},
                        keys_a.view(),
                        ascending_a.size(),
                        keys_b.view(),
                        ascending_b.size(),
                        ascending_output.view());
        merge.MergeKeys(commands,
                        BufferView<uint>{},
                        descending_keys_a.view(),
                        descending_a.size(),
                        descending_keys_b.view(),
                        descending_b.size(),
                        descending_output.view(),
                        greater);

        luisa::vector<int> ascending_result(expected_ascending.size());
        luisa::vector<int> descending_result(expected_descending.size());
        stream << commands.commit() << ascending_output.copy_to(luisa::span{ascending_result})
               << descending_output.copy_to(luisa::span{descending_result}) << synchronize();
        for(size_t index = 0u; index < expected_ascending.size(); ++index)
        {
            expect(ascending_result[index] == expected_ascending[index]);
        }
        for(size_t index = 0u; index < expected_descending.size(); ++index)
        {
            expect(descending_result[index] == expected_descending[index]);
        }
    };

    "device_merge_tiled_boundaries_duplicates_and_skew"_test = [&]
    {
        for(size_t count : {2047u, 2048u, 2049u, 70003u})
        {
            for(size_t left_count : {size_t{1u}, count / 2u, count - 1u})
            {
                for(bool descending : {false, true})
                {
                    luisa::vector<int> left(left_count);
                    luisa::vector<int> right(count - left_count);
                    for(size_t i = 0u; i < left.size(); ++i)
                    {
                        left[i] = static_cast<int>(i / 7u);
                    }
                    for(size_t i = 0u; i < right.size(); ++i)
                    {
                        right[i] = static_cast<int>(i / 3u);
                    }
                    if(descending)
                    {
                        std::reverse(left.begin(), left.end());
                        std::reverse(right.begin(), right.end());
                    }
                    luisa::vector<int> expected(count);
                    std::merge(left.begin(),
                               left.end(),
                               right.begin(),
                               right.end(),
                               expected.begin(),
                               [descending](int lhs, int rhs)
                               { return descending ? lhs > rhs : lhs < rhs; });
                    auto        in_left  = device.create_buffer<int>(left.size());
                    auto        in_right = device.create_buffer<int>(right.size());
                    auto        out      = device.create_buffer<int>(count);
                    CommandList commands;
                    if(descending)
                    {
                        auto greater = [](const Int& lhs, const Int& rhs) noexcept
                        { return lhs > rhs; };
                        merge.MergeKeys(commands,
                                        BufferView<uint>{},
                                        in_left.view(),
                                        left.size(),
                                        in_right.view(),
                                        right.size(),
                                        out.view(),
                                        greater);
                    }
                    else
                    {
                        merge.MergeKeys(commands,
                                        BufferView<uint>{},
                                        in_left.view(),
                                        left.size(),
                                        in_right.view(),
                                        right.size(),
                                        out.view());
                    }
                    luisa::vector<int> result(count);
                    stream << in_left.copy_from(luisa::span{left})
                           << in_right.copy_from(luisa::span{right}) << commands.commit()
                           << out.copy_to(luisa::span{result}) << synchronize();
                    expect(result == expected);
                }
            }
        }
    };

    "device_merge_pairs_preserves_key_value_association"_test = [&]
    {
        const luisa::vector<int> keys_a_values{1, 3, 3, 8};
        const luisa::vector<int> values_a_values{10, 30, 31, 80};
        const luisa::vector<int> keys_b_values{2, 3, 7};
        const luisa::vector<int> values_b_values{20, 32, 70};
        const luisa::vector<int> expected_keys{1, 2, 3, 3, 3, 7, 8};
        const luisa::vector<int> expected_values{10, 20, 30, 31, 32, 70, 80};

        auto keys_a        = device.create_buffer<int>(keys_a_values.size());
        auto values_a      = device.create_buffer<int>(values_a_values.size());
        auto keys_b        = device.create_buffer<int>(keys_b_values.size());
        auto values_b      = device.create_buffer<int>(values_b_values.size());
        auto keys_output   = device.create_buffer<int>(expected_keys.size());
        auto values_output = device.create_buffer<int>(expected_values.size());
        stream << keys_a.copy_from(luisa::span{keys_a_values})
               << values_a.copy_from(luisa::span{values_a_values})
               << keys_b.copy_from(luisa::span{keys_b_values})
               << values_b.copy_from(luisa::span{values_b_values});

        CommandList commands;
        merge.MergePairs(commands,
                         BufferView<uint>{},
                         keys_a.view(),
                         values_a.view(),
                         keys_a_values.size(),
                         keys_b.view(),
                         values_b.view(),
                         keys_b_values.size(),
                         keys_output.view(),
                         values_output.view());

        luisa::vector<int> keys_result(expected_keys.size());
        luisa::vector<int> values_result(expected_values.size());
        stream << commands.commit() << keys_output.copy_to(luisa::span{keys_result})
               << values_output.copy_to(luisa::span{values_result}) << synchronize();
        for(size_t index = 0u; index < expected_keys.size(); ++index)
        {
            expect(keys_result[index] == expected_keys[index]);
            expect(values_result[index] == expected_values[index]);
        }
    };

    "device_merge_handles_empty_inputs_without_scratch"_test = [&]
    {
        expect(DeviceMerge<>::GetTempStorageBytes<int>(0u, 2u) == 0u);
        expect(DeviceMerge<>::GetTempStorageBytes<int, int>(2u, 0u) == 0u);
        const luisa::vector<int> keys_values{2, 4};
        const luisa::vector<int> pair_values{20, 40};
        auto                     keys             = device.create_buffer<int>(keys_values.size());
        auto                     values           = device.create_buffer<int>(pair_values.size());
        auto                     keys_output      = device.create_buffer<int>(keys_values.size());
        auto                     pair_keys_output = device.create_buffer<int>(keys_values.size());
        auto                     values_output    = device.create_buffer<int>(pair_values.size());
        stream << keys.copy_from(luisa::span{keys_values}) << values.copy_from(luisa::span{pair_values});

        CommandList commands;
        merge.MergeKeys(commands,
                        BufferView<uint>{},
                        BufferView<int>{},
                        0u,
                        keys.view(),
                        keys_values.size(),
                        keys_output.view());
        merge.MergePairs(commands,
                         BufferView<uint>{},
                         keys.view(),
                         values.view(),
                         keys_values.size(),
                         BufferView<int>{},
                         BufferView<int>{},
                         0u,
                         pair_keys_output.view(),
                         values_output.view());
        merge.MergeKeys(commands, BufferView<uint>{}, BufferView<int>{}, 0u, BufferView<int>{}, 0u, BufferView<int>{});

        luisa::vector<int> keys_result(keys_values.size());
        luisa::vector<int> pair_keys_result(keys_values.size());
        luisa::vector<int> values_result(pair_values.size());
        stream << commands.commit() << keys_output.copy_to(luisa::span{keys_result})
               << pair_keys_output.copy_to(luisa::span{pair_keys_result})
               << values_output.copy_to(luisa::span{values_result}) << synchronize();
        for(size_t index = 0u; index < keys_values.size(); ++index)
        {
            expect(keys_result[index] == keys_values[index]);
            expect(pair_keys_result[index] == keys_values[index]);
            expect(values_result[index] == pair_values[index]);
        }
    };
}
