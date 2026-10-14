// Tests for DeviceSelect.
// Covers stable flagged, predicate, unique, and unique-by-key selection.

#include <lcpp/parallel_primitive.h>

#include <array>
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

    DeviceSelect<> selector;
    selector.create(device);

    "deviceselect_preserves_wide_values_and_boolean_flags"_test = [&]
    {
        constexpr uint                  count = 17u;
        std::array<luisa::ulong, count> values{};
        std::array<bool, count>         flags{};
        luisa::vector<luisa::ulong>     expected_flagged, expected_if, expected_unique;
        for(uint i = 0u; i < count; ++i)
        {
            values[i] = luisa::ulong{1} << 40u | luisa::ulong(i / 3u);
            flags[i]  = i % 3u == 1u;
            if(flags[i])
            {
                expected_flagged.emplace_back(values[i]);
            }
            if(values[i] % 2u == 0u)
            {
                expected_if.emplace_back(values[i]);
            }
            if(i % 3u == 0u)
            {
                expected_unique.emplace_back(values[i]);
            }
        }
        auto in       = device.create_buffer<luisa::ulong>(count);
        auto flag     = device.create_buffer<bool>(count);
        auto out      = device.create_buffer<luisa::ulong>(count);
        auto selected = device.create_buffer<uint>(1u);
        auto temp =
            device.create_buffer<uint>(bytes_to_uint_count(DeviceSelect<>::GetTempStorageBytes(count)));
        Callable<bool(luisa::ulong)> even = [](Var<luisa::ulong> value) noexcept
        { return value % luisa::ulong{2} == luisa::ulong{0}; };
        stream << in.copy_from(luisa::span{values}) << flag.copy_from(luisa::span{flags});
        for(uint mode = 0u; mode < 3u; ++mode)
        {
            CommandList commands;
            if(mode == 0u)
            {
                selector.Flagged(commands, temp.view(), in.view(), flag.view(), out.view(), selected.view(), count);
            }
            else if(mode == 1u)
            {
                selector.If(commands, temp.view(), in.view(), out.view(), selected.view(), count, even);
            }
            else
            {
                selector.Unique(commands, temp.view(), in.view(), out.view(), selected.view(), count);
            }
            luisa::vector<luisa::ulong> result(count);
            uint                        size = 0u;
            stream << commands.commit() << out.copy_to(luisa::span{result})
                   << selected.copy_to(luisa::span{&size, 1u}) << synchronize();
            const auto& expected = mode == 0u ? expected_flagged :
                                   mode == 1u ? expected_if :
                                                expected_unique;
            expect(size == expected.size());
            result.resize(expected.size());
            expect(result == expected);
        }
    };

    "deviceselect_fused_multitile_tails_and_empty_tiles"_test = [&]
    {
        for(uint count : {1u, 5759u, 5760u, 5761u, 230413u})
        {
            luisa::vector<int>  input(count + 3u);
            luisa::vector<uint> flags(count);
            auto                input_buffer  = device.create_buffer<int>(input.size());
            auto                flag_buffer   = device.create_buffer<uint>(count);
            auto                output_buffer = device.create_buffer<int>(count + 2u);
            auto                count_buffer  = device.create_buffer<uint>(2u);
            auto                bytes         = DeviceSelect<>::GetTempStorageBytes(count);
            auto scratch = device.create_buffer<uint>(bytes_to_uint_count(bytes) + 64u);
            auto temp    = scratch.view().subview(64u, bytes_to_uint_count(bytes));
            luisa::vector<uint> poison(scratch.size(), 99u);
            for(uint pattern = 0u; pattern < 4u; ++pattern)
            {
                luisa::vector<int> expected;
                for(uint i = 0u; i < count; ++i)
                {
                    input[i + 3u] = static_cast<int>(i);
                    flags[i]      = pattern == 0u ? 0u :
                                    pattern == 1u ? 1u :
                                    pattern == 2u ? i % 3u :
                                                    uint(i < 11u || i > 200000u);
                    if(flags[i] != 0u)
                    {
                        expected.emplace_back(input[i + 3u]);
                    }
                }
                CommandList commands;
                selector.Flagged(commands,
                                 temp,
                                 input_buffer.view().subview(3u, count),
                                 flag_buffer.view(),
                                 output_buffer.view().subview(2u, count),
                                 count_buffer.view().subview(1u, 1u),
                                 count);
                stream << scratch.copy_from(luisa::span{poison})
                       << input_buffer.copy_from(luisa::span{input})
                       << flag_buffer.copy_from(luisa::span{flags}) << commands.commit();
                luisa::vector<int> result(count + 2u);
                uint               selected = 0u;
                stream << output_buffer.copy_to(luisa::span{result})
                       << count_buffer.view().subview(1u, 1u).copy_to(luisa::span{&selected, 1u})
                       << synchronize();
                expect(selected == expected.size());
                expect(std::equal(expected.begin(), expected.end(), result.begin() + 2u));
            }

            for(uint run_length : {1u, 17u, 18001u, count})
            {
                luisa::vector<int> expected;
                for(uint i = 0u; i < count; ++i)
                {
                    input[i + 3u] = static_cast<int>(i / run_length);
                    if(i % run_length == 0u)
                    {
                        expected.emplace_back(input[i + 3u]);
                    }
                }
                CommandList commands;
                selector.Unique(commands,
                                temp,
                                input_buffer.view().subview(3u, count),
                                output_buffer.view().subview(2u, count),
                                count_buffer.view().subview(1u, 1u),
                                count);
                stream << scratch.copy_from(luisa::span{poison})
                       << input_buffer.copy_from(luisa::span{input}) << commands.commit();
                luisa::vector<int> result(count + 2u);
                uint               selected = 0u;
                stream << output_buffer.copy_to(luisa::span{result})
                       << count_buffer.view().subview(1u, 1u).copy_to(luisa::span{&selected, 1u})
                       << synchronize();
                expect(selected == expected.size());
                expect(std::equal(expected.begin(), expected.end(), result.begin() + 2u));
            }
        }
    };

    "deviceselect_flagged_and_if_are_stable"_test = [&]
    {
        constexpr uint      count = 17u;
        luisa::vector<int>  input(count);
        luisa::vector<uint> flags(count);
        for(uint i = 0u; i < count; ++i)
        {
            input[i] = static_cast<int>(100u + i);
            flags[i] = (i % 3u) == 1u ? 7u : 0u;
        }

        auto input_buffer   = device.create_buffer<int>(count);
        auto flag_buffer    = device.create_buffer<uint>(count);
        auto flagged_output = device.create_buffer<int>(count);
        auto if_output      = device.create_buffer<int>(count);
        auto flagged_count  = device.create_buffer<uint>(1u);
        auto if_count       = device.create_buffer<uint>(1u);
        auto temp_bytes     = DeviceSelect<>::GetTempStorageBytes(count);
        auto temp_storage   = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        Callable<bool(int)> is_even = [](Int value) noexcept { return value % 2 == 0; };

        CommandList commands;
        selector.Flagged(commands,
                         temp_storage.view(),
                         input_buffer.view(),
                         flag_buffer.view(),
                         flagged_output.view(),
                         flagged_count.view(),
                         count);
        selector.If(commands, temp_storage.view(), input_buffer.view(), if_output.view(), if_count.view(), count, is_even);
        stream << input_buffer.copy_from(luisa::span{input})
               << flag_buffer.copy_from(luisa::span{flags}) << commands.commit();

        luisa::vector<int> flagged_result(count, -1);
        luisa::vector<int> if_result(count, -1);
        uint               flagged_size = 0u;
        uint               if_size      = 0u;
        stream << flagged_output.copy_to(luisa::span{flagged_result})
               << if_output.copy_to(luisa::span{if_result})
               << flagged_count.copy_to(luisa::span{&flagged_size, 1u})
               << if_count.copy_to(luisa::span{&if_size, 1u}) << synchronize();

        luisa::vector<int> expected_flagged;
        luisa::vector<int> expected_if;
        for(uint i = 0u; i < count; ++i)
        {
            if(flags[i] != 0u)
            {
                expected_flagged.emplace_back(input[i]);
            }
            if(input[i] % 2 == 0)
            {
                expected_if.emplace_back(input[i]);
            }
        }
        expect(flagged_size == expected_flagged.size());
        expect(if_size == expected_if.size());
        for(uint i = 0u; i < flagged_size; ++i)
        {
            expect(flagged_result[i] == expected_flagged[i]);
        }
        for(uint i = 0u; i < if_size; ++i)
        {
            expect(if_result[i] == expected_if[i]);
        }
    };

    "deviceselect_unique_and_uniquebykey_keep_first_run_items"_test = [&]
    {
        luisa::vector<int> keys{1, 1, 2, 2, 2, 4, 7, 7, 9};
        luisa::vector<int> values{10, 11, 20, 21, 22, 40, 70, 71, 90};
        const uint         count               = static_cast<uint>(keys.size());
        auto               key_input           = device.create_buffer<int>(count);
        auto               value_input         = device.create_buffer<int>(count);
        auto               unique_output       = device.create_buffer<int>(count);
        auto               unique_key_output   = device.create_buffer<int>(count);
        auto               unique_value_output = device.create_buffer<int>(count);
        auto               unique_count        = device.create_buffer<uint>(1u);
        auto               unique_by_key_count = device.create_buffer<uint>(1u);
        auto               temp_bytes          = DeviceSelect<>::GetTempStorageBytes(count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

        CommandList commands;
        selector.Unique(
            commands, temp_storage.view(), key_input.view(), unique_output.view(), unique_count.view(), count);
        selector.UniqueByKey(commands,
                             temp_storage.view(),
                             key_input.view(),
                             value_input.view(),
                             unique_key_output.view(),
                             unique_value_output.view(),
                             unique_by_key_count.view(),
                             count);
        stream << key_input.copy_from(luisa::span{keys})
               << value_input.copy_from(luisa::span{values}) << commands.commit();

        luisa::vector<int> unique_result(count);
        luisa::vector<int> unique_keys(count);
        luisa::vector<int> unique_values(count);
        uint               unique_size        = 0u;
        uint               unique_by_key_size = 0u;
        stream << unique_output.copy_to(luisa::span{unique_result})
               << unique_key_output.copy_to(luisa::span{unique_keys})
               << unique_value_output.copy_to(luisa::span{unique_values})
               << unique_count.copy_to(luisa::span{&unique_size, 1u})
               << unique_by_key_count.copy_to(luisa::span{&unique_by_key_size, 1u}) << synchronize();

        std::array<int, 5u> expected_keys{1, 2, 4, 7, 9};
        std::array<int, 5u> expected_values{10, 20, 40, 70, 90};
        expect(unique_size == expected_keys.size());
        expect(unique_by_key_size == expected_keys.size());
        for(uint i = 0u; i < expected_keys.size(); ++i)
        {
            expect(unique_result[i] == expected_keys[i]);
            expect(unique_keys[i] == expected_keys[i]);
            expect(unique_values[i] == expected_values[i]);
        }
    };

    "deviceselect_flaggedif_supports_copy_and_in_place_compaction"_test = [&]
    {
        constexpr uint         count = 9u;
        std::array<int, count> input{100, 101, 102, 103, 104, 105, 106, 107, 108};
        std::array<int, count> flags{-3, -2, -1, 0, 1, 2, 3, 4, 5};
        auto                   input_buffer   = device.create_buffer<int>(count);
        auto                   in_place_data  = device.create_buffer<int>(count);
        auto                   flags_buffer   = device.create_buffer<int>(count);
        auto                   copy_output    = device.create_buffer<int>(count);
        auto                   copy_count     = device.create_buffer<uint>(1u);
        auto                   in_place_count = device.create_buffer<uint>(1u);
        auto                   temp_bytes     = DeviceSelect<>::GetTempStorageBytes(count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
        Callable<bool(int)> positive_odd = [](Int flag) noexcept
        { return flag > 0 & (flag % 2 != 0); };

        CommandList commands;
        selector.FlaggedIf(commands,
                           temp_storage.view(),
                           input_buffer.view(),
                           flags_buffer.view(),
                           copy_output.view(),
                           copy_count.view(),
                           count,
                           positive_odd);
        selector.FlaggedIf(
            commands, temp_storage.view(), in_place_data.view(), flags_buffer.view(), in_place_count.view(), count, positive_odd);
        stream << input_buffer.copy_from(luisa::span{input}) << in_place_data.copy_from(luisa::span{input})
               << flags_buffer.copy_from(luisa::span{flags}) << commands.commit();

        std::array<int, count> copy_result{};
        std::array<int, count> in_place_result{};
        std::array<int, count> input_after{};
        std::array<int, count> flags_after{};
        uint                   copy_size     = 0u;
        uint                   in_place_size = 0u;
        stream << copy_output.copy_to(luisa::span{copy_result})
               << in_place_data.copy_to(luisa::span{in_place_result})
               << input_buffer.copy_to(luisa::span{input_after})
               << flags_buffer.copy_to(luisa::span{flags_after})
               << copy_count.copy_to(luisa::span{&copy_size, 1u})
               << in_place_count.copy_to(luisa::span{&in_place_size, 1u}) << synchronize();

        constexpr std::array<int, 3u> expected{104, 106, 108};
        expect(copy_size == expected.size());
        expect(in_place_size == expected.size());
        for(size_t i = 0u; i < expected.size(); ++i)
        {
            expect(copy_result[i] == expected[i]);
            expect(in_place_result[i] == expected[i]);
        }
        for(size_t i = 0u; i < input.size(); ++i)
        {
            expect(input_after[i] == input[i]);
            expect(flags_after[i] == flags[i]);
        }
    };

    "deviceselect_in_place_flagged_if_and_unique_are_stable"_test = [&]
    {
        constexpr uint         count = 8u;
        std::array<int, count> aliased_flags{0, 2, 0, 3, 4, 0, 5, 0};
        std::array<int, count> predicate_data{-3, 4, -1, 6, 0, 2, 7, -8};
        std::array<int, count> unique_data{1, 3, 2, 4, 5, 7, 8, 10};
        std::array<int, count> default_unique_data{1, 1, 2, 2, 2, 4, 4, 9};
        auto                   aliased_buffer        = device.create_buffer<int>(count);
        auto                   predicate_buffer      = device.create_buffer<int>(count);
        auto                   unique_buffer         = device.create_buffer<int>(count);
        auto                   default_unique_buffer = device.create_buffer<int>(count);
        auto                   flagged_count         = device.create_buffer<uint>(1u);
        auto                   predicate_count       = device.create_buffer<uint>(1u);
        auto                   unique_count          = device.create_buffer<uint>(1u);
        auto                   default_unique_count  = device.create_buffer<uint>(1u);
        auto                   zero_count            = device.create_buffer<uint>(1u);
        auto                   temp_bytes            = DeviceSelect<>::GetTempStorageBytes(count);
        auto temp_storage = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
        Callable<bool(int)>      positive    = [](Int value) noexcept { return value > 0; };
        Callable<bool(int, int)> same_parity = [](Int lhs, Int rhs) noexcept
        { return lhs % 2 == rhs % 2; };

        CommandList commands;
        selector.Flagged(commands,
                         temp_storage.view(),
                         aliased_buffer.view(),
                         aliased_buffer.view(),
                         flagged_count.view(),
                         count);
        selector.If(commands, temp_storage.view(), predicate_buffer.view(), predicate_count.view(), count, positive);
        selector.Unique(commands, temp_storage.view(), unique_buffer.view(), unique_count.view(), count, same_parity);
        selector.Unique(
            commands, temp_storage.view(), default_unique_buffer.view(), default_unique_count.view(), count);
        selector.If(commands, BufferView<uint>{}, BufferView<int>{}, zero_count.view(), 0u, positive);
        stream << aliased_buffer.copy_from(luisa::span{aliased_flags})
               << predicate_buffer.copy_from(luisa::span{predicate_data})
               << unique_buffer.copy_from(luisa::span{unique_data})
               << default_unique_buffer.copy_from(luisa::span{default_unique_data}) << commands.commit();

        std::array<int, count> flagged_result{};
        std::array<int, count> predicate_result{};
        std::array<int, count> unique_result{};
        std::array<int, count> default_unique_result{};
        uint                   flagged_size        = 0u;
        uint                   predicate_size      = 0u;
        uint                   unique_size         = 0u;
        uint                   default_unique_size = 0u;
        uint                   zero_size           = 1u;
        stream << aliased_buffer.copy_to(luisa::span{flagged_result})
               << predicate_buffer.copy_to(luisa::span{predicate_result})
               << unique_buffer.copy_to(luisa::span{unique_result})
               << default_unique_buffer.copy_to(luisa::span{default_unique_result})
               << flagged_count.copy_to(luisa::span{&flagged_size, 1u})
               << predicate_count.copy_to(luisa::span{&predicate_size, 1u})
               << unique_count.copy_to(luisa::span{&unique_size, 1u})
               << default_unique_count.copy_to(luisa::span{&default_unique_size, 1u})
               << zero_count.copy_to(luisa::span{&zero_size, 1u}) << synchronize();

        constexpr std::array<int, 4u> expected_flagged{2, 3, 4, 5};
        constexpr std::array<int, 4u> expected_predicate{4, 6, 2, 7};
        constexpr std::array<int, 4u> expected_unique{1, 2, 5, 8};
        constexpr std::array<int, 4u> expected_default_unique{1, 2, 4, 9};
        expect(flagged_size == expected_flagged.size());
        expect(predicate_size == expected_predicate.size());
        expect(unique_size == expected_unique.size());
        expect(default_unique_size == expected_default_unique.size());
        expect(zero_size == 0u);
        for(size_t i = 0u; i < expected_flagged.size(); ++i)
        {
            expect(flagged_result[i] == expected_flagged[i]);
            expect(predicate_result[i] == expected_predicate[i]);
            expect(unique_result[i] == expected_unique[i]);
            expect(default_unique_result[i] == expected_default_unique[i]);
        }
    };
}
