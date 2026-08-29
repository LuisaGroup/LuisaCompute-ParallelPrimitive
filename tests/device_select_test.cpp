// Tests for DeviceSelect.
// Covers stable flagged, predicate, unique, and unique-by-key selection.

#include <lcpp/parallel_primitive.h>

#include <array>
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
}
