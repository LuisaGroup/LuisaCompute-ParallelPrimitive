// Tests for DeviceFor.
// Covers bounded dispatch and lowercase compatibility entry points.

#include <lcpp/parallel_primitive.h>

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

    DeviceFor device_for;
    device_for.create(device);

    "devicefor_mutates_exactly_the_requested_prefix"_test = [&]
    {
        constexpr uint     item_count   = 301u;
        constexpr uint     prefix_count = 173u;
        auto               buffer       = device.create_buffer<int>(item_count);
        luisa::vector<int> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            values[i] = static_cast<int>(i) - 150;
        }

        Callable<void(int&)> affine = [](Int& value) noexcept { value = value * 3 + 7; };

        CommandList commands;
        device_for.ForEachN(commands, buffer.view(), prefix_count, affine);
        stream << buffer.copy_from(luisa::span{values}) << commands.commit()
               << buffer.copy_to(luisa::span{values}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            int initial  = static_cast<int>(i) - 150;
            int expected = i < prefix_count ? initial * 3 + 7 : initial;
            expect(values[i] == expected);
        }
        expect(DeviceFor::GetTempStorageBytes(item_count) == 0u);
    };

    "devicefor_lowercase_compatibility_entry_points_execute_work"_test = [&]
    {
        constexpr uint     item_count = 193u;
        auto               buffer     = device.create_buffer<int>(item_count);
        luisa::vector<int> values(item_count);
        for(uint i = 0u; i < item_count; ++i)
        {
            values[i] = static_cast<int>(i);
        }

        Callable<void(int&)> twice     = [](Int& value) noexcept { value *= 2; };
        Callable<void(int&)> add_three = [](Int& value) noexcept { value += 3; };
        Callable<void(int&)> negate    = [](Int& value) noexcept { value = -value; };

        CommandList commands;
        device_for.for_each_n(commands, buffer.view(), 129u, twice);
        device_for.for_each_extents(commands, buffer.view(), 65u, add_three);
        device_for.for_each(commands, buffer.view(), item_count, negate);
        stream << buffer.copy_from(luisa::span{values}) << commands.commit()
               << buffer.copy_to(luisa::span{values}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            int expected = static_cast<int>(i);
            if(i < 129u)
            {
                expected *= 2;
            }
            if(i < 65u)
            {
                expected += 3;
            }
            expected = -expected;
            expect(values[i] == expected);
        }
    };
}
