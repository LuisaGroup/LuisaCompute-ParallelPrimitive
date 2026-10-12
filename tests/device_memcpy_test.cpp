// Tests for DeviceMemcpy.
// Covers raw byte counts, typed backing buffers, aliased inputs, and empty batches.

#include <lcpp/parallel_primitive.h>

#include <array>
#include <boost/ut.hpp>
#include <cstring>
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

    DeviceMemcpy<> memcpy;
    memcpy.create(device);

    "device_memcpy_batched_copies_independent_byte_ranges"_test = [&]
    {
        constexpr uint         range_count = 3u;
        std::array<ubyte, 12u> source{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u};
        std::array<ubyte, 20u> destination;
        destination.fill(0xeeu);
        std::array<uint, range_count> byte_counts{5u, 3u, 4u};

        auto source_buffer      = device.create_buffer<ubyte>(source.size());
        auto destination_buffer = device.create_buffer<ubyte>(destination.size());
        auto byte_counts_buffer = device.create_buffer<uint>(range_count);
        auto input_ranges       = device.create_bindless_array(range_count);
        auto output_ranges      = device.create_bindless_array(range_count);
        input_ranges.emplace_on_update(0u, source_buffer.view().subview(1u, 5u));
        input_ranges.emplace_on_update(1u, source_buffer.view().subview(7u, 3u));
        input_ranges.emplace_on_update(2u, source_buffer.view().subview(1u, 5u));
        output_ranges.emplace_on_update(0u, destination_buffer.view().subview(2u, 5u));
        output_ranges.emplace_on_update(1u, destination_buffer.view().subview(10u, 3u));
        output_ranges.emplace_on_update(2u, destination_buffer.view().subview(15u, 4u));

        CommandList commands;
        memcpy.Batched(commands, BufferView<uint>{}, input_ranges, output_ranges, byte_counts_buffer.view(), range_count);
        stream << source_buffer.copy_from(luisa::span{source})
               << destination_buffer.copy_from(luisa::span{destination})
               << byte_counts_buffer.copy_from(luisa::span{byte_counts}) << input_ranges.update()
               << output_ranges.update() << commands.commit()
               << destination_buffer.copy_to(luisa::span{destination}) << synchronize();

        std::array<ubyte, 20u> expected;
        expected.fill(0xeeu);
        std::memcpy(expected.data() + 2u, source.data() + 1u, 5u);
        std::memcpy(expected.data() + 10u, source.data() + 7u, 3u);
        std::memcpy(expected.data() + 15u, source.data() + 1u, 4u);
        for(size_t index = 0u; index < expected.size(); ++index)
        {
            expect(destination[index] == expected[index]);
        }
    };

    "device_memcpy_uses_byte_counts_for_typed_backing_buffers"_test = [&]
    {
        constexpr uint       byte_count = 7u;
        std::array<uint, 2u> source{0x04030201u, 0x08070605u};
        std::array<uint, 2u> destination{0xccccccccu, 0xccccccccu};
        auto                 source_buffer      = device.create_buffer<uint>(source.size());
        auto                 destination_buffer = device.create_buffer<uint>(destination.size());
        auto                 byte_count_buffer  = device.create_buffer<uint>(1u);
        auto                 input_ranges       = device.create_bindless_array(1u);
        auto                 output_ranges      = device.create_bindless_array(1u);
        input_ranges.emplace_on_update(0u, source_buffer.view());
        output_ranges.emplace_on_update(0u, destination_buffer.view());

        CommandList commands;
        memcpy.Batched(commands, BufferView<uint>{}, input_ranges, output_ranges, byte_count_buffer.view(), 1u);
        stream << source_buffer.copy_from(luisa::span{source})
               << destination_buffer.copy_from(luisa::span{destination})
               << byte_count_buffer.copy_from(luisa::span{&byte_count, 1u}) << input_ranges.update()
               << output_ranges.update() << commands.commit()
               << destination_buffer.copy_to(luisa::span{destination}) << synchronize();

        std::array<ubyte, sizeof(destination)> expected_bytes{};
        std::array<ubyte, sizeof(destination)> result_bytes{};
        std::memset(expected_bytes.data(), 0xcc, expected_bytes.size());
        std::memcpy(expected_bytes.data(), source.data(), byte_count);
        std::memcpy(result_bytes.data(), destination.data(), result_bytes.size());
        for(size_t index = 0u; index < expected_bytes.size(); ++index)
        {
            expect(result_bytes[index] == expected_bytes[index]);
        }
    };

    "device_memcpy_accepts_empty_batches"_test = [&]
    {
        expect(DeviceMemcpy<>::GetTempStorageBytes(0u) == 0u);
        CommandList commands;
        memcpy.Batched(commands, BufferView<uint>{}, BindlessArray{}, BindlessArray{}, BufferView<uint>{}, 0u);
        stream << commands.commit() << synchronize();
    };
}
