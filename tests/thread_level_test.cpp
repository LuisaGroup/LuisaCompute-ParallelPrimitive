// Tests for thread-level load and store helpers.
// Covers portable cache modifiers, volatile access, and value preservation.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/core/logging.h>
#include <luisa/dsl/builtin.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
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

    constexpr uint item_count      = 257u;
    auto           input           = device.create_buffer<int>(item_count);
    auto           output          = device.create_buffer<int>(item_count);
    auto           volatile_output = device.create_buffer<int>(item_count);

    luisa::vector<int> host_input(item_count);
    luisa::vector<int> host_output(item_count);
    luisa::vector<int> host_volatile_output(item_count);
    for(uint i = 0u; i < item_count; ++i)
    {
        host_input[i] = static_cast<int>(i * 3u + 1u);
    }

    "thread_load_and_store_preserve_values_for_portable_cache_modifiers"_test = [&]
    {
        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> output_buffer,
                             BufferVar<int> volatile_output_buffer,
                             UInt           count) noexcept
        {
            set_block_size(128u);
            UInt index = dispatch_id().x;
            $if(index < count)
            {
                Int value = ThreadLoad<CacheLoadModifier::LOAD_DEFAULT>(input_buffer, index);
                ThreadStore<CacheStoreModifier::STORE_DEFAULT>(output_buffer, index, value + 7);

                Int volatile_value = ThreadLoad<CacheLoadModifier::LOAD_VOLATILE>(input_buffer, index);
                ThreadStore<CacheStoreModifier::STORE_VOLATILE>(volatile_output_buffer, index, volatile_value - 2);
            };
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << shader(input, output, volatile_output, item_count).dispatch(item_count)
               << output.copy_to(luisa::span{host_output})
               << volatile_output.copy_to(luisa::span{host_volatile_output}) << synchronize();

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(host_output[i] == host_input[i] + 7);
            expect(host_volatile_output[i] == host_input[i] - 2);
        }
    };
}
