// Tests for WarpLoad and WarpStore.
// Covers blocked, striped, transpose, guarded, and default-value paths.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/dsl/builtin.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint warp_size           = 32u;
constexpr uint items_per_thread    = 4u;
constexpr uint thread_count        = 256u;
constexpr uint item_count          = thread_count * items_per_thread;
constexpr uint warp_items          = warp_size * items_per_thread;
constexpr uint guarded_valid_items = 37u;
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

    auto input            = device.create_buffer<int>(item_count);
    auto direct_loaded    = device.create_buffer<int>(item_count);
    auto striped_loaded   = device.create_buffer<int>(item_count);
    auto transpose_loaded = device.create_buffer<int>(item_count);
    auto direct_stored    = device.create_buffer<int>(item_count);
    auto striped_stored   = device.create_buffer<int>(item_count);
    auto transpose_stored = device.create_buffer<int>(item_count);

    luisa::vector<int> host_input(item_count);
    luisa::vector<int> host_direct_loaded(item_count);
    luisa::vector<int> host_striped_loaded(item_count);
    luisa::vector<int> host_transpose_loaded(item_count);
    luisa::vector<int> host_direct_stored(item_count);
    luisa::vector<int> host_striped_stored(item_count);
    luisa::vector<int> host_transpose_stored(item_count);
    for(uint i = 0u; i < item_count; ++i)
    {
        host_input[i] = static_cast<int>(i * 5u + 11u);
    }

    "warp_load_and_store_honor_blocked_striped_and_transpose_layouts"_test = [&]
    {
        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> direct_loaded_buffer,
                             BufferVar<int> striped_loaded_buffer,
                             BufferVar<int> transpose_loaded_buffer,
                             BufferVar<int> direct_stored_buffer,
                             BufferVar<int> striped_stored_buffer,
                             BufferVar<int> transpose_stored_buffer) noexcept
        {
            set_block_size(thread_count);
            UInt thread_index = dispatch_id().x;
            UInt lane         = warp_lane_id();
            UInt warp_begin   = thread_index / warp_size * warp_items;

            ArrayVar<int, items_per_thread> direct_items;
            ArrayVar<int, items_per_thread> striped_items;
            ArrayVar<int, items_per_thread> transpose_items;

            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_DIRECT, warp_size>().Load(
                input_buffer, direct_items, warp_begin);
            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_STRIPED, warp_size>().Load(
                input_buffer, striped_items, warp_begin);
            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE, warp_size>().Load(
                input_buffer, transpose_items, warp_begin);

            ArrayVar<int, items_per_thread> blocked_store_items;
            ArrayVar<int, items_per_thread> striped_store_items;
            for(uint i = 0u; i < items_per_thread; ++i)
            {
                UInt thread_item = thread_index * items_per_thread + i;
                direct_loaded_buffer.write(thread_item, direct_items[i]);
                striped_loaded_buffer.write(thread_item, striped_items[i]);
                transpose_loaded_buffer.write(thread_item, transpose_items[i]);

                blocked_store_items[i] = cast<int>(warp_begin + lane * items_per_thread + i);
                striped_store_items[i] = cast<int>(warp_begin + lane + i * warp_size);
            }

            WarpStore<int, items_per_thread, WarpStoreAlgorithm::WARP_STORE_DIRECT, warp_size>().Store(
                blocked_store_items, direct_stored_buffer, warp_begin);
            WarpStore<int, items_per_thread, WarpStoreAlgorithm::WARP_STORE_STRIPED, warp_size>().Store(
                striped_store_items, striped_stored_buffer, warp_begin);
            WarpStore<int, items_per_thread, WarpStoreAlgorithm::WARP_STORE_TRANSPOSE, warp_size>().Store(
                blocked_store_items, transpose_stored_buffer, warp_begin);
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << shader(input, direct_loaded, striped_loaded, transpose_loaded, direct_stored, striped_stored, transpose_stored)
                      .dispatch(thread_count)
               << direct_loaded.copy_to(luisa::span{host_direct_loaded})
               << striped_loaded.copy_to(luisa::span{host_striped_loaded})
               << transpose_loaded.copy_to(luisa::span{host_transpose_loaded})
               << direct_stored.copy_to(luisa::span{host_direct_stored})
               << striped_stored.copy_to(luisa::span{host_striped_stored})
               << transpose_stored.copy_to(luisa::span{host_transpose_stored}) << synchronize();

        for(uint thread = 0u; thread < thread_count; ++thread)
        {
            uint lane       = thread % warp_size;
            uint warp_begin = thread / warp_size * warp_items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                uint thread_item   = thread * items_per_thread + item;
                uint blocked_index = warp_begin + lane * items_per_thread + item;
                uint striped_index = warp_begin + lane + item * warp_size;
                expect(host_direct_loaded[thread_item] == host_input[blocked_index]);
                expect(host_striped_loaded[thread_item] == host_input[striped_index]);
                expect(host_transpose_loaded[thread_item] == host_input[blocked_index]);
            }
        }

        for(uint i = 0u; i < item_count; ++i)
        {
            expect(host_direct_stored[i] == static_cast<int>(i));
            expect(host_striped_stored[i] == static_cast<int>(i));
            expect(host_transpose_stored[i] == static_cast<int>(i));
        }
    };

    "warp_load_and_store_honor_guarded_item_counts"_test = [&]
    {
        auto guarded_input        = device.create_buffer<int>(warp_items);
        auto guarded_no_default   = device.create_buffer<int>(warp_items);
        auto guarded_transpose    = device.create_buffer<int>(warp_items);
        auto guarded_with_default = device.create_buffer<int>(warp_items);
        auto guarded_stored       = device.create_buffer<int>(warp_items);

        luisa::vector<int> host_guarded_input(warp_items);
        luisa::vector<int> host_guarded_no_default(warp_items);
        luisa::vector<int> host_guarded_transpose(warp_items);
        luisa::vector<int> host_guarded_with_default(warp_items);
        luisa::vector<int> host_guarded_stored(warp_items, -11);
        for(uint i = 0u; i < warp_items; ++i)
        {
            host_guarded_input[i] = static_cast<int>(i * 3u + 5u);
        }

        Kernel1D guarded_kernel = [](BufferVar<int> input_buffer,
                                     BufferVar<int> no_default_buffer,
                                     BufferVar<int> transpose_buffer,
                                     BufferVar<int> with_default_buffer,
                                     BufferVar<int> stored_buffer) noexcept
        {
            set_block_size(warp_size);
            UInt lane = warp_lane_id();

            ArrayVar<int, items_per_thread> no_default_items;
            ArrayVar<int, items_per_thread> transpose_items;
            ArrayVar<int, items_per_thread> with_default_items;
            ArrayVar<int, items_per_thread> store_items;
            for(uint i = 0u; i < items_per_thread; ++i)
            {
                no_default_items[i] = -7;
                transpose_items[i]  = -8;
                store_items[i]      = cast<int>(lane * items_per_thread + i);
            }

            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_DIRECT, warp_size>().Load(
                input_buffer, no_default_items, 0u, guarded_valid_items);
            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE, warp_size>().Load(
                input_buffer, transpose_items, 0u, guarded_valid_items);
            WarpLoad<int, items_per_thread, WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE, warp_size>().Load(
                input_buffer, with_default_items, 0u, guarded_valid_items, -9);

            for(uint i = 0u; i < items_per_thread; ++i)
            {
                UInt thread_item = lane * items_per_thread + i;
                no_default_buffer.write(thread_item, no_default_items[i]);
                transpose_buffer.write(thread_item, transpose_items[i]);
                with_default_buffer.write(thread_item, with_default_items[i]);
            }

            WarpStore<int, items_per_thread, WarpStoreAlgorithm::WARP_STORE_TRANSPOSE, warp_size>().Store(
                store_items, stored_buffer, 0u, guarded_valid_items);
        };

        auto guarded_shader = device.compile(guarded_kernel);
        stream << guarded_input.copy_from(luisa::span{host_guarded_input})
               << guarded_stored.copy_from(luisa::span{host_guarded_stored})
               << guarded_shader(guarded_input, guarded_no_default, guarded_transpose, guarded_with_default, guarded_stored)
                      .dispatch(warp_size)
               << guarded_no_default.copy_to(luisa::span{host_guarded_no_default})
               << guarded_transpose.copy_to(luisa::span{host_guarded_transpose})
               << guarded_with_default.copy_to(luisa::span{host_guarded_with_default})
               << guarded_stored.copy_to(luisa::span{host_guarded_stored}) << synchronize();

        for(uint i = 0u; i < warp_items; ++i)
        {
            expect(host_guarded_no_default[i] == (i < guarded_valid_items ? host_guarded_input[i] : -7));
            expect(host_guarded_transpose[i] == (i < guarded_valid_items ? host_guarded_input[i] : -8));
            expect(host_guarded_with_default[i] == (i < guarded_valid_items ? host_guarded_input[i] : -9));
            expect(host_guarded_stored[i] == (i < guarded_valid_items ? static_cast<int>(i) : -11));
        }
    };
}
