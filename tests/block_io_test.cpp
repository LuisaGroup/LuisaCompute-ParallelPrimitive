// Tests for BlockLoad and BlockStore.
// Covers direct, striped, vectorized, transpose, guarded, and warp-striped paths.

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
constexpr uint block_threads    = 64u;
constexpr uint items_per_thread = 3u;
constexpr uint block_items      = block_threads * items_per_thread;
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

    "block_load_and_store_implement_every_layout_policy"_test = [&]
    {
        auto input                 = device.create_buffer<int>(block_items);
        auto direct_loaded         = device.create_buffer<int>(block_items);
        auto striped_loaded        = device.create_buffer<int>(block_items);
        auto vector_loaded         = device.create_buffer<int>(block_items);
        auto transpose_loaded      = device.create_buffer<int>(block_items);
        auto warp_transpose_loaded = device.create_buffer<int>(block_items);
        auto direct_stored         = device.create_buffer<int>(block_items);
        auto striped_stored        = device.create_buffer<int>(block_items);
        auto vector_stored         = device.create_buffer<int>(block_items);
        auto transpose_stored      = device.create_buffer<int>(block_items);
        auto warp_transpose_stored = device.create_buffer<int>(block_items);

        luisa::vector<int> host_input(block_items);
        luisa::vector<int> host_direct_loaded(block_items);
        luisa::vector<int> host_striped_loaded(block_items);
        luisa::vector<int> host_vector_loaded(block_items);
        luisa::vector<int> host_transpose_loaded(block_items);
        luisa::vector<int> host_warp_transpose_loaded(block_items);
        luisa::vector<int> host_direct_stored(block_items);
        luisa::vector<int> host_striped_stored(block_items);
        luisa::vector<int> host_vector_stored(block_items);
        luisa::vector<int> host_transpose_stored(block_items);
        luisa::vector<int> host_warp_transpose_stored(block_items);
        for(uint i = 0u; i < block_items; ++i)
        {
            host_input[i] = static_cast<int>(i * 7u + 3u);
        }

        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> direct_loaded_buffer,
                             BufferVar<int> striped_loaded_buffer,
                             BufferVar<int> vector_loaded_buffer,
                             BufferVar<int> transpose_loaded_buffer,
                             BufferVar<int> warp_transpose_loaded_buffer,
                             BufferVar<int> direct_stored_buffer,
                             BufferVar<int> striped_stored_buffer,
                             BufferVar<int> vector_stored_buffer,
                             BufferVar<int> transpose_stored_buffer,
                             BufferVar<int> warp_transpose_stored_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<int, items_per_thread> direct_items;
            ArrayVar<int, items_per_thread> striped_items;
            ArrayVar<int, items_per_thread> vector_items;
            ArrayVar<int, items_per_thread> transpose_items;
            ArrayVar<int, items_per_thread> warp_transpose_items;
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>().Load(
                input_buffer, direct_items, 0u);
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_STRIPED>()
                .Load(input_buffer, striped_items, 0u);
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>()
                .Load(input_buffer, vector_items, 0u);
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>()
                .Load(input_buffer, transpose_items, 0u);
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE>()
                .Load(input_buffer, warp_transpose_items, 0u);

            ArrayVar<int, items_per_thread> blocked_store_items;
            ArrayVar<int, items_per_thread> striped_store_items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt blocked_index = tid * items_per_thread + item;
                UInt striped_index = tid + item * block_threads;
                direct_loaded_buffer.write(blocked_index, direct_items[item]);
                striped_loaded_buffer.write(blocked_index, striped_items[item]);
                vector_loaded_buffer.write(blocked_index, vector_items[item]);
                transpose_loaded_buffer.write(blocked_index, transpose_items[item]);
                warp_transpose_loaded_buffer.write(blocked_index, warp_transpose_items[item]);
                blocked_store_items[item] = cast<int>(blocked_index);
                striped_store_items[item] = cast<int>(striped_index);
            }

            BlockStore<int, block_threads, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_DIRECT>()
                .Store(blocked_store_items, direct_stored_buffer, 0u);
            BlockStore<int, block_threads, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_STRIPED>()
                .Store(striped_store_items, striped_stored_buffer, 0u);
            BlockStore<int, block_threads, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>()
                .Store(blocked_store_items, vector_stored_buffer, 0u);
            BlockStore<int, block_threads, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE>()
                .Store(blocked_store_items, transpose_stored_buffer, 0u);
            BlockStore<int, block_threads, items_per_thread, BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE>()
                .Store(blocked_store_items, warp_transpose_stored_buffer, 0u);
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << shader(input,
                         direct_loaded,
                         striped_loaded,
                         vector_loaded,
                         transpose_loaded,
                         warp_transpose_loaded,
                         direct_stored,
                         striped_stored,
                         vector_stored,
                         transpose_stored,
                         warp_transpose_stored)
                      .dispatch(block_threads)
               << direct_loaded.copy_to(luisa::span{host_direct_loaded})
               << striped_loaded.copy_to(luisa::span{host_striped_loaded})
               << vector_loaded.copy_to(luisa::span{host_vector_loaded})
               << transpose_loaded.copy_to(luisa::span{host_transpose_loaded})
               << warp_transpose_loaded.copy_to(luisa::span{host_warp_transpose_loaded})
               << direct_stored.copy_to(luisa::span{host_direct_stored})
               << striped_stored.copy_to(luisa::span{host_striped_stored})
               << vector_stored.copy_to(luisa::span{host_vector_stored})
               << transpose_stored.copy_to(luisa::span{host_transpose_stored})
               << warp_transpose_stored.copy_to(luisa::span{host_warp_transpose_stored}) << synchronize();

        for(uint thread = 0u; thread < block_threads; ++thread)
        {
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                uint output_index  = thread * items_per_thread + item;
                uint striped_index = thread + item * block_threads;
                expect(host_direct_loaded[output_index] == host_input[output_index]);
                expect(host_striped_loaded[output_index] == host_input[striped_index]);
                expect(host_vector_loaded[output_index] == host_input[output_index]);
                expect(host_transpose_loaded[output_index] == host_input[output_index]);
                expect(host_warp_transpose_loaded[output_index] == host_input[output_index]);
            }
        }
        for(uint item = 0u; item < block_items; ++item)
        {
            expect(host_direct_stored[item] == static_cast<int>(item));
            expect(host_striped_stored[item] == static_cast<int>(item));
            expect(host_vector_stored[item] == static_cast<int>(item));
            expect(host_transpose_stored[item] == static_cast<int>(item));
            expect(host_warp_transpose_stored[item] == static_cast<int>(item));
        }
    };

    "direct_block_i_o_helpers_honor_typed_offsets_and_warp_width"_test = [&]
    {
        constexpr uint tile_offset     = 11u;
        constexpr uint valid_items     = 139u;
        constexpr uint total_items     = tile_offset + block_items + 7u;
        constexpr int  load_default    = -77;
        constexpr int  output_sentinel = -999;

        auto input               = device.create_buffer<int>(total_items);
        auto striped_diagnostics = device.create_buffer<int>(block_items);
        auto warp_striped_output = device.create_buffer<int>(total_items);

        luisa::vector<int> host_input(total_items);
        luisa::vector<int> host_striped_diagnostics(block_items);
        luisa::vector<int> host_warp_striped_output(total_items, output_sentinel);
        for(uint i = 0u; i < total_items; ++i)
        {
            host_input[i] = static_cast<int>(1000u + i * 13u);
        }

        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> striped_diagnostics_buffer,
                             BufferVar<int> warp_striped_output_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<int, items_per_thread> striped_items;
            LoadDirectStriped<block_threads, int, items_per_thread>(
                tid, input_buffer, UInt(11u), striped_items, UInt(139u), Int(-77));
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                striped_diagnostics_buffer.write(tid * UInt(items_per_thread) + UInt(item),
                                                 striped_items[item]);
            }

            ArrayVar<int, items_per_thread> warp_striped_items;
            LoadDirectWarpStriped<int, items_per_thread>(
                tid, input_buffer, UInt(11u), warp_striped_items, UInt(139u), Int(-77));
            StoreDirectWarpStriped<int, items_per_thread>(
                tid, warp_striped_output_buffer, UInt(11u), warp_striped_items, UInt(139u));
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << warp_striped_output.copy_from(luisa::span{host_warp_striped_output})
               << shader(input, striped_diagnostics, warp_striped_output).dispatch(block_threads)
               << striped_diagnostics.copy_to(luisa::span{host_striped_diagnostics})
               << warp_striped_output.copy_to(luisa::span{host_warp_striped_output}) << synchronize();

        for(uint thread = 0u; thread < block_threads; ++thread)
        {
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                uint logical_index = thread + item * block_threads;
                int expected = logical_index < valid_items ? host_input[tile_offset + logical_index] : load_default;
                expect(host_striped_diagnostics[thread * items_per_thread + item] == expected);
            }
        }
        for(uint i = 0u; i < total_items; ++i)
        {
            int expected = i >= tile_offset && i < tile_offset + valid_items ? host_input[i] : output_sentinel;
            expect(host_warp_striped_output[i] == expected);
        }
    };

    "guarded_transpose_loads_keep_initialized_out_of_bounds_items"_test = [&]
    {
        constexpr uint valid_items = 139u;
        constexpr int  sentinel    = -88;

        auto input                 = device.create_buffer<int>(block_items);
        auto transpose_output      = device.create_buffer<int>(block_items);
        auto warp_transpose_output = device.create_buffer<int>(block_items);

        luisa::vector<int> host_input(block_items);
        luisa::vector<int> host_transpose_output(block_items);
        luisa::vector<int> host_warp_transpose_output(block_items);
        for(uint index = 0u; index < block_items; ++index)
        {
            host_input[index] = static_cast<int>(index * 11u + 5u);
        }

        Kernel1D kernel = [](BufferVar<int> input_buffer,
                             BufferVar<int> transpose_buffer,
                             BufferVar<int> warp_transpose_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<int, items_per_thread> transpose_items;
            ArrayVar<int, items_per_thread> warp_transpose_items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                transpose_items[item]      = -88;
                warp_transpose_items[item] = -88;
            }
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>()
                .Load(input_buffer, transpose_items, 0u, 139u);
            BlockLoad<int, block_threads, items_per_thread, BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE>()
                .Load(input_buffer, warp_transpose_items, 0u, 139u);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                transpose_buffer.write(index, transpose_items[item]);
                warp_transpose_buffer.write(index, warp_transpose_items[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << input.copy_from(luisa::span{host_input})
               << shader(input, transpose_output, warp_transpose_output).dispatch(block_threads)
               << transpose_output.copy_to(luisa::span{host_transpose_output})
               << warp_transpose_output.copy_to(luisa::span{host_warp_transpose_output}) << synchronize();

        for(uint index = 0u; index < block_items; ++index)
        {
            int expected = index < valid_items ? host_input[index] : sentinel;
            expect(host_transpose_output[index] == expected);
            expect(host_warp_transpose_output[index] == expected);
        }
    };
}
