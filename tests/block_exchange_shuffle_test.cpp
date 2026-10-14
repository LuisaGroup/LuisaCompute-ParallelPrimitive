// Tests for BlockExchange and BlockShuffle.
// Covers layout conversion, scatter, rotation, offsets, and prefix/suffix shifts.

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

    "block_exchange_and_shuffle_preserve_layouts_boundaries_and_aliases"_test = [&]
    {
        auto blocked_to_striped = device.create_buffer<int>(block_items);
        auto striped_to_blocked = device.create_buffer<int>(block_items);
        auto roundtrip          = device.create_buffer<int>(block_items);
        auto scatter_blocked    = device.create_buffer<int>(block_items);
        auto offset_output      = device.create_buffer<int>(block_threads);
        auto rotate_output      = device.create_buffer<int>(block_threads);
        auto up_output          = device.create_buffer<int>(block_items);
        auto down_output        = device.create_buffer<int>(block_items);
        auto suffix_output      = device.create_buffer<int>(block_threads);
        auto prefix_output      = device.create_buffer<int>(block_threads);

        luisa::vector<int> host_blocked_to_striped(block_items);
        luisa::vector<int> host_striped_to_blocked(block_items);
        luisa::vector<int> host_roundtrip(block_items);
        luisa::vector<int> host_scatter_blocked(block_items);
        luisa::vector<int> host_offset(block_threads);
        luisa::vector<int> host_rotate(block_threads);
        luisa::vector<int> host_up(block_items);
        luisa::vector<int> host_down(block_items);
        luisa::vector<int> host_suffix(block_threads);
        luisa::vector<int> host_prefix(block_threads);

        Kernel1D kernel = [](BufferVar<int> blocked_to_striped_buffer,
                             BufferVar<int> striped_to_blocked_buffer,
                             BufferVar<int> roundtrip_buffer,
                             BufferVar<int> scatter_blocked_buffer,
                             BufferVar<int> offset_buffer,
                             BufferVar<int> rotate_buffer,
                             BufferVar<int> up_buffer,
                             BufferVar<int> down_buffer,
                             BufferVar<int> suffix_buffer,
                             BufferVar<int> prefix_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<int, items_per_thread>  blocked_items;
            ArrayVar<int, items_per_thread>  striped_items;
            ArrayVar<int, items_per_thread>  blocked_output;
            ArrayVar<int, items_per_thread>  striped_output;
            ArrayVar<int, items_per_thread>  roundtrip_items;
            ArrayVar<int, items_per_thread>  scatter_items;
            ArrayVar<uint, items_per_thread> scatter_ranks;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                blocked_items[item]   = cast<int>(tid * items_per_thread + item);
                striped_items[item]   = cast<int>(tid + item * block_threads);
                roundtrip_items[item] = blocked_items[item];
                scatter_items[item]   = blocked_items[item];
                scatter_ranks[item]   = block_items - 1u - (tid * items_per_thread + item);
            }

            BlockExchange<int, block_threads, items_per_thread> exchange;
            exchange.BlockedToStriped(blocked_items, striped_output);
            exchange.StripedToBlocked(striped_items, blocked_output);
            exchange.BlockedToStriped(roundtrip_items, roundtrip_items);
            exchange.StripedToBlocked(roundtrip_items, roundtrip_items);
            exchange.ScatterToBlocked(scatter_items, scatter_ranks);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt output_index = tid * items_per_thread + item;
                blocked_to_striped_buffer.write(output_index, striped_output[item]);
                striped_to_blocked_buffer.write(output_index, blocked_output[item]);
                roundtrip_buffer.write(output_index, roundtrip_items[item]);
                scatter_blocked_buffer.write(output_index, scatter_items[item]);
            }

            BlockShuffle<int, block_threads> shuffle;
            Int                              offset = -1;
            Int                              rotate = -1;
            shuffle.Offset(cast<int>(tid), offset, 2);
            shuffle.Rotate(cast<int>(tid), rotate, 5u);
            offset_buffer.write(tid, offset);
            rotate_buffer.write(tid, rotate);

            ArrayVar<int, items_per_thread> up_items;
            ArrayVar<int, items_per_thread> down_items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                up_items[item]   = -1;
                down_items[item] = -1;
            }
            Int suffix;
            Int prefix;
            shuffle.Up(blocked_items, up_items, suffix);
            shuffle.Down(blocked_items, down_items, prefix);
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt output_index = tid * items_per_thread + item;
                up_buffer.write(output_index, up_items[item]);
                down_buffer.write(output_index, down_items[item]);
            }
            suffix_buffer.write(tid, suffix);
            prefix_buffer.write(tid, prefix);
        };

        auto shader = device.compile(kernel);
        stream << shader(blocked_to_striped,
                         striped_to_blocked,
                         roundtrip,
                         scatter_blocked,
                         offset_output,
                         rotate_output,
                         up_output,
                         down_output,
                         suffix_output,
                         prefix_output)
                      .dispatch(block_threads)
               << blocked_to_striped.copy_to(luisa::span{host_blocked_to_striped})
               << striped_to_blocked.copy_to(luisa::span{host_striped_to_blocked})
               << roundtrip.copy_to(luisa::span{host_roundtrip})
               << scatter_blocked.copy_to(luisa::span{host_scatter_blocked})
               << offset_output.copy_to(luisa::span{host_offset})
               << rotate_output.copy_to(luisa::span{host_rotate})
               << up_output.copy_to(luisa::span{host_up}) << down_output.copy_to(luisa::span{host_down})
               << suffix_output.copy_to(luisa::span{host_suffix})
               << prefix_output.copy_to(luisa::span{host_prefix}) << synchronize();

        for(uint thread = 0u; thread < block_threads; ++thread)
        {
            expect(host_offset[thread] == (thread + 2u < block_threads ? static_cast<int>(thread + 2u) : -1));
            expect(host_rotate[thread] == static_cast<int>((thread + 5u) % block_threads));
            expect(host_suffix[thread] == static_cast<int>(block_items - 1u));
            expect(host_prefix[thread] == 0);
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                uint output_index  = thread * items_per_thread + item;
                uint striped_index = thread + item * block_threads;
                expect(host_blocked_to_striped[output_index] == static_cast<int>(striped_index));
                expect(host_striped_to_blocked[output_index] == static_cast<int>(output_index));
                expect(host_roundtrip[output_index] == static_cast<int>(output_index));
                expect(host_scatter_blocked[output_index] == static_cast<int>(block_items - 1u - output_index));
                expect(host_up[output_index]
                       == (output_index == 0u ? -1 : static_cast<int>(output_index - 1u)));
                expect(host_down[output_index]
                       == (output_index + 1u == block_items ? -1 : static_cast<int>(output_index + 1u)));
            }
        }
    };
}
