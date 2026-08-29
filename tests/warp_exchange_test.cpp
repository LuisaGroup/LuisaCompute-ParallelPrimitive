// Tests for WarpExchange.
// Covers logical-warp transposes, round trips, and striped scatter.

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
constexpr uint logical_warp_threads  = 8u;
constexpr uint items_per_thread      = 3u;
constexpr uint physical_warp_threads = 32u;
constexpr uint logical_warp_items    = logical_warp_threads * items_per_thread;
constexpr uint item_count            = physical_warp_threads * items_per_thread;
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

    "warp_exchange_supports_logical_warp_transposes_and_scatter"_test = [&]
    {
        auto blocked_to_striped = device.create_buffer<int>(item_count);
        auto striped_to_blocked = device.create_buffer<int>(item_count);
        auto in_place_roundtrip = device.create_buffer<int>(item_count);
        auto scatter_to_striped = device.create_buffer<int>(item_count);

        luisa::vector<int> host_blocked_to_striped(item_count);
        luisa::vector<int> host_striped_to_blocked(item_count);
        luisa::vector<int> host_in_place_roundtrip(item_count);
        luisa::vector<int> host_scatter_to_striped(item_count);

        Kernel1D kernel = [](BufferVar<int> blocked_to_striped_buffer,
                             BufferVar<int> striped_to_blocked_buffer,
                             BufferVar<int> in_place_roundtrip_buffer,
                             BufferVar<int> scatter_to_striped_buffer) noexcept
        {
            set_block_size(physical_warp_threads);
            UInt thread_index      = dispatch_id().x;
            UInt lane              = warp_lane_id() % logical_warp_threads;
            UInt logical_warp_base = thread_index / logical_warp_threads * logical_warp_items;

            ArrayVar<int, items_per_thread>  blocked_items;
            ArrayVar<int, items_per_thread>  striped_items;
            ArrayVar<int, items_per_thread>  blocked_output;
            ArrayVar<int, items_per_thread>  striped_output;
            ArrayVar<int, items_per_thread>  roundtrip_items;
            ArrayVar<int, items_per_thread>  scatter_items;
            ArrayVar<uint, items_per_thread> scatter_ranks;

            for(uint i = 0u; i < items_per_thread; ++i)
            {
                blocked_items[i]   = cast<int>(logical_warp_base + lane * items_per_thread + i);
                striped_items[i]   = cast<int>(logical_warp_base + lane + i * logical_warp_threads);
                roundtrip_items[i] = blocked_items[i];
                scatter_items[i]   = blocked_items[i];
                scatter_ranks[i]   = logical_warp_items - 1u - (lane * items_per_thread + i);
            }

            WarpExchange<int, items_per_thread, logical_warp_threads, WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE> exchange;
            exchange.BlockedToStriped(blocked_items, striped_output);
            exchange.StripedToBlocked(striped_items, blocked_output);
            exchange.BlockedToStriped(roundtrip_items, roundtrip_items);
            exchange.StripedToBlocked(roundtrip_items, roundtrip_items);
            exchange.ScatterToStriped(scatter_items, scatter_ranks);

            for(uint i = 0u; i < items_per_thread; ++i)
            {
                UInt output_index = thread_index * items_per_thread + i;
                blocked_to_striped_buffer.write(output_index, striped_output[i]);
                striped_to_blocked_buffer.write(output_index, blocked_output[i]);
                in_place_roundtrip_buffer.write(output_index, roundtrip_items[i]);
                scatter_to_striped_buffer.write(output_index, scatter_items[i]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(blocked_to_striped, striped_to_blocked, in_place_roundtrip, scatter_to_striped)
                      .dispatch(physical_warp_threads)
               << blocked_to_striped.copy_to(luisa::span{host_blocked_to_striped})
               << striped_to_blocked.copy_to(luisa::span{host_striped_to_blocked})
               << in_place_roundtrip.copy_to(luisa::span{host_in_place_roundtrip})
               << scatter_to_striped.copy_to(luisa::span{host_scatter_to_striped}) << synchronize();

        for(uint thread = 0u; thread < physical_warp_threads; ++thread)
        {
            uint lane              = thread % logical_warp_threads;
            uint logical_warp_base = thread / logical_warp_threads * logical_warp_items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                uint output_index  = thread * items_per_thread + item;
                uint striped_index = logical_warp_base + lane + item * logical_warp_threads;
                uint blocked_index = logical_warp_base + lane * items_per_thread + item;
                uint reverse_index =
                    logical_warp_base + (logical_warp_items - 1u - (lane + item * logical_warp_threads));
                expect(host_blocked_to_striped[output_index] == static_cast<int>(striped_index));
                expect(host_striped_to_blocked[output_index] == static_cast<int>(blocked_index));
                expect(host_in_place_roundtrip[output_index] == static_cast<int>(blocked_index));
                expect(host_scatter_to_striped[output_index] == static_cast<int>(reverse_index));
            }
        }
    };
}
