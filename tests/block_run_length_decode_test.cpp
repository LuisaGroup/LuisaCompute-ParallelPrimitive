// Tests for BlockRunLengthDecode.
// Covers run lengths, offsets, windows, item-only output, and relative positions.

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
constexpr uint block_threads            = 32u;
constexpr uint runs_per_thread          = 1u;
constexpr uint decoded_items_per_thread = 2u;
constexpr uint block_runs               = block_threads * runs_per_thread;
constexpr uint window_items             = block_threads * decoded_items_per_thread;
constexpr uint nonempty_runs            = 12u;
constexpr uint run_length               = 8u;
constexpr uint total_decoded_items      = nonempty_runs * run_length;
constexpr uint shifted_window_offset    = 20u;
constexpr uint item_only_window_offset  = 8u;
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

    "block_run_length_decode_supports_lengths_offsets_windows_and_relative_positions"_test = [&]
    {
        auto first_items      = device.create_buffer<uint>(window_items);
        auto first_offsets    = device.create_buffer<uint>(window_items);
        auto shifted_items    = device.create_buffer<uint>(window_items);
        auto shifted_offsets  = device.create_buffer<uint>(window_items);
        auto item_only_output = device.create_buffer<uint>(window_items);
        auto decoded_size     = device.create_buffer<uint>(1u);

        luisa::vector<uint> host_first_items(window_items);
        luisa::vector<uint> host_first_offsets(window_items);
        luisa::vector<uint> host_shifted_items(window_items);
        luisa::vector<uint> host_shifted_offsets(window_items);
        luisa::vector<uint> host_item_only(window_items);
        uint                host_decoded_size = 0u;

        Kernel1D kernel = [](BufferVar<uint> first_items_buffer,
                             BufferVar<uint> first_offsets_buffer,
                             BufferVar<uint> shifted_items_buffer,
                             BufferVar<uint> shifted_offsets_buffer,
                             BufferVar<uint> item_only_buffer,
                             BufferVar<uint> decoded_size_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<uint, runs_per_thread> run_values;
            ArrayVar<uint, runs_per_thread> run_lengths;
            ArrayVar<uint, runs_per_thread> run_offsets;
            for(uint item = 0u; item < runs_per_thread; ++item)
            {
                UInt run          = tid * runs_per_thread + item;
                run_values[item]  = 100u + run;
                run_lengths[item] = select(0u, run_length, run < nonempty_runs);
                run_offsets[item] = min(run, nonempty_runs) * run_length;
            }

            UInt total_size;
            BlockRunLengthDecode<uint, block_threads, runs_per_thread, decoded_items_per_thread> lengths_decoder(
                run_values, run_lengths, total_size);

            ArrayVar<uint, decoded_items_per_thread> decoded_items;
            ArrayVar<uint, decoded_items_per_thread> relative_offsets;
            lengths_decoder.RunLengthDecode(decoded_items, relative_offsets);

            for(uint item = 0u; item < decoded_items_per_thread; ++item)
            {
                UInt index = tid * decoded_items_per_thread + item;
                first_items_buffer.write(index, decoded_items[item]);
                first_offsets_buffer.write(index, relative_offsets[item]);
            }
            $if(tid == 0u)
            {
                decoded_size_buffer.write(0u, total_size);
            };

            BlockRunLengthDecode<uint, block_threads, runs_per_thread, decoded_items_per_thread> offsets_decoder(
                run_values, run_offsets);
            offsets_decoder.RunLengthDecode(decoded_items, relative_offsets, shifted_window_offset);
            for(uint item = 0u; item < decoded_items_per_thread; ++item)
            {
                UInt index = tid * decoded_items_per_thread + item;
                shifted_items_buffer.write(index, decoded_items[item]);
                shifted_offsets_buffer.write(index, relative_offsets[item]);
            }

            offsets_decoder.RunLengthDecode(decoded_items, item_only_window_offset);
            for(uint item = 0u; item < decoded_items_per_thread; ++item)
            {
                UInt index = tid * decoded_items_per_thread + item;
                item_only_buffer.write(index, decoded_items[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(first_items, first_offsets, shifted_items, shifted_offsets, item_only_output, decoded_size)
                      .dispatch(block_threads)
               << first_items.copy_to(luisa::span{host_first_items})
               << first_offsets.copy_to(luisa::span{host_first_offsets})
               << shifted_items.copy_to(luisa::span{host_shifted_items})
               << shifted_offsets.copy_to(luisa::span{host_shifted_offsets})
               << item_only_output.copy_to(luisa::span{host_item_only})
               << decoded_size.copy_to(luisa::span{&host_decoded_size, 1u}) << synchronize();

        expect(host_decoded_size == total_decoded_items);
        for(uint index = 0u; index < window_items; ++index)
        {
            uint first_decoded = index;
            expect(host_first_items[index] == 100u + first_decoded / run_length);
            expect(host_first_offsets[index] == first_decoded % run_length);

            uint shifted_decoded = shifted_window_offset + index;
            expect(host_shifted_items[index] == 100u + shifted_decoded / run_length);
            expect(host_shifted_offsets[index] == shifted_decoded % run_length);

            uint item_only_decoded = item_only_window_offset + index;
            expect(host_item_only[index] == 100u + item_only_decoded / run_length);
        }
    };
}
