// Tests for BlockAdjacentDifference.
// Covers boundaries, partial tiles, and head/tail flag helpers.

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
constexpr uint block_threads    = 32u;
constexpr uint items_per_thread = 4u;
constexpr uint block_items      = block_threads * items_per_thread;
constexpr uint valid_items      = 73u;
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

    "block_adjacent_difference_handles_boundaries_partial_tiles_and_flags"_test = [&]
    {
        auto left_output    = device.create_buffer<int>(block_items);
        auto right_output   = device.create_buffer<int>(block_items);
        auto partial_output = device.create_buffer<int>(block_items);
        auto head_flags     = device.create_buffer<int>(block_items);
        auto tail_flags     = device.create_buffer<int>(block_items);

        luisa::vector<int> host_left(block_items);
        luisa::vector<int> host_right(block_items);
        luisa::vector<int> host_partial(block_items);
        luisa::vector<int> host_head(block_items);
        luisa::vector<int> host_tail(block_items);

        Kernel1D kernel = [](BufferVar<int> left_buffer,
                             BufferVar<int> right_buffer,
                             BufferVar<int> partial_buffer,
                             BufferVar<int> head_buffer,
                             BufferVar<int> tail_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt                            tid = thread_id().x;
            ArrayVar<int, items_per_thread> input;
            ArrayVar<int, items_per_thread> grouped_input;
            ArrayVar<int, items_per_thread> left;
            ArrayVar<int, items_per_thread> right;
            ArrayVar<int, items_per_thread> partial;
            ArrayVar<int, items_per_thread> heads;
            ArrayVar<int, items_per_thread> tails;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index          = tid * items_per_thread + item;
                input[item]         = cast<int>(index * index + 10u);
                grouped_input[item] = cast<int>(index / 5u);
            }

            auto difference = [](const Int& lhs, const Int& rhs) noexcept { return lhs - rhs; };
            auto different  = [](const Int& lhs, const Int& rhs) noexcept { return lhs != rhs; };

            BlockAdjacentDifference<int, block_threads> adjacent;
            adjacent.SubtractLeft(input, left, difference);
            adjacent.SubtractRight(input, right, difference);
            adjacent.SubtractLeftPartialTile(input, partial, difference, valid_items, -7);
            adjacent.FlagHeads(heads, grouped_input, different);
            adjacent.FlagTails(tails, grouped_input, different);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                left_buffer.write(index, left[item]);
                right_buffer.write(index, right[item]);
                partial_buffer.write(index, partial[item]);
                head_buffer.write(index, heads[item]);
                tail_buffer.write(index, tails[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(left_output, right_output, partial_output, head_flags, tail_flags).dispatch(block_threads)
               << left_output.copy_to(luisa::span{host_left})
               << right_output.copy_to(luisa::span{host_right})
               << partial_output.copy_to(luisa::span{host_partial})
               << head_flags.copy_to(luisa::span{host_head})
               << tail_flags.copy_to(luisa::span{host_tail}) << synchronize();

        for(uint index = 0u; index < block_items; ++index)
        {
            int value = static_cast<int>(index * index + 10u);
            int predecessor = index == 0u ? value : static_cast<int>((index - 1u) * (index - 1u) + 10u);
            int successor =
                index + 1u == block_items ? value : static_cast<int>((index + 1u) * (index + 1u) + 10u);
            expect(host_left[index] == (index == 0u ? value : value - predecessor));
            expect(host_right[index] == (index + 1u == block_items ? value : value - successor));

            int partial_expected = value;
            if(index < valid_items)
            {
                partial_expected = index == 0u ? value - (-7) : value - predecessor;
            }
            expect(host_partial[index] == partial_expected);

            int group          = static_cast<int>(index / 5u);
            int previous_group = index == 0u ? group : static_cast<int>((index - 1u) / 5u);
            int next_group = index + 1u == block_items ? group : static_cast<int>((index + 1u) / 5u);
            expect(host_head[index] == (index == 0u || group != previous_group ? 1 : 0));
            expect(host_tail[index] == (index + 1u == block_items || group != next_group ? 1 : 0));
        }
    };
}
