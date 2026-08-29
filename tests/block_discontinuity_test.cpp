// Tests for BlockDiscontinuity.
// Covers head/tail flags, boundary items, predecessors, and indexed operators.

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

    "block_discontinuity_exposes_plural_tails_boundaries_predecessors_and_tile_indices"_test = [&]
    {
        auto basic_heads   = device.create_buffer<int>(block_items);
        auto basic_tails   = device.create_buffer<int>(block_items);
        auto predecessors  = device.create_buffer<int>(block_items);
        auto indexed_heads = device.create_buffer<int>(block_items);
        auto indexed_tails = device.create_buffer<int>(block_items);

        luisa::vector<int> host_basic_heads(block_items);
        luisa::vector<int> host_basic_tails(block_items);
        luisa::vector<int> host_predecessors(block_items);
        luisa::vector<int> host_indexed_heads(block_items);
        luisa::vector<int> host_indexed_tails(block_items);

        Kernel1D kernel = [](BufferVar<int> basic_heads_buffer,
                             BufferVar<int> basic_tails_buffer,
                             BufferVar<int> predecessors_buffer,
                             BufferVar<int> indexed_heads_buffer,
                             BufferVar<int> indexed_tails_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt                            tid = thread_id().x;
            ArrayVar<int, items_per_thread> input;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index  = tid * items_per_thread + item;
                input[item] = cast<int>(index / 5u);
            }

            ArrayVar<int, items_per_thread> heads;
            ArrayVar<int, items_per_thread> tails;
            ArrayVar<int, items_per_thread> previous;
            auto different = [](const Int& lhs, const Int& rhs) noexcept { return lhs != rhs; };
            BlockDiscontinuity<int, block_threads, items_per_thread> discontinuity;
            discontinuity.FlagHeads(heads, previous, input, different, -1);
            discontinuity.FlagTails(tails, input, different);

            ArrayVar<int, items_per_thread> indexed_head;
            ArrayVar<int, items_per_thread> indexed_tail;
            auto indexed_different = [](const Int& lhs, const Int& rhs, UInt b_index) noexcept
            { return (lhs != rhs) | (b_index % 17u == 0u); };
            discontinuity.FlagHeadsAndTails(indexed_head, -1, indexed_tail, 999, input, indexed_different);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                basic_heads_buffer.write(index, heads[item]);
                basic_tails_buffer.write(index, tails[item]);
                predecessors_buffer.write(index, previous[item]);
                indexed_heads_buffer.write(index, indexed_head[item]);
                indexed_tails_buffer.write(index, indexed_tail[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(basic_heads, basic_tails, predecessors, indexed_heads, indexed_tails).dispatch(block_threads)
               << basic_heads.copy_to(luisa::span{host_basic_heads})
               << basic_tails.copy_to(luisa::span{host_basic_tails})
               << predecessors.copy_to(luisa::span{host_predecessors})
               << indexed_heads.copy_to(luisa::span{host_indexed_heads})
               << indexed_tails.copy_to(luisa::span{host_indexed_tails}) << synchronize();

        for(uint index = 0u; index < block_items; ++index)
        {
            int value    = static_cast<int>(index / 5u);
            int previous = index == 0u ? -1 : static_cast<int>((index - 1u) / 5u);
            int next     = index + 1u == block_items ? 999 : static_cast<int>((index + 1u) / 5u);
            expect(host_basic_heads[index] == (value != previous ? 1 : 0));
            expect(host_basic_tails[index] == (index + 1u == block_items || value != next ? 1 : 0));
            expect(host_predecessors[index] == previous);
            expect(host_indexed_heads[index] == (value != previous || index % 17u == 0u ? 1 : 0));
            expect(host_indexed_tails[index] == (value != next || (index + 1u) % 17u == 0u ? 1 : 0));
        }
    };
}
