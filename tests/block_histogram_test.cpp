// Tests for BlockHistogram.
// Covers initialization and sort-, atomic-, and composite-policy histograms.

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
constexpr uint histogram_bins   = 11u;
constexpr uint initial_count    = 5u;
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

    "block_histogram_initializes_counts_and_composites_both_policies"_test = [&]
    {
        auto sort_output      = device.create_buffer<uint>(histogram_bins);
        auto atomic_output    = device.create_buffer<uint>(histogram_bins);
        auto composite_output = device.create_buffer<uint>(histogram_bins);

        luisa::vector<uint> host_sort(histogram_bins);
        luisa::vector<uint> host_atomic(histogram_bins);
        luisa::vector<uint> host_composite(histogram_bins);

        Kernel1D kernel = [](BufferVar<uint> sort_buffer, BufferVar<uint> atomic_buffer, BufferVar<uint> composite_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt tid = thread_id().x;

            ArrayVar<uint, items_per_thread> items;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index  = tid * items_per_thread + item;
                items[item] = (index * 7u + 3u) % histogram_bins;
            }

            auto sort_bins      = new Shared<uint>{histogram_bins};
            auto atomic_bins    = new Shared<uint>{histogram_bins};
            auto composite_bins = new Shared<uint>{histogram_bins};

            BlockHistogram<uint, block_threads, items_per_thread, histogram_bins, BLOCK_HISTO_SORT> sort_histogram;
            sort_histogram.Histogram(items, sort_bins);

            BlockHistogram<uint, block_threads, items_per_thread, histogram_bins, BLOCK_HISTO_ATOMIC> atomic_histogram;
            atomic_histogram.Histogram(items, atomic_bins);

            sort_histogram.InitHistogram(composite_bins);
            sync_block();
            $if(tid < histogram_bins)
            {
                composite_bins->write(tid, initial_count);
            };
            sync_block();
            sort_histogram.Composite(items, composite_bins);

            $if(tid < histogram_bins)
            {
                sort_buffer.write(tid, sort_bins->read(tid));
                atomic_buffer.write(tid, atomic_bins->read(tid));
                composite_buffer.write(tid, composite_bins->read(tid));
            };
        };

        auto shader = device.compile(kernel);
        stream << shader(sort_output, atomic_output, composite_output).dispatch(block_threads)
               << sort_output.copy_to(luisa::span{host_sort})
               << atomic_output.copy_to(luisa::span{host_atomic})
               << composite_output.copy_to(luisa::span{host_composite}) << synchronize();

        luisa::vector<uint> expected(histogram_bins, 0u);
        for(uint index = 0u; index < block_items; ++index)
        {
            ++expected[(index * 7u + 3u) % histogram_bins];
        }
        for(uint bin = 0u; bin < histogram_bins; ++bin)
        {
            expect(host_sort[bin] == expected[bin]);
            expect(host_atomic[bin] == expected[bin]);
            expect(host_composite[bin] == expected[bin] + initial_count);
        }
    };
}
