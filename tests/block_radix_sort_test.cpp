// Tests for BlockRadixSort.
// Covers signed pairs, ordering direction, striped output, and bit ranges.

#include <lcpp/parallel_primitive.h>

#include <bit>
#include <boost/ut.hpp>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <numeric>
#include <tuple>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

struct BlockRadixCompositeKey
{
    uint primary;
    int  secondary;
};

LUISA_STRUCT(BlockRadixCompositeKey, primary, secondary){};

namespace
{
constexpr uint block_threads    = 32u;
constexpr uint items_per_thread = 2u;
constexpr uint block_items      = block_threads * items_per_thread;

[[nodiscard]] bool composite_equal(const BlockRadixCompositeKey& lhs, const BlockRadixCompositeKey& rhs) noexcept
{
    return lhs.primary == rhs.primary && lhs.secondary == rhs.secondary;
}

[[nodiscard]] uint cross_member_bits(const BlockRadixCompositeKey& key) noexcept
{
    auto secondary_bits = std::bit_cast<uint>(key.secondary) ^ 0x80000000u;
    return ((key.primary & 0x0fu) << 4u) | ((secondary_bits >> 28u) & 0x0fu);
}
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

    "block_radix_sort_handles_signed_pairs_descending_striped_and_bit_ranges"_test = [&]
    {
        auto ascending_keys   = device.create_buffer<int>(block_items);
        auto ascending_values = device.create_buffer<uint>(block_items);
        auto descending_keys  = device.create_buffer<int>(block_items);
        auto striped_keys     = device.create_buffer<int>(block_items);
        auto bit_range_keys   = device.create_buffer<uint>(block_items);
        auto empty_range_keys = device.create_buffer<uint>(block_items);

        luisa::vector<int>  host_ascending_keys(block_items);
        luisa::vector<uint> host_ascending_values(block_items);
        luisa::vector<int>  host_descending_keys(block_items);
        luisa::vector<int>  host_striped_keys(block_items);
        luisa::vector<uint> host_bit_range_keys(block_items);
        luisa::vector<uint> host_empty_range_keys(block_items);

        Kernel1D kernel = [](BufferVar<int>  ascending_keys_buffer,
                             BufferVar<uint> ascending_values_buffer,
                             BufferVar<int>  descending_keys_buffer,
                             BufferVar<int>  striped_keys_buffer,
                             BufferVar<uint> bit_range_keys_buffer,
                             BufferVar<uint> empty_range_keys_buffer) noexcept
        {
            set_block_size(block_threads);
            UInt                             tid = thread_id().x;
            ArrayVar<int, items_per_thread>  keys;
            ArrayVar<uint, items_per_thread> values;
            ArrayVar<int, items_per_thread>  descending;
            ArrayVar<int, items_per_thread>  striped;
            ArrayVar<uint, items_per_thread> bit_range;
            ArrayVar<uint, items_per_thread> empty_range;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index        = tid * items_per_thread + item;
                Int  key          = cast<int>((index * 17u + 3u) % 13u) - 6;
                keys[item]        = key;
                values[item]      = index;
                descending[item]  = key;
                striped[item]     = key;
                bit_range[item]   = (index * 37u + 9u) & 255u;
                empty_range[item] = (index * 29u + 7u) & 255u;
            }

            BlockRadixSort<int, block_threads, items_per_thread, uint> pair_sort;
            pair_sort.Sort(keys, values);

            BlockRadixSort<int, block_threads, items_per_thread> descending_sort;
            descending_sort.SortDescending(descending);

            BlockRadixSort<int, block_threads, items_per_thread> striped_sort;
            striped_sort.SortBlockedToStriped(striped);

            BlockRadixSort<uint, block_threads, items_per_thread> bit_sort;
            bit_sort.Sort(bit_range, 2u, 6u);

            BlockRadixSort<uint, block_threads, items_per_thread> empty_range_sort;
            empty_range_sort.Sort(empty_range, 5u, 5u);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                ascending_keys_buffer.write(index, keys[item]);
                ascending_values_buffer.write(index, values[item]);
                descending_keys_buffer.write(index, descending[item]);
                striped_keys_buffer.write(index, striped[item]);
                bit_range_keys_buffer.write(index, bit_range[item]);
                empty_range_keys_buffer.write(index, empty_range[item]);
            }
        };

        auto shader = device.compile(kernel);
        stream << shader(ascending_keys, ascending_values, descending_keys, striped_keys, bit_range_keys, empty_range_keys)
                      .dispatch(block_threads)
               << ascending_keys.copy_to(luisa::span{host_ascending_keys})
               << ascending_values.copy_to(luisa::span{host_ascending_values})
               << descending_keys.copy_to(luisa::span{host_descending_keys})
               << striped_keys.copy_to(luisa::span{host_striped_keys})
               << bit_range_keys.copy_to(luisa::span{host_bit_range_keys})
               << empty_range_keys.copy_to(luisa::span{host_empty_range_keys}) << synchronize();

        auto signed_key_at = [](uint index)
        { return static_cast<int>((index * 17u + 3u) % 13u) - 6; };
        luisa::vector<uint> order(block_items);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](uint lhs, uint rhs) { return signed_key_at(lhs) < signed_key_at(rhs); });
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(host_ascending_keys[index] == signed_key_at(order[index]));
            expect(host_ascending_values[index] == order[index]);
            expect(host_descending_keys[index] == signed_key_at(order[block_items - 1u - index]));

            uint lane         = index / items_per_thread;
            uint item         = index % items_per_thread;
            uint sorted_index = lane + item * block_threads;
            expect(host_striped_keys[index] == signed_key_at(order[sorted_index]));
        }

        luisa::vector<uint> bit_order(block_items);
        std::iota(bit_order.begin(), bit_order.end(), 0u);
        auto bit_key_at = [](uint index) { return (index * 37u + 9u) & 255u; };
        std::stable_sort(bit_order.begin(),
                         bit_order.end(),
                         [&](uint lhs, uint rhs) {
                             return ((bit_key_at(lhs) >> 2u) & 15u) < ((bit_key_at(rhs) >> 2u) & 15u);
                         });
        for(uint index = 0u; index < block_items; ++index)
        {
            expect(host_bit_range_keys[index] == bit_key_at(bit_order[index]));
            expect(host_empty_range_keys[index] == ((index * 29u + 7u) & 255u));
        }
    };

    "block_radix_sort_decomposer_covers_all_method_groups"_test = [&]
    {
        luisa::vector<BlockRadixCompositeKey> input(block_items);
        for(uint index = 0u; index < block_items; ++index)
        {
            input[index] =
                BlockRadixCompositeKey{(index + 1u) % 4u, static_cast<int>((index * 3u + 2u) % 5u) - 2};
        }

        luisa::vector<uint> full_ascending(block_items);
        luisa::vector<uint> full_descending(block_items);
        luisa::vector<uint> partial_ascending(block_items);
        luisa::vector<uint> partial_descending(block_items);
        std::iota(full_ascending.begin(), full_ascending.end(), 0u);
        std::iota(full_descending.begin(), full_descending.end(), 0u);
        std::iota(partial_ascending.begin(), partial_ascending.end(), 0u);
        std::iota(partial_descending.begin(), partial_descending.end(), 0u);
        std::stable_sort(full_ascending.begin(),
                         full_ascending.end(),
                         [&](uint lhs, uint rhs)
                         {
                             return std::tie(input[lhs].primary, input[lhs].secondary)
                                    < std::tie(input[rhs].primary, input[rhs].secondary);
                         });
        std::stable_sort(full_descending.begin(),
                         full_descending.end(),
                         [&](uint lhs, uint rhs)
                         {
                             return std::tie(input[lhs].primary, input[lhs].secondary)
                                    > std::tie(input[rhs].primary, input[rhs].secondary);
                         });
        std::stable_sort(partial_ascending.begin(),
                         partial_ascending.end(),
                         [&](uint lhs, uint rhs)
                         { return cross_member_bits(input[lhs]) < cross_member_bits(input[rhs]); });
        std::stable_sort(partial_descending.begin(),
                         partial_descending.end(),
                         [&](uint lhs, uint rhs)
                         { return cross_member_bits(input[lhs]) > cross_member_bits(input[rhs]); });

        auto input_buffer        = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto full_ascending_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto partial_ascending_pair_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto partial_ascending_pair_values = device.create_buffer<uint>(block_items);
        auto full_descending_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto partial_descending_pair_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto partial_descending_pair_values = device.create_buffer<uint>(block_items);
        auto partial_ascending_striped_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto full_ascending_striped_pair_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto full_ascending_striped_pair_values = device.create_buffer<uint>(block_items);
        auto partial_descending_striped_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto full_descending_striped_pair_keys = device.create_buffer<BlockRadixCompositeKey>(block_items);
        auto full_descending_striped_pair_values = device.create_buffer<uint>(block_items);

        Kernel1D kernel = [](BufferVar<BlockRadixCompositeKey> input_keys,
                             BufferVar<BlockRadixCompositeKey> full_ascending_keys_output,
                             BufferVar<BlockRadixCompositeKey> partial_ascending_pair_keys_output,
                             BufferVar<uint>                   partial_ascending_pair_values_output,
                             BufferVar<BlockRadixCompositeKey> full_descending_keys_output,
                             BufferVar<BlockRadixCompositeKey> partial_descending_pair_keys_output,
                             BufferVar<uint> partial_descending_pair_values_output,
                             BufferVar<BlockRadixCompositeKey> partial_ascending_striped_keys_output,
                             BufferVar<BlockRadixCompositeKey> full_ascending_striped_pair_keys_output,
                             BufferVar<uint> full_ascending_striped_pair_values_output,
                             BufferVar<BlockRadixCompositeKey> partial_descending_striped_keys_output,
                             BufferVar<BlockRadixCompositeKey> full_descending_striped_pair_keys_output,
                             BufferVar<uint> full_descending_striped_pair_values_output) noexcept
        {
            set_block_size(block_threads);
            UInt                                               tid = thread_id().x;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> full_ascending_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> partial_ascending_pair_local;
            ArrayVar<uint, items_per_thread>                   partial_ascending_pair_values_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> full_descending_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> partial_descending_pair_local;
            ArrayVar<uint, items_per_thread>                   partial_descending_pair_values_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> partial_ascending_striped_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> full_ascending_striped_pair_local;
            ArrayVar<uint, items_per_thread> full_ascending_striped_pair_values_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> partial_descending_striped_local;
            ArrayVar<BlockRadixCompositeKey, items_per_thread> full_descending_striped_pair_local;
            ArrayVar<uint, items_per_thread> full_descending_striped_pair_values_local;
            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt                        index               = tid * items_per_thread + item;
                Var<BlockRadixCompositeKey> key                 = input_keys.read(index);
                full_ascending_local[item]                      = key;
                partial_ascending_pair_local[item]              = key;
                partial_ascending_pair_values_local[item]       = index;
                full_descending_local[item]                     = key;
                partial_descending_pair_local[item]             = key;
                partial_descending_pair_values_local[item]      = index;
                partial_ascending_striped_local[item]           = key;
                full_ascending_striped_pair_local[item]         = key;
                full_ascending_striped_pair_values_local[item]  = index;
                partial_descending_striped_local[item]          = key;
                full_descending_striped_pair_local[item]        = key;
                full_descending_striped_pair_values_local[item] = index;
            }

            auto decomposer = [](const Var<BlockRadixCompositeKey>& key) noexcept
            { return std::tuple{key.primary, key.secondary}; };
            BlockRadixSort<BlockRadixCompositeKey, block_threads, items_per_thread, uint> sorter;
            sorter.Sort(full_ascending_local, decomposer);
            sorter.Sort(partial_ascending_pair_local, partial_ascending_pair_values_local, decomposer, 28u, 36u);
            sorter.SortDescending(full_descending_local, decomposer);
            sorter.SortDescending(
                partial_descending_pair_local, partial_descending_pair_values_local, decomposer, 28u, 36u);
            sorter.SortBlockedToStriped(partial_ascending_striped_local, decomposer, 28u, 36u);
            sorter.SortBlockedToStriped(full_ascending_striped_pair_local,
                                        full_ascending_striped_pair_values_local,
                                        decomposer);
            sorter.SortDescendingBlockedToStriped(partial_descending_striped_local, decomposer, 28u, 36u);
            sorter.SortDescendingBlockedToStriped(full_descending_striped_pair_local,
                                                  full_descending_striped_pair_values_local,
                                                  decomposer);

            for(uint item = 0u; item < items_per_thread; ++item)
            {
                UInt index = tid * items_per_thread + item;
                full_ascending_keys_output.write(index, full_ascending_local[item]);
                partial_ascending_pair_keys_output.write(index, partial_ascending_pair_local[item]);
                partial_ascending_pair_values_output.write(index, partial_ascending_pair_values_local[item]);
                full_descending_keys_output.write(index, full_descending_local[item]);
                partial_descending_pair_keys_output.write(index, partial_descending_pair_local[item]);
                partial_descending_pair_values_output.write(index, partial_descending_pair_values_local[item]);
                partial_ascending_striped_keys_output.write(index, partial_ascending_striped_local[item]);
                full_ascending_striped_pair_keys_output.write(index, full_ascending_striped_pair_local[item]);
                full_ascending_striped_pair_values_output.write(index, full_ascending_striped_pair_values_local[item]);
                partial_descending_striped_keys_output.write(index, partial_descending_striped_local[item]);
                full_descending_striped_pair_keys_output.write(index, full_descending_striped_pair_local[item]);
                full_descending_striped_pair_values_output.write(
                    index, full_descending_striped_pair_values_local[item]);
            }
        };

        auto                                  shader = device.compile(kernel);
        luisa::vector<BlockRadixCompositeKey> full_ascending_result(block_items);
        luisa::vector<BlockRadixCompositeKey> partial_ascending_pair_key_result(block_items);
        luisa::vector<uint>                   partial_ascending_pair_value_result(block_items);
        luisa::vector<BlockRadixCompositeKey> full_descending_result(block_items);
        luisa::vector<BlockRadixCompositeKey> partial_descending_pair_key_result(block_items);
        luisa::vector<uint>                   partial_descending_pair_value_result(block_items);
        luisa::vector<BlockRadixCompositeKey> partial_ascending_striped_result(block_items);
        luisa::vector<BlockRadixCompositeKey> full_ascending_striped_pair_key_result(block_items);
        luisa::vector<uint>                   full_ascending_striped_pair_value_result(block_items);
        luisa::vector<BlockRadixCompositeKey> partial_descending_striped_result(block_items);
        luisa::vector<BlockRadixCompositeKey> full_descending_striped_pair_key_result(block_items);
        luisa::vector<uint> full_descending_striped_pair_value_result(block_items);
        stream
            << input_buffer.copy_from(luisa::span{input})
            << shader(input_buffer,
                      full_ascending_keys,
                      partial_ascending_pair_keys,
                      partial_ascending_pair_values,
                      full_descending_keys,
                      partial_descending_pair_keys,
                      partial_descending_pair_values,
                      partial_ascending_striped_keys,
                      full_ascending_striped_pair_keys,
                      full_ascending_striped_pair_values,
                      partial_descending_striped_keys,
                      full_descending_striped_pair_keys,
                      full_descending_striped_pair_values)
                   .dispatch(block_threads)
            << full_ascending_keys.copy_to(luisa::span{full_ascending_result})
            << partial_ascending_pair_keys.copy_to(luisa::span{partial_ascending_pair_key_result})
            << partial_ascending_pair_values.copy_to(luisa::span{partial_ascending_pair_value_result})
            << full_descending_keys.copy_to(luisa::span{full_descending_result})
            << partial_descending_pair_keys.copy_to(luisa::span{partial_descending_pair_key_result})
            << partial_descending_pair_values.copy_to(luisa::span{partial_descending_pair_value_result})
            << partial_ascending_striped_keys.copy_to(luisa::span{partial_ascending_striped_result})
            << full_ascending_striped_pair_keys.copy_to(luisa::span{full_ascending_striped_pair_key_result})
            << full_ascending_striped_pair_values.copy_to(luisa::span{full_ascending_striped_pair_value_result})
            << partial_descending_striped_keys.copy_to(luisa::span{partial_descending_striped_result})
            << full_descending_striped_pair_keys.copy_to(luisa::span{full_descending_striped_pair_key_result})
            << full_descending_striped_pair_values.copy_to(luisa::span{full_descending_striped_pair_value_result})
            << synchronize();

        for(uint index = 0u; index < block_items; ++index)
        {
            expect(composite_equal(full_ascending_result[index], input[full_ascending[index]]));
            expect(composite_equal(partial_ascending_pair_key_result[index], input[partial_ascending[index]]));
            expect(partial_ascending_pair_value_result[index] == partial_ascending[index]);
            expect(composite_equal(full_descending_result[index], input[full_descending[index]]));
            expect(composite_equal(partial_descending_pair_key_result[index], input[partial_descending[index]]));
            expect(partial_descending_pair_value_result[index] == partial_descending[index]);

            uint lane          = index / items_per_thread;
            uint item          = index % items_per_thread;
            uint striped_index = lane + item * block_threads;
            expect(composite_equal(partial_ascending_striped_result[index],
                                   input[partial_ascending[striped_index]]));
            expect(composite_equal(full_ascending_striped_pair_key_result[index],
                                   input[full_ascending[striped_index]]));
            expect(full_ascending_striped_pair_value_result[index] == full_ascending[striped_index]);
            expect(composite_equal(partial_descending_striped_result[index],
                                   input[partial_descending[striped_index]]));
            expect(composite_equal(full_descending_striped_pair_key_result[index],
                                   input[full_descending[striped_index]]));
            expect(full_descending_striped_pair_value_result[index] == full_descending[striped_index]);
        }
    };
}
