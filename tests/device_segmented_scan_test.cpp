// Tests for DeviceSegmentedScan.
// Covers every pinned method group, independent output offsets, empty segments,
// non-commutative operators, and exact in-place execution.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

namespace
{
constexpr uint item_count    = 14u;
constexpr uint segment_count = 4u;
constexpr int  sentinel      = -777;

const luisa::vector<int>  input_values{100, 1, 2, 3, 5, 6, 7, 200, 9, 10, 11, 12, 300, 400};
const luisa::vector<uint> input_begins{1u, 4u, 4u, 8u};
const luisa::vector<uint> input_ends{4u, 4u, 7u, 12u};
const luisa::vector<uint> packed_begins{0u, 4u, 6u, 10u};

enum class ScanMode
{
    EXCLUSIVE_SUM,
    EXCLUSIVE_RIGHT,
    INCLUSIVE_SUM,
    INCLUSIVE_RIGHT,
    INCLUSIVE_INIT_SUM
};

[[nodiscard]] luisa::vector<int> expected_values(const luisa::vector<uint>& output_begins, ScanMode mode)
{
    luisa::vector<int> expected(item_count, sentinel);
    for(uint segment = 0u; segment < segment_count; ++segment)
    {
        uint input_begin  = input_begins[segment];
        uint input_end    = input_ends[segment];
        uint output_begin = output_begins[segment];
        if(input_begin == input_end)
        {
            continue;
        }

        if(mode == ScanMode::EXCLUSIVE_SUM || mode == ScanMode::EXCLUSIVE_RIGHT)
        {
            int aggregate = mode == ScanMode::EXCLUSIVE_SUM ? 0 : 99;
            for(uint index = input_begin; index < input_end; ++index)
            {
                expected[output_begin + index - input_begin] = aggregate;
                aggregate = mode == ScanMode::EXCLUSIVE_SUM ? aggregate + input_values[index] :
                                                              input_values[index];
            }
        }
        else
        {
            int aggregate = mode == ScanMode::INCLUSIVE_INIT_SUM ? 10 : input_values[input_begin];
            uint first_index = mode == ScanMode::INCLUSIVE_INIT_SUM ? input_begin : input_begin + 1u;
            if(mode != ScanMode::INCLUSIVE_INIT_SUM)
            {
                expected[output_begin] = aggregate;
            }
            for(uint index = first_index; index < input_end; ++index)
            {
                aggregate = mode == ScanMode::INCLUSIVE_RIGHT ? input_values[index] :
                                                                aggregate + input_values[index];
                expected[output_begin + index - input_begin] = aggregate;
            }
        }
    }
    return expected;
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

    DeviceSegmentedScan<> segmented_scan;
    segmented_scan.create(device);

    "device_segmented_scan_covers_same_and_independent_output_offsets"_test = [&]
    {
        auto               input                 = device.create_buffer<int>(item_count);
        auto               begins                = device.create_buffer<uint>(segment_count);
        auto               ends                  = device.create_buffer<uint>(segment_count);
        auto               output_begins         = device.create_buffer<uint>(segment_count);
        auto               exclusive_sum         = device.create_buffer<int>(item_count);
        auto               exclusive_scan        = device.create_buffer<int>(item_count);
        auto               inclusive_sum         = device.create_buffer<int>(item_count);
        auto               inclusive_scan        = device.create_buffer<int>(item_count);
        auto               inclusive_init        = device.create_buffer<int>(item_count);
        auto               packed_exclusive_sum  = device.create_buffer<int>(item_count);
        auto               packed_exclusive_scan = device.create_buffer<int>(item_count);
        auto               packed_inclusive_sum  = device.create_buffer<int>(item_count);
        auto               packed_inclusive_scan = device.create_buffer<int>(item_count);
        auto               packed_inclusive_init = device.create_buffer<int>(item_count);
        luisa::vector<int> initial(item_count, sentinel);

        stream << input.copy_from(luisa::span{input_values})
               << begins.copy_from(luisa::span{input_begins}) << ends.copy_from(luisa::span{input_ends})
               << output_begins.copy_from(luisa::span{packed_begins})
               << exclusive_sum.copy_from(luisa::span{initial})
               << exclusive_scan.copy_from(luisa::span{initial})
               << inclusive_sum.copy_from(luisa::span{initial})
               << inclusive_scan.copy_from(luisa::span{initial})
               << inclusive_init.copy_from(luisa::span{initial})
               << packed_exclusive_sum.copy_from(luisa::span{initial})
               << packed_exclusive_scan.copy_from(luisa::span{initial})
               << packed_inclusive_sum.copy_from(luisa::span{initial})
               << packed_inclusive_scan.copy_from(luisa::span{initial})
               << packed_inclusive_init.copy_from(luisa::span{initial}) << synchronize();

        auto        right = [](const Int&, const Int& rhs) noexcept { return rhs; };
        auto        add   = [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; };
        CommandList commands;
        segmented_scan.ExclusiveSegmentedSum(
            commands, BufferView<uint>{}, input.view(), exclusive_sum.view(), begins.view(), ends.view(), segment_count);
        segmented_scan.ExclusiveSegmentedScan(
            commands, BufferView<uint>{}, input.view(), exclusive_scan.view(), begins.view(), ends.view(), segment_count, right, 99);
        segmented_scan.InclusiveSegmentedSum(
            commands, BufferView<uint>{}, input.view(), inclusive_sum.view(), begins.view(), ends.view(), segment_count);
        segmented_scan.InclusiveSegmentedScan(
            commands, BufferView<uint>{}, input.view(), inclusive_scan.view(), begins.view(), ends.view(), segment_count, right);
        segmented_scan.InclusiveSegmentedScanInit(
            commands, BufferView<uint>{}, input.view(), inclusive_init.view(), begins.view(), ends.view(), segment_count, add, 10);

        segmented_scan.ExclusiveSegmentedSum(commands,
                                             BufferView<uint>{},
                                             input.view(),
                                             packed_exclusive_sum.view(),
                                             begins.view(),
                                             ends.view(),
                                             output_begins.view(),
                                             segment_count);
        segmented_scan.ExclusiveSegmentedScan(commands,
                                              BufferView<uint>{},
                                              input.view(),
                                              packed_exclusive_scan.view(),
                                              begins.view(),
                                              ends.view(),
                                              output_begins.view(),
                                              segment_count,
                                              right,
                                              99);
        segmented_scan.InclusiveSegmentedSum(commands,
                                             BufferView<uint>{},
                                             input.view(),
                                             packed_inclusive_sum.view(),
                                             begins.view(),
                                             ends.view(),
                                             output_begins.view(),
                                             segment_count);
        segmented_scan.InclusiveSegmentedScan(commands,
                                              BufferView<uint>{},
                                              input.view(),
                                              packed_inclusive_scan.view(),
                                              begins.view(),
                                              ends.view(),
                                              output_begins.view(),
                                              segment_count,
                                              right);
        segmented_scan.InclusiveSegmentedScanInit(commands,
                                                  BufferView<uint>{},
                                                  input.view(),
                                                  packed_inclusive_init.view(),
                                                  begins.view(),
                                                  ends.view(),
                                                  output_begins.view(),
                                                  segment_count,
                                                  add,
                                                  10);
        stream << commands.commit() << synchronize();

        auto check = [&](Buffer<int>& buffer, const luisa::vector<int>& expected)
        {
            luisa::vector<int> result(item_count);
            stream << buffer.copy_to(luisa::span{result}) << synchronize();
            for(uint index = 0u; index < item_count; ++index)
            {
                expect(result[index] == expected[index]);
            }
        };

        check(exclusive_sum, expected_values(input_begins, ScanMode::EXCLUSIVE_SUM));
        check(exclusive_scan, expected_values(input_begins, ScanMode::EXCLUSIVE_RIGHT));
        check(inclusive_sum, expected_values(input_begins, ScanMode::INCLUSIVE_SUM));
        check(inclusive_scan, expected_values(input_begins, ScanMode::INCLUSIVE_RIGHT));
        check(inclusive_init, expected_values(input_begins, ScanMode::INCLUSIVE_INIT_SUM));
        check(packed_exclusive_sum, expected_values(packed_begins, ScanMode::EXCLUSIVE_SUM));
        check(packed_exclusive_scan, expected_values(packed_begins, ScanMode::EXCLUSIVE_RIGHT));
        check(packed_inclusive_sum, expected_values(packed_begins, ScanMode::INCLUSIVE_SUM));
        check(packed_inclusive_scan, expected_values(packed_begins, ScanMode::INCLUSIVE_RIGHT));
        check(packed_inclusive_init, expected_values(packed_begins, ScanMode::INCLUSIVE_INIT_SUM));
    };

    "device_segmented_scan_supports_exact_in_place_and_zero_segments"_test = [&]
    {
        expect(DeviceSegmentedScan<>::GetTempStorageBytes<int>(item_count, segment_count) == 0u);
        auto values = device.create_buffer<int>(item_count);
        auto begins = device.create_buffer<uint>(segment_count);
        auto ends   = device.create_buffer<uint>(segment_count);

        CommandList commands;
        segmented_scan.InclusiveSegmentedSum(
            commands, BufferView<uint>{}, values.view(), values.view(), begins.view(), ends.view(), segment_count);
        segmented_scan.InclusiveSegmentedSum(
            commands, BufferView<uint>{}, values.view(), values.view(), BufferView<uint>{}, BufferView<uint>{}, 0u);

        luisa::vector<int> result(item_count);
        stream << values.copy_from(luisa::span{input_values})
               << begins.copy_from(luisa::span{input_begins}) << ends.copy_from(luisa::span{input_ends})
               << commands.commit() << values.copy_to(luisa::span{result}) << synchronize();

        auto expected = expected_values(input_begins, ScanMode::INCLUSIVE_SUM);
        for(uint index = 0u; index < item_count; ++index)
        {
            int expected_value = expected[index] == sentinel ? input_values[index] : expected[index];
            expect(result[index] == expected_value);
        }
    };
}
