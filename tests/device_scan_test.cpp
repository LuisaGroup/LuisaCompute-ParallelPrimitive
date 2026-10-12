// Tests for DeviceScan.
// Covers ordinary scans, scan-by-key, non-commutative operand ordering, and
// temporary-storage alignment for wide element types.

#include <luisa/core/basic_traits.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/config.h>
#include <algorithm>
#include <cstdint>
#include <lcpp/parallel_primitive.h>
#include <array>
#include <numeric>
#include <random>

#include <boost/ut.hpp>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;
int main(int argc, char* argv[])
{
    luisa::string_view ctx_dir;
    luisa::string_view backend_name;
#ifdef _WIN32
    backend_name = "cuda";
#elif __APPLE__
    backend_name = "metal";
#else
    backend_name = "cuda";
#endif

    // Parse command-line arguments: --ctx=./dir --backend=dx
    for(int i = 1; i < argc; ++i)
    {
        luisa::string_view arg(argv[i]);
        if(arg.starts_with("--ctx="))
        {
            ctx_dir = arg.substr(6);
            LUISA_INFO("Use context directory from command-line: {}", ctx_dir);
        }
        else if(arg.starts_with("--backend="))
        {
            backend_name = arg.substr(10);
            LUISA_INFO("Use backend from command-line: {}", backend_name);
        }
    }

    Context     context{ctx_dir.empty() ? argv[0] : ctx_dir};
    Device      device = context.create_device(backend_name);
    Stream      stream = device.create_stream();
    CommandList cmdlist;

    constexpr int32_t BLOCK_SIZE       = 256;
    constexpr int32_t ITEMS_PER_THREAD = 4;
    constexpr int32_t WARP_NUMS        = 32;
    constexpr int32_t MAX_NUM_LOGIC    = 24;

    using ScannerT = DeviceScan<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
    ScannerT scanner;
    scanner.create(device, &stream);

    "exclusive_scan"_test = [&]
    {
        for(int i = 0; i < MAX_NUM_LOGIC; i++)
        {
            const uint          array_size = 1 << i;
            luisa::vector<uint> input_data(array_size, 1);
            auto                in_buffer            = device.create_buffer<uint>(array_size);
            auto                exclusive_out_buffer = device.create_buffer<uint>(array_size);
            stream << in_buffer.copy_from(luisa::span{input_data}) << synchronize();

            // CUB-style: get temp storage size
            size_t temp_bytes  = ScannerT::GetTempStorageBytes<uint>(array_size);
            auto   temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

            scanner.ExclusiveSum(cmdlist,
                                 temp_buffer.view(),
                                 in_buffer.view(),
                                 exclusive_out_buffer.view(),
                                 in_buffer.size());
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> exclusive_result(array_size);
            stream << exclusive_out_buffer.copy_to(luisa::span{exclusive_result}) << synchronize();

            auto inclusive_out_buffer = device.create_buffer<uint>(array_size);

            // Reuse same temp_buffer for inclusive scan (same size requirement)
            scanner.InclusiveSum(cmdlist,
                                 temp_buffer.view(),
                                 in_buffer.view(),
                                 inclusive_out_buffer.view(),
                                 in_buffer.size());
            stream << cmdlist.commit() << synchronize();
            luisa::vector<uint> inclusive_result(array_size);
            stream << inclusive_out_buffer.copy_to(luisa::span{inclusive_result}) << synchronize();

            luisa::vector<uint> exclusive_expected(array_size);
            luisa::vector<uint> inclusive_expected(array_size);
            std::exclusive_scan(input_data.begin(), input_data.end(), exclusive_expected.begin(), 0);
            std::inclusive_scan(input_data.begin(), input_data.end(), inclusive_expected.begin());

            auto exclusive_expected_match =
                std::equal(exclusive_result.begin(), exclusive_result.end(), exclusive_expected.begin());
            auto inclusive_expected_match =
                std::equal(inclusive_result.begin(), inclusive_result.end(), inclusive_expected.begin());
            expect(exclusive_expected_match && inclusive_expected_match)
                << "Scan failed for array size " << i << "\n";
        }
    };

    "exclusive_scan_preserves_left_to_right_operand_order_across_tiles"_test = [&]
    {
        constexpr size_t   array_size = BLOCK_SIZE * ITEMS_PER_THREAD + 1u;
        luisa::vector<int> input_data(array_size, 7);
        luisa::vector<int> result(array_size);
        auto               input  = device.create_buffer<int>(array_size);
        auto               output = device.create_buffer<int>(array_size);
        auto               temp =
            device.create_buffer<uint>(bytes_to_uint_count(ScannerT::GetTempStorageBytes<int>(array_size)));

        CommandList commands;
        scanner.ExclusiveScan(
            commands,
            temp.view(),
            input.view(),
            output.view(),
            array_size,
            [](const Int& lhs, const Int&) noexcept { return lhs; },
            42);
        stream << input.copy_from(luisa::span{input_data}) << commands.commit()
               << output.copy_to(luisa::span{result}) << synchronize();

        for(auto value : result)
        {
            expect(value == 42);
        }
    };

    "device_scan_aligns_temporary_storage_for_double_values"_test = [&]
    {
        constexpr size_t      array_size = 17u;
        luisa::vector<double> input_data(array_size, 1.0);
        luisa::vector<double> result(array_size);
        auto                  input  = device.create_buffer<double>(array_size);
        auto                  output = device.create_buffer<double>(array_size);
        auto                  temp   = device.create_buffer<uint>(
            bytes_to_uint_count(ScannerT::GetTempStorageBytes<double>(array_size)));

        CommandList commands;
        scanner.InclusiveSum(commands, temp.view(), input.view(), output.view(), array_size);
        stream << input.copy_from(luisa::span{input_data}) << commands.commit()
               << output.copy_to(luisa::span{result}) << synchronize();

        for(size_t index = 0u; index < array_size; ++index)
        {
            expect(result[index] == static_cast<double>(index + 1u));
        }
    };

    "inclusive_scan_init_applies_initial_value_in_place"_test = [&]
    {
        const std::array<int, 4u> input_data{1, 2, 3, 4};
        const std::array<int, 4u> expected{11, 13, 16, 20};
        auto                      data = device.create_buffer<int>(input_data.size());
        auto                      temp = device.create_buffer<uint>(
            bytes_to_uint_count(ScannerT::GetTempStorageBytes<int>(input_data.size())));

        CommandList commands;
        scanner.InclusiveScanInit(
            commands,
            temp.view(),
            data.view(),
            data.view(),
            input_data.size(),
            [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; },
            10);
        scanner.InclusiveScanInit(
            commands,
            BufferView<uint>{},
            BufferView<int>{},
            BufferView<int>{},
            0u,
            [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; },
            10);

        std::array<int, 4u> result{};
        stream << data.copy_from(luisa::span{input_data}) << commands.commit()
               << data.copy_to(luisa::span{result}) << synchronize();
        for(size_t index = 0u; index < expected.size(); ++index)
        {
            expect(result[index] == expected[index]);
        }
    };


    "exclusive_scan_by_key"_test = [&]
    {
        for(auto i = 5; i < MAX_NUM_LOGIC; i++)
        {
            const uint           array_size = 1 << i;
            luisa::vector<int32> input_data(array_size, 1);
            auto                 key_buffer   = device.create_buffer<int32>(array_size);
            auto                 value_buffer = device.create_buffer<int32>(array_size);

            constexpr int items_per_segment = 100;
            const int     segments = (array_size + items_per_segment - 1) / items_per_segment;

            luisa::vector<int32> input_keys(array_size);
            for(auto i = 0; i < array_size; i++)
            {
                input_keys[i] = i / items_per_segment;
            }

            LUISA_INFO("Array size: {}, Items per segment: {}, Total segments: {}", array_size, items_per_segment, segments);

            stream << key_buffer.copy_from(luisa::span{input_keys}) << synchronize();
            stream << value_buffer.copy_from(luisa::span{input_data}) << synchronize();

            auto value_out_buffer = device.create_buffer<int32>(array_size);

            // CUB-style: get temp storage size for ScanByKey
            size_t temp_bytes  = ScannerT::GetScanByKeyTempStorageBytes<int32, int32>(array_size);
            auto   temp_buffer = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));

            scanner.ExclusiveSumByKey(cmdlist,
                                      temp_buffer.view(),
                                      key_buffer.view(),
                                      value_buffer.view(),
                                      value_out_buffer.view(),
                                      key_buffer.size());
            stream << cmdlist.commit() << synchronize();

            luisa::vector<int32> result(array_size);
            stream << value_out_buffer.copy_to(luisa::span{result}) << synchronize();
            luisa::vector<int32> expected(array_size);
            std::exclusive_scan(input_data.begin(), input_data.end(), expected.begin(), 0);

            for(auto i = 0; i < segments; i++)
            {
                luisa::vector<int32> expect_segment(items_per_segment, 0);
                std::exclusive_scan(input_data.begin() + i * items_per_segment,
                                    input_data.begin()
                                        + std::min((i + 1) * items_per_segment, static_cast<int>(array_size)),
                                    expect_segment.begin(),
                                    0);
                for(auto j = i * items_per_segment; j < (i + 1) * items_per_segment && j < array_size; j++)
                {
                    expect(expect_segment[j - i * items_per_segment] == result[j]);
                }
            }
        }
    };

    "inclusive_scan_by_key_handles_distinct_types_across_three_tiles"_test = [&]
    {
        constexpr size_t            tile_size  = BLOCK_SIZE * ITEMS_PER_THREAD;
        constexpr size_t            array_size = tile_size * 2u + 2u;
        luisa::vector<int>          keys(array_size);
        luisa::vector<luisa::slong> values(array_size, 1);
        luisa::vector<luisa::slong> expected(array_size);
        luisa::slong                segment_sum = 0;
        for(size_t index = 0u; index < array_size; ++index)
        {
            keys[index] = index < tile_size + 1u ? 1 : index < tile_size * 2u + 1u ? 2 : 3;
            if(index == 0u || keys[index] != keys[index - 1u])
            {
                segment_sum = 0;
            }
            expected[index] = ++segment_sum;
        }

        auto key_buffer    = device.create_buffer<int>(array_size);
        auto value_buffer  = device.create_buffer<luisa::slong>(array_size);
        auto output_buffer = device.create_buffer<luisa::slong>(array_size);
        auto temp_buffer   = device.create_buffer<uint>(
            bytes_to_uint_count(ScannerT::GetScanByKeyTempStorageBytes<int, luisa::slong>(array_size)));

        CommandList commands;
        scanner.InclusiveSumByKey(
            commands, temp_buffer.view(), key_buffer.view(), value_buffer.view(), output_buffer.view(), array_size);
        luisa::vector<luisa::slong> result(array_size);
        stream << key_buffer.copy_from(luisa::span{keys}) << value_buffer.copy_from(luisa::span{values})
               << commands.commit() << output_buffer.copy_to(luisa::span{result}) << synchronize();

        for(size_t index = 0u; index < array_size; ++index)
        {
            expect(result[index] == expected[index]);
        }
    };

    "inclusive_scan_by_key_honors_custom_key_equality"_test = [&]
    {
        luisa::vector<int> keys{1, 3, 5, 2, 4};
        luisa::vector<int> values(keys.size(), 1);
        luisa::vector<int> result(keys.size());
        auto               key_buffer    = device.create_buffer<int>(keys.size());
        auto               value_buffer  = device.create_buffer<int>(values.size());
        auto               output_buffer = device.create_buffer<int>(values.size());
        auto               temp_buffer   = device.create_buffer<uint>(
            bytes_to_uint_count(ScannerT::GetScanByKeyTempStorageBytes<int, int>(keys.size())));

        CommandList commands;
        scanner.InclusiveSumByKey(commands,
                                  temp_buffer.view(),
                                  key_buffer.view(),
                                  value_buffer.view(),
                                  output_buffer.view(),
                                  keys.size(),
                                  [](const Int& lhs, const Int& rhs) noexcept
                                  { return (lhs & 1) == (rhs & 1); });
        stream << key_buffer.copy_from(luisa::span{keys}) << value_buffer.copy_from(luisa::span{values})
               << commands.commit() << output_buffer.copy_to(luisa::span{result}) << synchronize();

        std::array<int, 5u> expected{1, 2, 3, 1, 2};
        for(size_t index = 0u; index < expected.size(); ++index)
        {
            expect(result[index] == expected[index]);
        }
    };
}
