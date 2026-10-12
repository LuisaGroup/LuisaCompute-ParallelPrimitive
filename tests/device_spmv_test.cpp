// Tests for DeviceSpmv::CsrMV.
// Covers rectangular matrices, empty and long rows, output overwrite, and empty work.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <cmath>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

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

    DeviceSpmv<> spmv;
    spmv.create(device);

    "device_spmv_multiplies_rectangular_integer_csr"_test = [&]
    {
        constexpr uint            num_rows = 8u;
        constexpr uint            num_cols = 13u;
        const luisa::vector<uint> row_lengths{0u, 1u, 5u, 33u, 2u, 64u, 0u, 7u};
        luisa::vector<int>        row_offsets(num_rows + 1u);
        for(uint row = 0u; row < num_rows; ++row)
        {
            row_offsets[row + 1u] = row_offsets[row] + static_cast<int>(row_lengths[row]);
        }
        auto               num_nonzeros = static_cast<uint>(row_offsets.back());
        luisa::vector<int> column_indices(num_nonzeros);
        luisa::vector<int> values(num_nonzeros);
        for(uint row = 0u; row < num_rows; ++row)
        {
            for(uint item = 0u; item < row_lengths[row]; ++item)
            {
                auto index            = static_cast<uint>(row_offsets[row]) + item;
                column_indices[index] = static_cast<int>((row * 5u + item * 7u) % num_cols);
                values[index]         = static_cast<int>((row + item * 3u) % 9u) - 4;
            }
        }
        luisa::vector<int> vector_x(num_cols);
        for(uint column = 0u; column < num_cols; ++column)
        {
            vector_x[column] = static_cast<int>(column) - 6;
        }
        luisa::vector<int> expected(num_rows);
        for(uint row = 0u; row < num_rows; ++row)
        {
            for(int item = row_offsets[row]; item < row_offsets[row + 1u]; ++item)
            {
                expected[row] += values[item] * vector_x[column_indices[item]];
            }
        }

        auto               values_buffer         = device.create_buffer<int>(num_nonzeros);
        auto               row_offsets_buffer    = device.create_buffer<int>(num_rows + 1u);
        auto               column_indices_buffer = device.create_buffer<int>(num_nonzeros);
        auto               vector_x_buffer       = device.create_buffer<int>(num_cols);
        auto               vector_y_buffer       = device.create_buffer<int>(num_rows);
        luisa::vector<int> initial_y(num_rows, 0x12345678);

        CommandList commands;
        spmv.CsrMV(commands,
                   BufferView<uint>{},
                   values_buffer.view(),
                   row_offsets_buffer.view(),
                   column_indices_buffer.view(),
                   vector_x_buffer.view(),
                   vector_y_buffer.view(),
                   num_rows,
                   num_cols,
                   num_nonzeros);
        stream << values_buffer.copy_from(luisa::span{values})
               << row_offsets_buffer.copy_from(luisa::span{row_offsets})
               << column_indices_buffer.copy_from(luisa::span{column_indices})
               << vector_x_buffer.copy_from(luisa::span{vector_x})
               << vector_y_buffer.copy_from(luisa::span{initial_y}) << commands.commit();

        luisa::vector<int> result(num_rows);
        stream << vector_y_buffer.copy_to(luisa::span{result}) << synchronize();
        for(uint row = 0u; row < num_rows; ++row)
        {
            expect(result[row] == expected[row]);
        }
    };

    "device_spmv_supports_float_values_and_unsigned_indices"_test = [&]
    {
        constexpr uint             num_rows = 4u;
        constexpr uint             num_cols = 5u;
        const luisa::vector<uint>  row_offsets{0u, 3u, 3u, 6u, 8u};
        const luisa::vector<uint>  column_indices{0u, 2u, 4u, 1u, 2u, 3u, 0u, 4u};
        const luisa::vector<float> values{1.5f, -2.0f, 0.25f, 3.0f, 1.25f, -0.5f, 2.0f, 4.0f};
        const luisa::vector<float> vector_x{2.0f, -1.0f, 0.5f, 3.0f, -2.0f};
        luisa::vector<float>       expected(num_rows);
        for(uint row = 0u; row < num_rows; ++row)
        {
            for(uint item = row_offsets[row]; item < row_offsets[row + 1u]; ++item)
            {
                expected[row] += values[item] * vector_x[column_indices[item]];
            }
        }

        auto values_buffer         = device.create_buffer<float>(values.size());
        auto row_offsets_buffer    = device.create_buffer<uint>(row_offsets.size());
        auto column_indices_buffer = device.create_buffer<uint>(column_indices.size());
        auto vector_x_buffer       = device.create_buffer<float>(vector_x.size());
        auto vector_y_buffer       = device.create_buffer<float>(num_rows);

        CommandList commands;
        spmv.CsrMV(commands,
                   BufferView<uint>{},
                   values_buffer.view(),
                   row_offsets_buffer.view(),
                   column_indices_buffer.view(),
                   vector_x_buffer.view(),
                   vector_y_buffer.view(),
                   num_rows,
                   num_cols,
                   values.size());
        stream << values_buffer.copy_from(luisa::span{values})
               << row_offsets_buffer.copy_from(luisa::span{row_offsets})
               << column_indices_buffer.copy_from(luisa::span{column_indices})
               << vector_x_buffer.copy_from(luisa::span{vector_x}) << commands.commit();

        luisa::vector<float> result(num_rows);
        stream << vector_y_buffer.copy_to(luisa::span{result}) << synchronize();
        for(uint row = 0u; row < num_rows; ++row)
        {
            expect(std::abs(result[row] - expected[row]) < 1.0e-5f);
        }
    };

    "device_spmv_handles_empty_work"_test = [&]
    {
        CommandList commands;
        spmv.CsrMV(commands,
                   BufferView<uint>{},
                   BufferView<float>{},
                   BufferView<int>{},
                   BufferView<int>{},
                   BufferView<float>{},
                   BufferView<float>{},
                   0u,
                   0u,
                   0u);
        stream << commands.commit() << synchronize();
        expect(DeviceSpmv<>::GetTempStorageBytes(0u, 0u, 0u) == 0u);
    };

    "device_spmv_zero_matrix_does_not_bind_empty_inputs"_test = [&]
    {
        constexpr uint           num_rows = 5u;
        constexpr uint           num_cols = 7u;
        const luisa::vector<int> row_offsets(num_rows + 1u, 0);
        luisa::vector<float>     initial(num_rows, 42.0f);
        auto                     row_offsets_buffer = device.create_buffer<int>(row_offsets.size());
        auto                     vector_y_buffer    = device.create_buffer<float>(num_rows);

        CommandList commands;
        spmv.CsrMV(commands,
                   BufferView<uint>{},
                   BufferView<float>{},
                   row_offsets_buffer.view(),
                   BufferView<int>{},
                   BufferView<float>{},
                   vector_y_buffer.view(),
                   num_rows,
                   num_cols,
                   0u);
        stream << row_offsets_buffer.copy_from(luisa::span{row_offsets})
               << vector_y_buffer.copy_from(luisa::span{initial}) << commands.commit();

        luisa::vector<float> result(num_rows);
        stream << vector_y_buffer.copy_to(luisa::span{result}) << synchronize();
        for(auto value : result)
        {
            expect(value == 0.0f);
        }
    };
}
