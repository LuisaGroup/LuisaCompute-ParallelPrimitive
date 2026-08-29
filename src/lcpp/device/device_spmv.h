/*
 * @Author: Ligo
 * @Date: 2026-08-30
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <limits>
#include <lcpp/common/type_trait.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/warp_reduce.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE>
class DeviceSpmv : public LuisaModule
{
  private:
    static constexpr size_t WARP_SIZE = details::WARP_SIZE;
    static_assert(BLOCK_SIZE > 0u && BLOCK_SIZE % WARP_SIZE == 0u,
                  "DeviceSpmv block size must be a positive multiple of the physical warp size.");

    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_spmv_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> m_zero_shaders;

  public:
    DeviceSpmv()  = default;
    ~DeviceSpmv() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t, size_t) noexcept
    {
        return 0u;
    }

    template <NumericT ValueT, std::integral IndexT>
    void CsrMV(CommandList& cmdlist,
               BufferView<uint>,
               BufferView<ValueT> values,
               BufferView<IndexT> row_offsets,
               BufferView<IndexT> column_indices,
               BufferView<ValueT> vector_x,
               BufferView<ValueT> vector_y,
               size_t             num_rows,
               size_t             num_cols,
               size_t             num_nonzeros)
    {
        validate_views(values, row_offsets, column_indices, vector_x, vector_y, num_rows, num_cols, num_nonzeros);
        if(num_rows == 0u)
        {
            return;
        }
        if(num_nonzeros == 0u)
        {
            dispatch_zero<ValueT>(cmdlist, vector_y.subview(0u, num_rows), num_rows);
            return;
        }

        using SpmvShader =
            Shader<1, Buffer<ValueT>, Buffer<IndexT>, Buffer<IndexT>, Buffer<ValueT>, Buffer<ValueT>, uint, uint>;
        luisa::string key{"device_spmv_csr_mv+"};
        key += Type::of<ValueT>()->description();
        key += "+";
        key += Type::of<IndexT>()->description();
        auto shader_it = m_spmv_shaders.find(key);
        if(shader_it == m_spmv_shaders.end())
        {
            U<SpmvShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<ValueT> values_buffer,
                            BufferVar<IndexT> row_offsets_buffer,
                            BufferVar<IndexT> column_indices_buffer,
                            BufferVar<ValueT> vector_x_buffer,
                            BufferVar<ValueT> vector_y_buffer,
                            UInt              num_cols_var,
                            UInt              num_nonzeros_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt thread    = dispatch_id().x;
                             UInt row       = thread / UInt(WARP_SIZE);
                             UInt lane      = thread % UInt(WARP_SIZE);
                             UInt row_begin = compute::cast<uint>(row_offsets_buffer.read(row));
                             UInt row_end = compute::cast<uint>(row_offsets_buffer.read(row + 1u));
                             Var<ValueT> lane_sum = ValueT{0};
                             Bool valid_row = row_begin <= row_end & row_end <= num_nonzeros_var;
                             $if(valid_row)
                             {
                                 $for(item, row_begin + lane, row_end, UInt(WARP_SIZE))
                                 {
                                     UInt column = compute::cast<uint>(column_indices_buffer.read(item));
                                     $if(column < num_cols_var)
                                     {
                                         lane_sum += values_buffer.read(item) * vector_x_buffer.read(column);
                                     };
                                 };
                             };
                             WarpReduce<ValueT, WARP_SIZE> warp_reduce;
                             Var<ValueT>                   row_sum = warp_reduce.Sum(lane_sum);
                             $if(lane == 0u)
                             {
                                 vector_y_buffer.write(row, row_sum);
                             };
                         });
            m_spmv_shaders.try_emplace(key, std::move(shader));
            shader_it = m_spmv_shaders.find(key);
        }

        auto shader = reinterpret_cast<SpmvShader*>(&(*shader_it->second));
        cmdlist << (*shader)(values,
                             row_offsets,
                             column_indices,
                             vector_x,
                             vector_y,
                             static_cast<uint>(num_cols),
                             static_cast<uint>(num_nonzeros))
                       .dispatch(num_rows * WARP_SIZE);
    }

  private:
    template <NumericT ValueT>
    void dispatch_zero(CommandList& cmdlist, BufferView<ValueT> vector_y, size_t num_rows)
    {
        using ZeroShader = Shader<1, Buffer<ValueT>>;
        luisa::string key{"device_spmv_zero+"};
        key += Type::of<ValueT>()->description();
        auto shader_it = m_zero_shaders.find(key);
        if(shader_it == m_zero_shaders.end())
        {
            U<ZeroShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<ValueT> vector_y_buffer) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             vector_y_buffer.write(dispatch_id().x, ValueT{0});
                         });
            m_zero_shaders.try_emplace(key, std::move(shader));
            shader_it = m_zero_shaders.find(key);
        }
        auto shader = reinterpret_cast<ZeroShader*>(&(*shader_it->second));
        cmdlist << (*shader)(vector_y).dispatch(num_rows);
    }

    template <NumericT ValueT, std::integral IndexT>
    void validate_views(BufferView<ValueT> values,
                        BufferView<IndexT> row_offsets,
                        BufferView<IndexT> column_indices,
                        BufferView<ValueT> vector_x,
                        BufferView<ValueT> vector_y,
                        size_t             num_rows,
                        size_t             num_cols,
                        size_t             num_nonzeros) const
    {
        LUISA_ASSERT(m_created, "DeviceSpmv::create must be called before dispatch.");
        LUISA_ASSERT(num_rows <= std::numeric_limits<uint>::max() / WARP_SIZE, "Too many rows for DeviceSpmv.");
        LUISA_ASSERT(num_cols <= std::numeric_limits<uint>::max(), "Too many columns for DeviceSpmv.");
        LUISA_ASSERT(num_nonzeros <= std::numeric_limits<uint>::max(), "Too many nonzero values for DeviceSpmv.");
        if(num_rows == 0u)
        {
            return;
        }
        LUISA_ASSERT(row_offsets.size() >= num_rows + 1u, "DeviceSpmv row-offset view is too small.");
        LUISA_ASSERT(vector_y.size() >= num_rows, "DeviceSpmv output-vector view is too small.");
        if(num_nonzeros == 0u)
        {
            return;
        }
        LUISA_ASSERT(values.size() >= num_nonzeros, "DeviceSpmv value view is too small.");
        LUISA_ASSERT(column_indices.size() >= num_nonzeros, "DeviceSpmv column-index view is too small.");
        LUISA_ASSERT(vector_x.size() >= num_cols, "DeviceSpmv input-vector view is too small.");
        LUISA_ASSERT(num_cols > 0u || num_nonzeros == 0u,
                     "A DeviceSpmv matrix with zero columns cannot contain nonzero values.");
    }
};

}  // namespace luisa::parallel_primitive
