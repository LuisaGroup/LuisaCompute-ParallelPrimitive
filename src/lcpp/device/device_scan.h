/*
 * @Author: Ligo 
 * @Date: 2025-10-09 09:52:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-07 00:12:25
 */


#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/scan.h>
#include <lcpp/device/details/single_pass_scan_operator.h>
#include <lcpp/device/details/scan_by_key.h>

namespace luisa::parallel_primitive
{

using namespace luisa::compute;
template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceScan : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;
    bool   m_created = false;

    template <typename T>
    struct ScanTempViews
    {
        BufferView<uint> tile_status;
        BufferView<T>    tile_partial;
        BufferView<T>    tile_inclusive;
    };

    template <typename KeyT, typename ValueT>
    struct ScanByKeyTempViews
    {
        using FlagValuePairT = KeyValuePair<int, ValueT>;

        BufferView<uint>           tile_status;
        BufferView<FlagValuePairT> tile_partial;
        BufferView<FlagValuePairT> tile_inclusive;
        BufferView<KeyT>           previous_keys;
    };

    [[nodiscard]] static size_t num_tiles_for(size_t num_items, size_t block_size)
    {
        LUISA_ASSERT(block_size > 0u && ITEMS_PER_THREAD > 0u, "DeviceScan tile dimensions must be non-zero.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceScan.");
        return std::max<size_t>(1u, ceil_div(num_items, block_size * ITEMS_PER_THREAD));
    }

    template <typename T>
    [[nodiscard]] static size_t scan_temp_storage_bytes(size_t num_items, size_t block_size)
    {
        size_t tile_count = details::WARP_SIZE + num_tiles_for(num_items, block_size);
        size_t bytes      = tile_count * sizeof(uint);
        bytes             = align_up_uint(bytes, alignof(T));
        bytes += bytes_to_uint_count(tile_count * sizeof(T)) * sizeof(uint);
        bytes = align_up_uint(bytes, alignof(T));
        bytes += bytes_to_uint_count(tile_count * sizeof(T)) * sizeof(uint);
        return bytes;
    }

    template <typename KeyT, typename ValueT>
    [[nodiscard]] static size_t scan_by_key_temp_storage_bytes(size_t num_items, size_t block_size)
    {
        using FlagValuePairT = KeyValuePair<int, ValueT>;
        size_t num_tiles     = num_tiles_for(num_items, block_size);
        size_t tile_count    = details::WARP_SIZE + num_tiles;
        size_t bytes         = tile_count * sizeof(uint);
        bytes                = align_up_uint(bytes, alignof(FlagValuePairT));
        bytes += bytes_to_uint_count(tile_count * sizeof(FlagValuePairT)) * sizeof(uint);
        bytes = align_up_uint(bytes, alignof(FlagValuePairT));
        bytes += bytes_to_uint_count(tile_count * sizeof(FlagValuePairT)) * sizeof(uint);
        bytes = align_up_uint(bytes, alignof(KeyT));
        bytes += bytes_to_uint_count(num_tiles * sizeof(KeyT)) * sizeof(uint);
        return bytes;
    }

    template <typename T>
    [[nodiscard]] static ScanTempViews<T> scan_temp_views(BufferView<uint> temp_storage, size_t num_items, size_t block_size)
    {
        size_t tile_count   = details::WARP_SIZE + num_tiles_for(num_items, block_size);
        size_t offset_bytes = 0u;

        auto tile_status = temp_storage.subview(0u, tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(T));

        size_t partial_uint_count = bytes_to_uint_count(tile_count * sizeof(T));
        auto   tile_partial =
            temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<T>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(T));

        size_t inclusive_uint_count = bytes_to_uint_count(tile_count * sizeof(T));
        auto   tile_inclusive =
            temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<T>();
        return {tile_status, tile_partial, tile_inclusive};
    }

    template <typename KeyT, typename ValueT>
    [[nodiscard]] static ScanByKeyTempViews<KeyT, ValueT> scan_by_key_temp_views(BufferView<uint> temp_storage,
                                                                                 size_t num_items,
                                                                                 size_t block_size)
    {
        using FlagValuePairT = KeyValuePair<int, ValueT>;
        size_t num_tiles     = num_tiles_for(num_items, block_size);
        size_t tile_count    = details::WARP_SIZE + num_tiles;
        size_t offset_bytes  = 0u;

        auto tile_status = temp_storage.subview(0u, tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(FlagValuePairT));

        size_t partial_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_partial =
            temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(FlagValuePairT));

        size_t inclusive_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_inclusive =
            temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();
        offset_bytes += inclusive_uint_count * sizeof(uint);
        offset_bytes = align_up_uint(offset_bytes, alignof(KeyT));

        size_t previous_key_uint_count = bytes_to_uint_count(num_tiles * sizeof(KeyT));
        auto   previous_keys =
            temp_storage.subview(offset_bytes / sizeof(uint), previous_key_uint_count).template as<KeyT>();
        return {tile_status, tile_partial, tile_inclusive, previous_keys};
    }

    template <NumericT T>
    void validate_scan_views(BufferView<uint> temp_storage, BufferView<T> input, BufferView<T> output, size_t num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceScan::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceScan.");
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(input.size() >= num_items, "DeviceScan input view is too small.");
        LUISA_ASSERT(output.size() >= num_items, "DeviceScan output view is too small.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(scan_temp_storage_bytes<T>(num_items, m_block_size)),
                     "DeviceScan temporary storage is too small.");
    }

    template <NumericT KeyT, NumericT ValueT>
    void validate_scan_by_key_views(BufferView<uint>   temp_storage,
                                    BufferView<KeyT>   keys,
                                    BufferView<ValueT> values,
                                    BufferView<ValueT> output,
                                    size_t             num_items) const
    {
        LUISA_ASSERT(m_created, "DeviceScan::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceScan.");
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(keys.size() >= num_items, "DeviceScan key view is too small.");
        LUISA_ASSERT(values.size() >= num_items, "DeviceScan value view is too small.");
        LUISA_ASSERT(output.size() >= num_items, "DeviceScan output view is too small.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(
                         scan_by_key_temp_storage_bytes<KeyT, ValueT>(num_items, m_block_size)),
                     "DeviceScan temporary storage is too small.");
    }

  public:
    DeviceScan()  = default;
    ~DeviceScan() = default;

#ifndef NDEBUG
    Stream* m_debug_stream = nullptr;  // bind debug stream for sync
#endif
    inline Stream* debug_stream() noexcept
    {
#ifndef NDEBUG
        return m_debug_stream;
#else
        return nullptr;
#endif
    }
    void create(Device& device, Stream* debug_stream = nullptr)
    {
        m_device = device;
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        int num_elements_per_block = m_block_size * ITEMS_PER_THREAD;
        int extra_space            = num_elements_per_block / m_warp_nums;
        m_shared_mem_size          = (num_elements_per_block + extra_space);
        m_created                  = true;
    }

    // ============================================================
    // GetTempStorageBytes: compute required temp buffer size (in bytes)
    // ============================================================

    /// Temp storage bytes for ExclusiveScan / InclusiveScan / ExclusiveSum / InclusiveSum
    template <typename T>
    static size_t GetTempStorageBytes(size_t num_items)
    {
        return scan_temp_storage_bytes<T>(num_items, BLOCK_SIZE);
    }

    /// Temp storage bytes for ExclusiveScanByKey / InclusiveScanByKey
    template <typename KeyType, typename ValueType>
    static size_t GetScanByKeyTempStorageBytes(size_t num_items)
    {
        return scan_by_key_temp_storage_bytes<KeyType, ValueType>(num_items, BLOCK_SIZE);
    }

    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT Type4Byte, typename ScanOp>
    void ExclusiveScan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        dispatch_scan<false, true>(cmdlist, temp_storage, d_in, d_out, num_items, scan_op, initial_value);
    }

    // CUB-compatible inclusive scan: no initial value is applied.
    template <NumericT Type4Byte, typename ScanOp>
    void InclusiveScan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op)
    {
        dispatch_scan<true, false>(cmdlist, temp_storage, d_in, d_out, num_items, scan_op, Type4Byte{});
    }

    // Backward-compatible extension: prepend an initial value to an inclusive scan.
    template <NumericT Type4Byte, typename ScanOp>
    void InclusiveScan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        dispatch_scan<true, true>(cmdlist, temp_storage, d_in, d_out, num_items, scan_op, initial_value);
    }

    template <NumericT Type4Byte, typename ScanOp>
    void InclusiveScanInit(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<Type4Byte> d_in,
                           BufferView<Type4Byte> d_out,
                           size_t                num_items,
                           ScanOp                scan_op,
                           Type4Byte             initial_value)
    {
        dispatch_scan<true, true>(cmdlist, temp_storage, d_in, d_out, num_items, scan_op, initial_value);
    }

    template <NumericT Type4Byte>
    void ExclusiveSum(CommandList&          cmdlist,
                      BufferView<uint>      temp_storage,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_items)
    {
        ExclusiveScan(
            cmdlist,
            temp_storage,
            d_in,
            d_out,
            num_items,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void InclusiveSum(CommandList&          cmdlist,
                      BufferView<uint>      temp_storage,
                      BufferView<Type4Byte> d_in,
                      BufferView<Type4Byte> d_out,
                      size_t                num_items)
    {
        InclusiveScan(cmdlist,
                      temp_storage,
                      d_in,
                      d_out,
                      num_items,
                      [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; });
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void ExclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        auto equality_op = [](const Var<KeyType>& lhs, const Var<KeyType>& rhs) noexcept
        { return lhs == rhs; };
        dispatch_scan_by_key<false, true>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, initial_value, equality_op);
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp, typename EqualityOp>
    void ExclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value,
                            EqualityOp            equality_op)
    {
        dispatch_scan_by_key<false, true>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, initial_value, equality_op);
    }

    // CUB-compatible inclusive scan-by-key: no initial value is applied.
    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void InclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items)
    {
        auto equality_op = [](const Var<KeyType>& lhs, const Var<KeyType>& rhs) noexcept
        { return lhs == rhs; };
        dispatch_scan_by_key<true, false>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, ValueType{}, equality_op);
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp, typename EqualityOp>
        requires requires(EqualityOp equality_op, const Var<KeyType>& lhs, const Var<KeyType>& rhs) {
            equality_op(lhs, rhs);
        }
    void InclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            EqualityOp            equality_op)
    {
        dispatch_scan_by_key<true, false>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, ValueType{}, equality_op);
    }

    // Backward-compatible extension: prepend an initial value to every segment.
    template <NumericT KeyType, NumericT ValueType, typename ScanOp>
    void InclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value)
    {
        auto equality_op = [](const Var<KeyType>& lhs, const Var<KeyType>& rhs) noexcept
        { return lhs == rhs; };
        dispatch_scan_by_key<true, true>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, initial_value, equality_op);
    }

    template <NumericT KeyType, NumericT ValueType, typename ScanOp, typename EqualityOp>
    void InclusiveScanByKey(CommandList&          cmdlist,
                            BufferView<uint>      temp_storage,
                            BufferView<KeyType>   d_keys_in,
                            BufferView<ValueType> d_values_in,
                            BufferView<ValueType> d_values_out,
                            ScanOp                scan_op,
                            size_t                num_items,
                            ValueType             initial_value,
                            EqualityOp            equality_op)
    {
        dispatch_scan_by_key<true, true>(
            cmdlist, temp_storage, d_keys_in, d_values_in, d_values_out, scan_op, num_items, initial_value, equality_op);
    }

    template <NumericT KeyType, NumericT ValueType>
    void ExclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<KeyType>   d_keys_in,
                           BufferView<ValueType> d_values_in,
                           BufferView<ValueType> d_values_out,
                           size_t                num_items)
    {
        ExclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<ValueType>& a, const Var<ValueType>& b) { return a + b; },
            num_items,
            ValueType(0));
    }

    template <NumericT KeyType, NumericT ValueType, typename EqualityOp>
    void ExclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<KeyType>   d_keys_in,
                           BufferView<ValueType> d_values_in,
                           BufferView<ValueType> d_values_out,
                           size_t                num_items,
                           EqualityOp            equality_op)
    {
        ExclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<ValueType>& a, const Var<ValueType>& b) { return a + b; },
            num_items,
            ValueType(0),
            equality_op);
    }

    template <NumericT KeyType, NumericT ValueType>
    void InclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<KeyType>   d_keys_in,
                           BufferView<ValueType> d_values_in,
                           BufferView<ValueType> d_values_out,
                           size_t                num_items)
    {
        InclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<ValueType>& a, const Var<ValueType>& b) { return a + b; },
            num_items);
    }

    template <NumericT KeyType, NumericT ValueType, typename EqualityOp>
    void InclusiveSumByKey(CommandList&          cmdlist,
                           BufferView<uint>      temp_storage,
                           BufferView<KeyType>   d_keys_in,
                           BufferView<ValueType> d_values_in,
                           BufferView<ValueType> d_values_out,
                           size_t                num_items,
                           EqualityOp            equality_op)
    {
        InclusiveScanByKey(
            cmdlist,
            temp_storage,
            d_keys_in,
            d_values_in,
            d_values_out,
            [](const Var<ValueType>& a, const Var<ValueType>& b) { return a + b; },
            num_items,
            equality_op);
    }


  private:
    template <bool is_inclusive, bool use_initial_value, NumericT Type4Byte, typename ScanOp>
    void dispatch_scan(CommandList&          cmdlist,
                       BufferView<uint>      temp_storage,
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       size_t                num_items,
                       ScanOp                scan_op,
                       Type4Byte             initial_value)
    {
        validate_scan_views(temp_storage, d_in, d_out, num_items);
        if(num_items == 0u)
        {
            return;
        }
        auto views = scan_temp_views<Type4Byte>(temp_storage, num_items, m_block_size);
        lcpp_check((scan_array<is_inclusive, use_initial_value, Type4Byte>(
                       cmdlist, views.tile_status, views.tile_partial, views.tile_inclusive, d_in, d_out, num_items, scan_op, initial_value)),
                   cmdlist,
                   debug_stream());
    }

    template <bool is_inclusive, bool use_initial_value, NumericT KeyType, NumericT ValueType, typename ScanOp, typename EqualityOp>
    void dispatch_scan_by_key(CommandList&          cmdlist,
                              BufferView<uint>      temp_storage,
                              BufferView<KeyType>   d_keys_in,
                              BufferView<ValueType> d_values_in,
                              BufferView<ValueType> d_values_out,
                              ScanOp                scan_op,
                              size_t                num_items,
                              ValueType             initial_value,
                              EqualityOp            equality_op)
    {
        validate_scan_by_key_views(temp_storage, d_keys_in, d_values_in, d_values_out, num_items);
        if(num_items == 0u)
        {
            return;
        }
        auto views = scan_by_key_temp_views<KeyType, ValueType>(temp_storage, num_items, m_block_size);
        lcpp_check((scan_by_key_array<is_inclusive, use_initial_value, KeyType, ValueType>(cmdlist,
                                                                                           views.tile_status,
                                                                                           views.tile_partial,
                                                                                           views.tile_inclusive,
                                                                                           d_keys_in,
                                                                                           views.previous_keys,
                                                                                           d_values_in,
                                                                                           d_values_out,
                                                                                           num_items,
                                                                                           scan_op,
                                                                                           initial_value,
                                                                                           equality_op)),
                   cmdlist,
                   debug_stream());
    }

    template <bool is_inclusive, bool use_initial_value, NumericT Type4Byte, typename ScanOp>
    [[nodiscard]] int scan_array(CommandList&          cmdlist,
                                 BufferView<uint>      tile_states,
                                 BufferView<Type4Byte> tile_partial,
                                 BufferView<Type4Byte> tile_inclusive,
                                 BufferView<Type4Byte> d_in,
                                 BufferView<Type4Byte> d_out,
                                 size_t                num_items,
                                 ScanOp                scan_op,
                                 Type4Byte             initial_value)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));

        using ScanShader = details::ScanModule<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanTileStateInitKernel = ScanShader::ScanTileStateInitKernel;
        using ScanShaderKernel        = ScanShader::ScanKernel;


        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        auto   init_key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto   ms_tile_state_init_it = ms_scan_key.find(init_key);
        if(ms_tile_state_init_it == ms_scan_key.end())
        {
            auto shader = ScanShader().compile_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_scan_key.try_emplace(init_key, std::move(shader));
            ms_tile_state_init_it = ms_scan_key.find(init_key);
        }
        if(ms_tile_state_init_it == ms_scan_key.end())
        {
            return -1;
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ScanTileStateInitKernel*>(&(*ms_tile_state_init_it->second));
        if(!ms_scan_tile_state_init_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, tile_partial, tile_inclusive, uint(num_tiles))
                       .dispatch(m_block_size * init_num_blocks);

        // scan
        auto key = luisa::format("{}+initial:{}", get_type_and_op_desc<Type4Byte>(scan_op), use_initial_value);
        auto& scan_map   = is_inclusive ? ms_inclusive_scan_map : ms_exclusive_scan_map;
        auto  ms_scan_it = scan_map.find(key);
        if(ms_scan_it == scan_map.end())
        {
            auto shader = ScanShader().template compile<is_inclusive, use_initial_value>(
                m_device, m_shared_mem_size, scan_op);
            if(!shader)
            {
                return -1;
            }
            scan_map.try_emplace(key, std::move(shader));
            ms_scan_it = scan_map.find(key);
        }
        if(ms_scan_it == scan_map.end())
        {
            return -1;
        }
        auto ms_scan_ptr = reinterpret_cast<ScanShaderKernel*>(&(*ms_scan_it->second));
        if(!ms_scan_ptr)
        {
            return -1;
        }
        cmdlist
            << (*ms_scan_ptr)(tile_states, tile_partial, tile_inclusive, d_in, d_out, initial_value, static_cast<uint>(num_items))
                   .dispatch(m_block_size * num_tiles);
        return 0;
    }


    template <bool is_inclusive, bool use_initial_value, NumericT KeyValue, NumericT ValueType, typename ScanOp, typename EqualityOp, typename FlagValueT = KeyValuePair<int, ValueType>>
    [[nodiscard]] int scan_by_key_array(CommandList&           cmdlist,
                                        BufferView<uint>       tile_states,
                                        BufferView<FlagValueT> tile_partial,
                                        BufferView<FlagValueT> tile_inclusive,
                                        BufferView<KeyValue>   d_keys_in,
                                        BufferView<KeyValue>   d_prev_keys_in,
                                        BufferView<ValueType>  d_values_in,
                                        BufferView<ValueType>  d_values_out,
                                        size_t                 num_items,
                                        ScanOp                 scan_op,
                                        ValueType              initial_value,
                                        EqualityOp             equality_op)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));

        using ScanByKeyShader = details::ScanByKeyModule<KeyValue, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ScanByKeyTileStateInitKernel = ScanByKeyShader::ScanTileStateInitKernel;
        using ScanByKeyShaderKernel        = ScanByKeyShader::ScanByKeyKernel;

        size_t init_num_blocks                 = ceil_div(num_tiles, m_block_size);
        auto   init_key                        = get_type_and_op_desc<KeyValue, ValueType>();
        auto ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        if(ms_scan_by_key_tile_state_init_it == ms_scan_by_key_tile_state_init_map.end())
        {
            auto shader = ScanByKeyShader().compile_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_scan_by_key_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_by_key_tile_state_init_it = ms_scan_by_key_tile_state_init_map.find(init_key);
        }
        if(ms_scan_by_key_tile_state_init_it == ms_scan_by_key_tile_state_init_map.end())
        {
            return -1;
        }
        auto ms_scan_by_key_tile_state_init_ptr =
            reinterpret_cast<ScanByKeyTileStateInitKernel*>(&(*ms_scan_by_key_tile_state_init_it->second));
        if(!ms_scan_by_key_tile_state_init_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_scan_by_key_tile_state_init_ptr)(
                       tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, uint(num_tiles))
                       .dispatch(m_block_size * init_num_blocks);

        // scan
        auto key = luisa::format("{}+{}+initial:{}",
                                 get_type_and_op_desc<KeyValue, ValueType>(scan_op),
                                 get_binary_op_hash<KeyValue>(equality_op),
                                 use_initial_value);
        auto& scan_by_key_map = is_inclusive ? ms_inclusive_scan_by_key_map : ms_exclusive_scan_by_key_map;
        auto ms_scan_by_it = scan_by_key_map.find(key);
        if(ms_scan_by_it == scan_by_key_map.end())
        {
            LUISA_INFO("Compiling Scan By Key shader for key: {}", key);
            auto shader = ScanByKeyShader().template compile<is_inclusive, use_initial_value>(
                m_device, m_shared_mem_size, scan_op, equality_op);
            if(!shader)
            {
                return -1;
            }
            scan_by_key_map.try_emplace(key, std::move(shader));
            ms_scan_by_it = scan_by_key_map.find(key);
        }
        if(ms_scan_by_it == scan_by_key_map.end())
        {
            return -1;
        }
        auto ms_scan_by_key_ptr = reinterpret_cast<ScanByKeyShaderKernel*>(&(*ms_scan_by_it->second));
        if(!ms_scan_by_key_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_scan_by_key_ptr)(tile_states,
                                         tile_partial,
                                         tile_inclusive,
                                         d_keys_in,
                                         d_prev_keys_in,
                                         d_values_in,
                                         d_values_out,
                                         initial_value,
                                         static_cast<uint>(num_items))
                       .dispatch(m_block_size * num_tiles);
        return 0;
    }

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_key;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_exclusive_scan_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_inclusive_scan_map;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_by_key_tile_state_init_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_exclusive_scan_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_inclusive_scan_by_key_map;
};
}  // namespace luisa::parallel_primitive
