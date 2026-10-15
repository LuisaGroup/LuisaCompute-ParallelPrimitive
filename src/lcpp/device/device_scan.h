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
#include <lcpp/runtime/interop_dispatch.h>
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
// Tuned policy for 4-byte element types (epoch 2). The scan kernel uses a
// coalesced block-striped load/store transposed through shared memory (see
// device/details/scan.h). 256 threads x 15 items/tile balances DRAM
// coalescing against decoupled-look-back depth and stays within the DX 32 KiB
// shared-memory floor: (256*15 + 256*15/32)*4 = 15840 B per block.
// The load/store algorithm, the warp-local transpose sync and the packed
// tile-state knobs below are the CUB-parity opt-ins; DeviceScanCudaParity at
// the end of this header is the measured CUDA-parity configuration.
template <size_t BLOCK_SIZE                         = details::BLOCK_SIZE,
          size_t WARP_NUMS                          = details::WARP_SIZE,
          size_t ITEMS_PER_THREAD                   = 15,
          BlockLoadAlgorithm LOAD_ALGORITHM         = BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
          BlockStoreAlgorithm STORE_ALGORITHM       = BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
          // CUDA-only opt-in for warp-local transpose synchronization.
          bool WARP_SYNC = false,
          // CUDA-only opt-in for CUB-style packed status/value descriptors.
          bool PACKED_TILE_STATE = false>
class DeviceScan : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;

    uint   m_shared_mem_size = 0;
    Device m_device;
    // Optional vk-cuda interop sink (epoch B).  Null => pure-vulkan DSL path.
    InteropContext* m_interop = nullptr;
    bool   m_created = false;

    template <typename T>
    struct ScanTempViews
    {
        BufferView<uint> tile_status;
        BufferView<T>    tile_partial;
        BufferView<T>    tile_inclusive;
    };

    struct PackedScanTempViews
    {
        BufferView<ulong> tile_state;
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
        if constexpr(PACKED_TILE_STATE && sizeof(T) == sizeof(uint))
        {
            size_t tile_count = details::WARP_SIZE + num_tiles_for(num_items, block_size);
            return align_up_temp_storage(tile_count * sizeof(ulong));
        }
        size_t tile_count = details::WARP_SIZE + num_tiles_for(num_items, block_size);
        size_t bytes      = tile_count * sizeof(uint);
        bytes             = align_up_temp_storage(bytes);
        bytes += bytes_to_uint_count(tile_count * sizeof(T)) * sizeof(uint);
        bytes = align_up_temp_storage(bytes);
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
        bytes                = align_up_temp_storage(bytes);
        bytes += bytes_to_uint_count(tile_count * sizeof(FlagValuePairT)) * sizeof(uint);
        bytes = align_up_temp_storage(bytes);
        bytes += bytes_to_uint_count(tile_count * sizeof(FlagValuePairT)) * sizeof(uint);
        bytes = align_up_temp_storage(bytes);
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
        offset_bytes = align_up_temp_storage(offset_bytes);

        size_t partial_uint_count = bytes_to_uint_count(tile_count * sizeof(T));
        auto   tile_partial =
            temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<T>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        size_t inclusive_uint_count = bytes_to_uint_count(tile_count * sizeof(T));
        auto   tile_inclusive =
            temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<T>();
        return {tile_status, tile_partial, tile_inclusive};
    }

    template <typename T>
    [[nodiscard]] bool use_packed_tile_state() const noexcept
    {
        return sizeof(T) == sizeof(uint) && (PACKED_TILE_STATE || m_device.backend_name() == "cuda");
    }

    [[nodiscard]] static PackedScanTempViews packed_scan_temp_views(BufferView<uint> temp_storage,
                                                                      size_t         num_items,
                                                                      size_t         block_size)
    {
        size_t tile_count = details::WARP_SIZE + num_tiles_for(num_items, block_size);
        return {temp_storage.subview(0u, bytes_to_uint_count(tile_count * sizeof(ulong))).template as<ulong>()};
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
        offset_bytes = align_up_temp_storage(offset_bytes);

        size_t partial_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_partial =
            temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        size_t inclusive_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_inclusive =
            temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();
        offset_bytes += inclusive_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

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
    // `interop` is optional: pass a valid InteropContext (see
    // lcpp/runtime/interop_dispatch.h) to launch the scan / scan-by-key
    // kernels as cuda-backend modules inside the vk command buffer.  Without
    // it (or when the context is null because the fast path is unavailable)
    // every op keeps its pure-vulkan DSL dispatch path.
    void create(Device& device, Stream* debug_stream = nullptr, InteropContext* interop = nullptr)
    {
        m_device                   = device;
        // Epoch C per-op policy: `LCPP_INTEROP` (1|all|auto|a,b,c) decides
        // per module whether this one takes the fast path; see
        // lcpp/runtime/interop_dispatch.h.
        m_interop = interop_for(interop, "scan");
#ifndef NDEBUG
        m_debug_stream = debug_stream;
#endif
        // BlockLoad/BlockStore transpose paths use a dense tile. The old
        // shared-memory padding belonged to the removed tree-reduction path
        // and only inflated the scan kernel's static shared footprint.
        m_shared_mem_size = m_block_size * ITEMS_PER_THREAD;
        m_created                  = true;
    }

    /// True when this module dispatches its scan kernels through the vk-cuda
    /// interop fast path (cuda-compiled module, vk command buffer).
    [[nodiscard]] bool interop_active() const noexcept { return m_interop != nullptr; }

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

    template <NumericT InputT, typename TransformOp>
    void ExclusiveSumTransform(CommandList&          cmdlist,
                               BufferView<uint>      temp_storage,
                               BufferView<InputT>    d_in,
                               BufferView<uint>      d_out,
                               size_t                num_items,
                               TransformOp           transform)
    {
        dispatch_scan_transform(cmdlist, temp_storage, d_in, d_out, num_items, transform);
    }

    template <NumericT InputT, typename EqualityOp>
    void ExclusiveSumAdjacent(CommandList&          cmdlist,
                              BufferView<uint>      temp_storage,
                              BufferView<InputT>    d_in,
                              BufferView<uint>      d_out,
                              size_t                num_items,
                              EqualityOp            equality)
    {
        dispatch_scan_adjacent(cmdlist, temp_storage, d_in, d_out, num_items, equality);
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
    template <NumericT InputT, typename TransformOp>
    void dispatch_scan_transform(CommandList&       cmdlist,
                                 BufferView<uint>   temp_storage,
                                 BufferView<InputT> d_in,
                                 BufferView<uint>   d_out,
                                 size_t             num_items,
                                 TransformOp        transform)
    {
        static_assert(sizeof(InputT) == sizeof(uint), "ExclusiveSumTransform requires 4-byte input values.");
        LUISA_ASSERT(m_created, "DeviceScan::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceScan.");
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(d_in.size() >= num_items, "DeviceScan transform input view is too small.");
        LUISA_ASSERT(d_out.size() >= num_items, "DeviceScan transform output view is too small.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(scan_temp_storage_bytes<uint>(num_items, m_block_size)),
                     "DeviceScan transform temporary storage is too small.");
        if(use_packed_tile_state<uint>())
        {
            auto views = packed_scan_temp_views(temp_storage, num_items, m_block_size);
            lcpp_check(scan_array_transform_packed(cmdlist, views.tile_state, d_in, d_out, num_items, transform),
                       cmdlist,
                       debug_stream());
        }
        else
        {
            auto views = scan_temp_views<uint>(temp_storage, num_items, m_block_size);
            lcpp_check(scan_array_transform(cmdlist,
                                            views.tile_status,
                                            views.tile_partial,
                                            views.tile_inclusive,
                                            d_in,
                                            d_out,
                                            num_items,
                                            transform),
                       cmdlist,
                       debug_stream());
        }
    }

    template <NumericT InputT, typename EqualityOp>
    [[nodiscard]] int scan_array_adjacent(CommandList&       cmdlist,
                                          BufferView<uint>   tile_states,
                                          BufferView<uint>   tile_partial,
                                          BufferView<uint>   tile_inclusive,
                                          BufferView<InputT> d_in,
                                          BufferView<uint>   d_out,
                                          size_t             num_items,
                                          EqualityOp          equality)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));
        using ScanShader = details::ScanModule<uint,
                                               BLOCK_SIZE,
                                               ITEMS_PER_THREAD,
                                               LOAD_ALGORITHM,
                                               STORE_ALGORITHM,
                                               WARP_SYNC>;
        auto init_key = detail::make_shader_cache_key(detail::type_ptr<uint>());
        auto init_it  = ms_scan_key.find(init_key);
        if(init_it == ms_scan_key.end())
        {
            auto shader = ScanShader().compile_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_scan_key.try_emplace(init_key, std::move(shader));
            init_it = ms_scan_key.find(init_key);
        }
        auto init_shader = reinterpret_cast<typename ScanShader::ScanTileStateInitKernel*>(&(*init_it->second));
        if(!init_shader)
        {
            return -1;
        }
        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        cmdlist << (*init_shader)(tile_states, tile_partial, tile_inclusive, num_tiles)
                       .dispatch(m_block_size * init_num_blocks);

        auto key = detail::make_shader_cache_key(detail::type_ptr<InputT>(),
                                         get_binary_op_hash<InputT>(equality));
        auto it = ms_exclusive_scan_transform_map.find(key);
        if(it == ms_exclusive_scan_transform_map.end())
        {
            auto shader = ScanShader().template compile_adjacent_transform<InputT>(m_device, m_shared_mem_size, equality);
            if(!shader)
            {
                return -1;
            }
            ms_exclusive_scan_transform_map.try_emplace(key, std::move(shader));
            it = ms_exclusive_scan_transform_map.find(key);
        }
        auto scan_shader = reinterpret_cast<typename ScanShader::template TransformScanKernel<InputT>*>(&(*it->second));
        if(!scan_shader)
        {
            return -1;
        }
        cmdlist << (*scan_shader)(tile_states,
                                  tile_partial,
                                  tile_inclusive,
                                  d_in,
                                  d_out,
                                  uint{0},
                                  static_cast<uint>(num_items))
                       .dispatch(m_block_size * num_tiles);
        return 0;
    }

    template <NumericT InputT, typename EqualityOp>
    [[nodiscard]] int scan_array_adjacent_packed(CommandList&       cmdlist,
                                                 BufferView<ulong>  tile_state,
                                                 BufferView<InputT> d_in,
                                                 BufferView<uint>   d_out,
                                                 size_t             num_items,
                                                 EqualityOp          equality)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));
        using ScanShader = details::ScanModule<uint,
                                               BLOCK_SIZE,
                                               ITEMS_PER_THREAD,
                                               LOAD_ALGORITHM,
                                               STORE_ALGORITHM,
                                               WARP_SYNC>;
        auto init_key = detail::make_shader_cache_key(detail::type_ptr<uint>());
        auto init_it  = ms_packed_scan_key.find(init_key);
        if(init_it == ms_packed_scan_key.end())
        {
            auto shader = ScanShader().compile_packed_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_packed_scan_key.try_emplace(init_key, std::move(shader));
            init_it = ms_packed_scan_key.find(init_key);
        }
        auto init_shader = reinterpret_cast<typename ScanShader::PackedScanTileStateInitKernel*>(&(*init_it->second));
        if(!init_shader)
        {
            return -1;
        }
        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        cmdlist << (*init_shader)(tile_state, num_tiles).dispatch(m_block_size * init_num_blocks);

        auto key = detail::make_shader_cache_key(detail::type_ptr<InputT>(),
                                         get_binary_op_hash<InputT>(equality));
        auto it = ms_packed_exclusive_scan_transform_map.find(key);
        if(it == ms_packed_exclusive_scan_transform_map.end())
        {
            auto shader = ScanShader().template compile_packed_adjacent_transform<InputT>(m_device,
                                                                                           m_shared_mem_size,
                                                                                           equality);
            if(!shader)
            {
                return -1;
            }
            ms_packed_exclusive_scan_transform_map.try_emplace(key, std::move(shader));
            it = ms_packed_exclusive_scan_transform_map.find(key);
        }
        auto scan_shader = reinterpret_cast<typename ScanShader::template PackedTransformScanKernel<InputT>*>(&(*it->second));
        if(!scan_shader)
        {
            return -1;
        }
        cmdlist << (*scan_shader)(tile_state, d_in, d_out, uint{0}, static_cast<uint>(num_items))
                       .dispatch(m_block_size * num_tiles);
        return 0;
    }

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
        if constexpr(sizeof(Type4Byte) == sizeof(uint))
        {
            if(use_packed_tile_state<Type4Byte>())
            {
                auto views = packed_scan_temp_views(temp_storage, num_items, m_block_size);
                lcpp_check((scan_array_packed<is_inclusive, use_initial_value, Type4Byte>(
                               cmdlist, views.tile_state, d_in, d_out, num_items, scan_op, initial_value)),
                           cmdlist,
                           debug_stream());
            }
            else
            {
                auto views = scan_temp_views<Type4Byte>(temp_storage, num_items, m_block_size);
                lcpp_check((scan_array<is_inclusive, use_initial_value, Type4Byte>(cmdlist,
                                                                                    views.tile_status,
                                                                                    views.tile_partial,
                                                                                    views.tile_inclusive,
                                                                                    d_in,
                                                                                    d_out,
                                                                                    num_items,
                                                                                    scan_op,
                                                                                    initial_value)),
                           cmdlist,
                           debug_stream());
            }
        }
        else
        {
            auto views = scan_temp_views<Type4Byte>(temp_storage, num_items, m_block_size);
            lcpp_check((scan_array<is_inclusive, use_initial_value, Type4Byte>(cmdlist,
                                                                                views.tile_status,
                                                                                views.tile_partial,
                                                                                views.tile_inclusive,
                                                                                d_in,
                                                                                d_out,
                                                                                num_items,
                                                                                scan_op,
                                                                                initial_value)),
                       cmdlist,
                       debug_stream());
        }
    }

    template <NumericT InputT, typename EqualityOp>
    void dispatch_scan_adjacent(CommandList&       cmdlist,
                                BufferView<uint>   temp_storage,
                                BufferView<InputT> d_in,
                                BufferView<uint>   d_out,
                                size_t             num_items,
                                EqualityOp          equality)
    {
        static_assert(sizeof(InputT) == sizeof(uint), "ExclusiveSumAdjacent requires 4-byte input values.");
        LUISA_ASSERT(m_created, "DeviceScan::create must be called before dispatch.");
        LUISA_ASSERT(num_items <= std::numeric_limits<uint>::max(), "Too many items for DeviceScan.");
        if(num_items == 0u)
        {
            return;
        }
        LUISA_ASSERT(d_in.size() >= num_items, "DeviceScan adjacent input view is too small.");
        LUISA_ASSERT(d_out.size() >= num_items, "DeviceScan adjacent output view is too small.");
        LUISA_ASSERT(temp_storage.size() >= bytes_to_uint_count(scan_temp_storage_bytes<uint>(num_items, m_block_size)),
                     "DeviceScan adjacent temporary storage is too small.");
        if(use_packed_tile_state<uint>())
        {
            auto views = packed_scan_temp_views(temp_storage, num_items, m_block_size);
            lcpp_check(scan_array_adjacent_packed(cmdlist, views.tile_state, d_in, d_out, num_items, equality),
                       cmdlist,
                       debug_stream());
        }
        else
        {
            auto views = scan_temp_views<uint>(temp_storage, num_items, m_block_size);
            lcpp_check(scan_array_adjacent(cmdlist,
                                           views.tile_status,
                                           views.tile_partial,
                                           views.tile_inclusive,
                                           d_in,
                                           d_out,
                                           num_items,
                                           equality),
                       cmdlist,
                       debug_stream());
        }
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

    template <NumericT InputT, typename TransformOp>
    [[nodiscard]] int scan_array_transform(CommandList&       cmdlist,
                                           BufferView<uint>   tile_states,
                                           BufferView<uint>   tile_partial,
                                           BufferView<uint>   tile_inclusive,
                                           BufferView<InputT> d_in,
                                           BufferView<uint>   d_out,
                                           size_t             num_items,
                                           TransformOp        transform)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));
        using ScanShader = details::ScanModule<uint,
                                               BLOCK_SIZE,
                                               ITEMS_PER_THREAD,
                                               LOAD_ALGORITHM,
                                               STORE_ALGORITHM,
                                               WARP_SYNC>;
        auto init_key = detail::make_shader_cache_key(detail::type_ptr<uint>());
        auto init_it  = ms_scan_key.find(init_key);
        if(init_it == ms_scan_key.end())
        {
            auto shader = ScanShader().compile_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_scan_key.try_emplace(init_key, std::move(shader));
            init_it = ms_scan_key.find(init_key);
        }
        auto init_shader = reinterpret_cast<typename ScanShader::ScanTileStateInitKernel*>(&(*init_it->second));
        if(!init_shader)
        {
            return -1;
        }
        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        cmdlist << (*init_shader)(tile_states, tile_partial, tile_inclusive, num_tiles)
                       .dispatch(m_block_size * init_num_blocks);

        auto key = detail::make_shader_cache_key(detail::type_ptr<InputT>(),
                                         get_unary_op_hash<InputT>(transform));
        auto it = ms_exclusive_scan_transform_map.find(key);
        if(it == ms_exclusive_scan_transform_map.end())
        {
            auto shader = ScanShader().template compile_transform<InputT>(m_device, m_shared_mem_size, transform);
            if(!shader)
            {
                return -1;
            }
            ms_exclusive_scan_transform_map.try_emplace(key, std::move(shader));
            it = ms_exclusive_scan_transform_map.find(key);
        }
        auto scan_shader = reinterpret_cast<typename ScanShader::template TransformScanKernel<InputT>*>(&(*it->second));
        if(!scan_shader)
        {
            return -1;
        }
        cmdlist << (*scan_shader)(tile_states,
                                  tile_partial,
                                  tile_inclusive,
                                  d_in,
                                  d_out,
                                  uint{0},
                                  static_cast<uint>(num_items))
                       .dispatch(m_block_size * num_tiles);
        return 0;
    }

    template <NumericT InputT, typename TransformOp>
    [[nodiscard]] int scan_array_transform_packed(CommandList&       cmdlist,
                                                  BufferView<ulong>  tile_state,
                                                  BufferView<InputT> d_in,
                                                  BufferView<uint>   d_out,
                                                  size_t             num_items,
                                                  TransformOp        transform)
    {
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));
        using ScanShader = details::ScanModule<uint,
                                               BLOCK_SIZE,
                                               ITEMS_PER_THREAD,
                                               LOAD_ALGORITHM,
                                               STORE_ALGORITHM,
                                               WARP_SYNC>;
        auto init_key = detail::make_shader_cache_key(detail::type_ptr<uint>());
        auto init_it  = ms_packed_scan_key.find(init_key);
        if(init_it == ms_packed_scan_key.end())
        {
            auto shader = ScanShader().compile_packed_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_packed_scan_key.try_emplace(init_key, std::move(shader));
            init_it = ms_packed_scan_key.find(init_key);
        }
        auto init_shader = reinterpret_cast<typename ScanShader::PackedScanTileStateInitKernel*>(&(*init_it->second));
        if(!init_shader)
        {
            return -1;
        }
        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        cmdlist << (*init_shader)(tile_state, num_tiles).dispatch(m_block_size * init_num_blocks);

        auto key = detail::make_shader_cache_key(detail::type_ptr<InputT>(),
                                         get_unary_op_hash<InputT>(transform));
        auto it = ms_packed_exclusive_scan_transform_map.find(key);
        if(it == ms_packed_exclusive_scan_transform_map.end())
        {
            auto shader = ScanShader().template compile_packed_transform<InputT>(m_device, m_shared_mem_size, transform);
            if(!shader)
            {
                return -1;
            }
            ms_packed_exclusive_scan_transform_map.try_emplace(key, std::move(shader));
            it = ms_packed_exclusive_scan_transform_map.find(key);
        }
        auto scan_shader = reinterpret_cast<typename ScanShader::template PackedTransformScanKernel<InputT>*>(&(*it->second));
        if(!scan_shader)
        {
            return -1;
        }
        cmdlist << (*scan_shader)(tile_state, d_in, d_out, uint{0}, static_cast<uint>(num_items))
                       .dispatch(m_block_size * num_tiles);
        return 0;
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

        using ScanShader =
            details::ScanModule<Type4Byte,
                                BLOCK_SIZE,
                                ITEMS_PER_THREAD,
                                LOAD_ALGORITHM,
                                STORE_ALGORITHM,
                                WARP_SYNC>;
        using ScanTileStateInitKernel = ScanShader::ScanTileStateInitKernel;
        using ScanShaderKernel        = ScanShader::ScanKernel;


        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        auto   init_key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>());
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
        if(!ms_scan_tile_state_init_ptr) { return -1; }
        uint init_threads = m_block_size * static_cast<uint>(init_num_blocks);
        auto init_launched = false;
        // ---- interop fast path: tile-state init ----
        // Usage table: `InitializeWardStatus` only writes `tile_status`
        // (head padding + per-tile INVALID); tile_partial / tile_inclusive are
        // never touched by the init kernel.  They are declared READ_WRITE
        // conservatively -- the following scan kernel is ordered behind this
        // dispatch either way (tile_status RAW), so no extra barrier is paid.
        if(m_interop != nullptr)
        {
            auto it = ms_scan_key_interop.find(init_key);
            if(it == ms_scan_key_interop.end())
            {
                it = import_into_interop_cache(
                    m_interop,
                    ms_scan_key_interop,
                    init_key,
                    ScanShader().compile_scan_tile_state_init(m_interop->cuda_device()));
            }
            if(it != ms_scan_key_interop.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    uint>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, uint(num_tiles))
                               .dispatch(init_threads);
                init_launched = true;
            }
        }
        if(!init_launched)
        {
            cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, tile_partial, tile_inclusive, uint(num_tiles))
                           .dispatch(init_threads);
        }

        // scan
        // use_initial_value selects a different compiled shader.
        auto key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>(),
                                                 get_binary_op_hash<Type4Byte>(scan_op),
                                                 detail::bool_ptr(use_initial_value));
        auto &scan_map = is_inclusive ? ms_inclusive_scan_map : ms_exclusive_scan_map;
        auto ms_scan_it = scan_map.find(key);
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
        if(!ms_scan_ptr) { return -1; }
        uint scan_threads = m_block_size * num_tiles;
        auto scan_launched = false;
        // ---- interop fast path: decoupled-look-back scan ----
        // Usage table: the tile-state triple is the decoupled-look-back channel
        // (volatile_write of partial/inclusive values + status words, then
        // volatile_read/WaitForValid spin-polling by later tiles) so it is
        // READ_WRITE; d_in is read once per element and never written; d_out is
        // written and never read.
        if(m_interop != nullptr)
        {
            auto &scan_interop_map = is_inclusive ? ms_inclusive_scan_interop_map : ms_exclusive_scan_interop_map;
            auto  it               = scan_interop_map.find(key);
            if(it == scan_interop_map.end())
            {
                auto compiled = ScanShader().template compile<is_inclusive, use_initial_value>(
                    m_interop->cuda_device(), m_shared_mem_size, scan_op);
                it = import_into_interop_cache(m_interop, scan_interop_map, key, compiled);
            }
            if(it != scan_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<Type4Byte>, Usage::WRITE>,
                    Type4Byte,
                    uint>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, d_in, d_out, initial_value, uint(num_items))
                               .dispatch(scan_threads);
                scan_launched = true;
            }
        }
        if(!scan_launched)
        {
            cmdlist << (*ms_scan_ptr)(tile_states, tile_partial, tile_inclusive, d_in, d_out, initial_value, num_items)
                           .dispatch(scan_threads);
        }

        return 0;
    }

    template <bool is_inclusive, bool use_initial_value, NumericT Type4Byte, typename ScanOp>
    [[nodiscard]] int scan_array_packed(CommandList&          cmdlist,
                                        BufferView<ulong>     tile_state,
                                        BufferView<Type4Byte> d_in,
                                        BufferView<Type4Byte> d_out,
                                        size_t                num_items,
                                        ScanOp                scan_op,
                                        Type4Byte             initial_value)
    {
        static_assert(sizeof(Type4Byte) == sizeof(uint), "Packed scan tile state requires a 4-byte value.");
        uint num_tiles = static_cast<uint>(num_tiles_for(num_items, m_block_size));

        using ScanShader                 = details::ScanModule<Type4Byte,
                                               BLOCK_SIZE,
                                               ITEMS_PER_THREAD,
                                               LOAD_ALGORITHM,
                                               STORE_ALGORITHM,
                                               WARP_SYNC>;
        using ScanTileStateInitKernel   = ScanShader::PackedScanTileStateInitKernel;
        using ScanShaderKernel           = ScanShader::PackedScanKernel;

        size_t init_num_blocks = ceil_div(num_tiles, m_block_size);
        auto init_key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>());
        auto   ms_tile_state_init_it = ms_packed_scan_key.find(init_key);
        if(ms_tile_state_init_it == ms_packed_scan_key.end())
        {
            auto shader = ScanShader().compile_packed_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_packed_scan_key.try_emplace(init_key, std::move(shader));
            ms_tile_state_init_it = ms_packed_scan_key.find(init_key);
        }
        if(ms_tile_state_init_it == ms_packed_scan_key.end())
        {
            return -1;
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ScanTileStateInitKernel*>(&(*ms_tile_state_init_it->second));
        if(!ms_scan_tile_state_init_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_state, uint(num_tiles)).dispatch(m_block_size * init_num_blocks);

        auto key = detail::make_shader_cache_key(detail::type_ptr<Type4Byte>(),
                                         get_binary_op_hash<Type4Byte>(scan_op),
                                         detail::bool_ptr(use_initial_value));
        auto& scan_map = is_inclusive ? ms_packed_inclusive_scan_map : ms_packed_exclusive_scan_map;
        auto  ms_scan_it = scan_map.find(key);
        if(ms_scan_it == scan_map.end())
        {
            auto shader = ScanShader().template compile_packed<is_inclusive, use_initial_value>(
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
        cmdlist << (*ms_scan_ptr)(tile_state, d_in, d_out, initial_value, static_cast<uint>(num_items))
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
        if(!ms_scan_by_key_tile_state_init_ptr) { return -1; }
        uint init_threads = m_block_size * static_cast<uint>(init_num_blocks);
        auto init_launched = false;
        // ---- interop fast path: tile-state init (+ d_prev_keys_in) ----
        // Usage table: `InitializeWardStatus` writes tile_status, and the same
        // kernel writes the one key preceding each tile into d_prev_keys_in;
        // d_keys_in is read for it.  tile_partial / tile_inclusive are declared
        // READ_WRITE conservatively (see the plain scan path above).
        if(m_interop != nullptr)
        {
            auto it = ms_scan_by_key_tile_state_init_interop_map.find(init_key);
            if(it == ms_scan_by_key_tile_state_init_interop_map.end())
            {
                it = import_into_interop_cache(
                    m_interop,
                    ms_scan_by_key_tile_state_init_interop_map,
                    init_key,
                    ScanByKeyShader().compile_scan_tile_state_init(m_interop->cuda_device()));
            }
            if(it != ms_scan_by_key_tile_state_init_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::WRITE>,
                    int>();
                cmdlist << kernel(tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, int(num_tiles))
                               .dispatch(init_threads);
                init_launched = true;
            }
        }
        if(!init_launched)
        {
            cmdlist << (*ms_scan_by_key_tile_state_init_ptr)(
                           tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, uint(num_tiles))
                           .dispatch(init_threads);
        }

        // scan
        // The equality operator and use_initial_value also select the shader.
        auto key = detail::make_shader_cache_key(detail::type_ptr<KeyValue>(),
                                                 detail::type_ptr<ValueType>(),
                                                 get_binary_op_hash<ValueType>(scan_op),
                                                 get_binary_op_hash<KeyValue>(equality_op),
                                                 detail::bool_ptr(use_initial_value));
        auto& scan_by_key_map = is_inclusive ? ms_inclusive_scan_by_key_map : ms_exclusive_scan_by_key_map;
        auto ms_scan_by_it = scan_by_key_map.find(key);
        if(ms_scan_by_it == scan_by_key_map.end())
        {
            LUISA_INFO("Compiling Scan By Key shader (cache miss)");
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
        if(!ms_scan_by_key_ptr) { return -1; }
        uint scan_threads  = m_block_size * num_tiles;
        auto scan_launched = false;
        // ---- interop fast path: decoupled-look-back scan-by-key ----
        // Usage table: same tile-state channel as the plain scan (READ_WRITE),
        // d_keys_in / d_prev_keys_in / d_values_in are read only, d_values_out
        // is written only.
        if(m_interop != nullptr)
        {
            auto &scan_by_key_interop_map =
                is_inclusive ? ms_inclusive_scan_by_key_interop_map : ms_exclusive_scan_by_key_interop_map;
            auto it = scan_by_key_interop_map.find(key);
            if(it == scan_by_key_interop_map.end())
            {
                auto compiled = ScanByKeyShader().template compile<is_inclusive, use_initial_value>(
                    m_interop->cuda_device(), m_shared_mem_size, scan_op, equality_op);
                it = import_into_interop_cache(m_interop, scan_by_key_interop_map, key, compiled);
            }
            if(it != scan_by_key_interop_map.end())
            {
                auto kernel = it->second.shader.template kernel<
                    vk_cuda_interop::CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<FlagValueT>, Usage::READ_WRITE>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<KeyValue>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::READ>,
                    vk_cuda_interop::CudaArg<Buffer<ValueType>, Usage::WRITE>,
                    ValueType,
                    uint>();
                cmdlist << kernel(tile_states,
                                  tile_partial,
                                  tile_inclusive,
                                  d_keys_in,
                                  d_prev_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  initial_value,
                                  uint(num_items))
                               .dispatch(scan_threads);
                scan_launched = true;
            }
        }
        if(!scan_launched)
        {
            cmdlist << (*ms_scan_by_key_ptr)(
                           tile_states, tile_partial, tile_inclusive, d_keys_in, d_prev_keys_in, d_values_in, d_values_out, initial_value, num_items)
                           .dispatch(scan_threads);
        }
        return 0;
    }

    detail::ShaderCacheMap ms_scan_key;
    detail::ShaderCacheMap ms_exclusive_scan_map;
    detail::ShaderCacheMap ms_inclusive_scan_map;
    detail::ShaderCacheMap ms_exclusive_scan_transform_map;

    detail::ShaderCacheMap ms_packed_scan_key;
    detail::ShaderCacheMap ms_packed_exclusive_scan_map;
    detail::ShaderCacheMap ms_packed_inclusive_scan_map;
    detail::ShaderCacheMap ms_packed_exclusive_scan_transform_map;

    detail::ShaderCacheMap ms_scan_by_key_tile_state_init_map;
    detail::ShaderCacheMap ms_exclusive_scan_by_key_map;
    detail::ShaderCacheMap ms_inclusive_scan_by_key_map;

    // epoch B: vk-cuda interop kernels shadowing the maps above.  Declared
    // after them so teardown order is the same; they must be destroyed before
    // the vk device the borrowed m_interop points at (risk R6).
    detail::InteropShaderMap ms_scan_key_interop;
    detail::InteropShaderMap ms_exclusive_scan_interop_map;
    detail::InteropShaderMap ms_inclusive_scan_interop_map;
    detail::InteropShaderMap ms_scan_by_key_tile_state_init_interop_map;
    detail::InteropShaderMap ms_exclusive_scan_by_key_interop_map;
    detail::InteropShaderMap ms_inclusive_scan_by_key_interop_map;
};

// CUDA parity policy measured on Blackwell: the generic DSL scan reaches the
// CUB 3.5.0 execution time most closely with 320 threads and 25 items/thread.
// Keep this opt-in so the portable DeviceScan defaults remain unchanged.
using DeviceScanCudaParity = DeviceScan<
    320u,
    details::WARP_SIZE,
    25u,
    BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
    BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
    true,
    true>;
}  // namespace luisa::parallel_primitive
