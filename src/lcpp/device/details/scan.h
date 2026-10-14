/*
 * @Author: Ligo 
 * @Date: 2025-10-21 23:03:40 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 16:33:11
 */

#pragma once
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/utils.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/device/details/single_pass_scan_operator.h>

namespace luisa::parallel_primitive
{
namespace details
{
    using namespace luisa::compute;

    template <NumericT Type4Byte,
              size_t BLOCK_SIZE       = details::BLOCK_SIZE,
              size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD,
              BlockLoadAlgorithm LOAD_ALGORITHM   = BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
              BlockStoreAlgorithm STORE_ALGORITHM = BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
              bool WARP_SYNC = false>
    class ScanModule : public LuisaModule
    {
      public:
        using TileStateViewer = ScanTileStateViewer<Type4Byte>;

        using ScanTileStateInitKernel = Shader<1, Buffer<uint>, Buffer<Type4Byte>, Buffer<Type4Byte>, uint>;

        using PackedScanTileStateInitKernel = Shader<1, Buffer<ulong>, uint>;

        using ScanKernel =
            Shader<1, Buffer<uint>, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, Type4Byte, uint>;

        using PackedScanKernel = Shader<1, Buffer<ulong>, Buffer<Type4Byte>, Buffer<Type4Byte>, Type4Byte, uint>;

        template <typename InputT>
        using TransformScanKernel =
            Shader<1, Buffer<uint>, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<InputT>, Buffer<Type4Byte>, Type4Byte, uint>;

        template <typename InputT>
        using PackedTransformScanKernel =
            Shader<1, Buffer<ulong>, Buffer<InputT>, Buffer<Type4Byte>, Type4Byte, uint>;

        U<ScanTileStateInitKernel> compile_scan_tile_state_init(Device& device)
        {
            U<ScanTileStateInitKernel> scan_tile_state_init_shader = nullptr;
            lazy_compile(device,
                         scan_tile_state_init_shader,
                         [](BufferVar<uint>      tile_status,
                            BufferVar<Type4Byte> tile_partial,
                            BufferVar<Type4Byte> tile_inclusive,
                            UInt                 num_tiles) noexcept
                         { InitializeWardStatus(num_tiles, tile_status); });
            return scan_tile_state_init_shader;
        }

        template <bool is_inclusive, bool use_initial_value, typename ScanOp>
        U<ScanKernel> compile(Device& device, size_t shared_mem_size, ScanOp scan_op)
        {
            U<ScanKernel> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<uint>      tile_status,
                    BufferVar<Type4Byte> tile_partial,
                    BufferVar<Type4Byte> tile_inclusive,
                    BufferVar<Type4Byte> d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                 num_elements)
                {
                    TileStateViewer tile_state_viewer(tile_status, tile_partial, tile_inclusive);
                    build_scan<is_inclusive, use_initial_value>(
                        shared_mem_size, scan_op, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });

            return scan_shader;
        }

        U<PackedScanTileStateInitKernel> compile_packed_scan_tile_state_init(Device& device)
        {
            LUISA_ASSERT(device.backend_name() == "cuda", "Packed scan tile state requires the CUDA backend.");
            U<PackedScanTileStateInitKernel> scan_tile_state_init_shader = nullptr;
            lazy_compile(device,
                         scan_tile_state_init_shader,
                         [](BufferVar<ulong> tile_state, UInt num_tiles) noexcept
                         { InitializePackedWardStatus(num_tiles, tile_state); });
            return scan_tile_state_init_shader;
        }

        template <bool is_inclusive, bool use_initial_value, typename ScanOp>
        U<PackedScanKernel> compile_packed(Device& device, size_t shared_mem_size, ScanOp scan_op)
        {
            LUISA_ASSERT(device.backend_name() == "cuda", "Packed scan tile state requires the CUDA backend.");
            U<PackedScanKernel> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<ulong>     tile_state,
                    BufferVar<Type4Byte> d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                 num_elements)
                {
                    PackedScanTileStateViewer<Type4Byte> tile_state_viewer(tile_state);
                    build_scan<is_inclusive, use_initial_value>(
                        shared_mem_size, scan_op, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });
            return scan_shader;
        }

        template <typename InputT, typename TransformOp>
        U<TransformScanKernel<InputT>> compile_transform(Device& device, size_t shared_mem_size, TransformOp transform)
        {
            static_assert(sizeof(InputT) == sizeof(Type4Byte));
            U<TransformScanKernel<InputT>> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<uint>      tile_status,
                    BufferVar<Type4Byte> tile_partial,
                    BufferVar<Type4Byte> tile_inclusive,
                    BufferVar<InputT>    d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                  num_elements)
                {
                    TileStateViewer tile_state_viewer(tile_status, tile_partial, tile_inclusive);
                    auto indexed_transform = [transform](Var<InputT> value, UInt) { return transform(value); };
                    build_scan_transform(
                        shared_mem_size, indexed_transform, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });
            return scan_shader;
        }

        template <typename InputT, typename TransformOp>
        U<PackedTransformScanKernel<InputT>> compile_packed_transform(Device& device,
                                                                       size_t shared_mem_size,
                                                                       TransformOp transform)
        {
            static_assert(sizeof(InputT) == sizeof(Type4Byte));
            LUISA_ASSERT(device.backend_name() == "cuda", "Packed scan tile state requires the CUDA backend.");
            U<PackedTransformScanKernel<InputT>> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<ulong>     tile_state,
                    BufferVar<InputT>    d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                  num_elements)
                {
                    PackedScanTileStateViewer<Type4Byte> tile_state_viewer(tile_state);
                    auto indexed_transform = [transform](Var<InputT> value, UInt) { return transform(value); };
                    build_scan_transform(
                        shared_mem_size, indexed_transform, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });
            return scan_shader;
        }

        template <typename InputT, typename EqualityOp>
        U<TransformScanKernel<InputT>> compile_adjacent_transform(Device& device,
                                                                   size_t shared_mem_size,
                                                                   EqualityOp equality)
        {
            static_assert(sizeof(InputT) == sizeof(Type4Byte));
            U<TransformScanKernel<InputT>> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<uint>      tile_status,
                    BufferVar<Type4Byte> tile_partial,
                    BufferVar<Type4Byte> tile_inclusive,
                    BufferVar<InputT>    d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                  num_elements)
                {
                    TileStateViewer tile_state_viewer(tile_status, tile_partial, tile_inclusive);
                    auto indexed_transform = [equality, &d_in](Var<InputT> value, UInt index) {
                        Bool selected = true;
                        $if(index > 0u)
                        {
                            selected = !equality(d_in.read(index - 1u), value);
                        };
                        return selected;
                    };
                    build_scan_transform(
                        shared_mem_size, indexed_transform, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });
            return scan_shader;
        }

        template <typename InputT, typename EqualityOp>
        U<PackedTransformScanKernel<InputT>> compile_packed_adjacent_transform(Device& device,
                                                                                size_t shared_mem_size,
                                                                                EqualityOp equality)
        {
            static_assert(sizeof(InputT) == sizeof(Type4Byte));
            LUISA_ASSERT(device.backend_name() == "cuda", "Packed scan tile state requires the CUDA backend.");
            U<PackedTransformScanKernel<InputT>> scan_shader = nullptr;
            lazy_compile(
                device,
                scan_shader,
                [&](BufferVar<ulong>     tile_state,
                    BufferVar<InputT>    d_in,
                    BufferVar<Type4Byte> d_out,
                    Var<Type4Byte>       init_value,
                    UInt                  num_elements)
                {
                    PackedScanTileStateViewer<Type4Byte> tile_state_viewer(tile_state);
                    auto indexed_transform = [equality, &d_in](Var<InputT> value, UInt index) {
                        Bool selected = true;
                        $if(index > 0u)
                        {
                            selected = !equality(d_in.read(index - 1u), value);
                        };
                        return selected;
                    };
                    build_scan_transform(
                        shared_mem_size, indexed_transform, tile_state_viewer, d_in, d_out, init_value, num_elements);
                });
            return scan_shader;
        }

      private:
        template <bool is_inclusive, bool use_initial_value, typename ScanOp, typename TileState>
        static void build_scan(size_t                 shared_mem_size,
                               ScanOp                 scan_op,
                               TileState&             tile_state_viewer,
                               BufferVar<Type4Byte>& d_in,
                               BufferVar<Type4Byte>& d_out,
                               Var<Type4Byte>&       init_value,
                               UInt                  num_elements)
        {
                    set_block_size(BLOCK_SIZE);
                    UInt tile_id    = block_id().x;
                    UInt tile_items = UInt(ITEMS_PER_THREAD) * block_size_x();
                    UInt tile_start = tile_id * tile_items;

                    UInt num_remaining = num_elements - tile_start;
                    Bool is_last_tile  = num_remaining <= tile_items;

                    ArrayVar<Type4Byte, ITEMS_PER_THREAD> items;
                    SmemTypePtr<Type4Byte> s_data = new SmemType<Type4Byte>{shared_mem_size};
                    // Coalesced (block-striped) global I/O transposed through shared
                    // memory so the per-thread register array stays in the blocked,
                    // thread-major order that BlockScan requires for an order-correct
                    // prefix sum. Thread `thid` fetches tile positions
                    // {i*BLOCK + thid} (adjacent threads -> adjacent addresses) into
                    // shared, then reads the blocked range {thid*ITEMS + i} back into
                    // registers. The end bound `num_remaining` is compared against the
                    // RELATIVE tile position before adding `tile_start`, exactly as the
                    // previous guarded DIRECT load did.
                    UInt block_stride = UInt(BLOCK_SIZE);
                    $if(is_last_tile)
                    {
                        BlockLoad<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, LOAD_ALGORITHM, WARP_SYNC>(s_data).Load(
                            d_in, items, tile_start, num_remaining);
                    }
                    $else
                    {
                        BlockLoad<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, LOAD_ALGORITHM, WARP_SYNC>(s_data).Load(
                            d_in, items, tile_start);
                    };

                    ArrayVar<Type4Byte, ITEMS_PER_THREAD> output_items;
                    $if(tile_id == 0u)
                    {
                        Var<Type4Byte> block_aggregate;
                        BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                        if constexpr(is_inclusive)
                        {
                            if constexpr(use_initial_value)
                            {
                                block_scan.InclusiveScan(items, output_items, block_aggregate, scan_op, init_value);
                                block_aggregate = scan_op(init_value, block_aggregate);
                            }
                            else
                            {
                                block_scan.InclusiveScan(items, output_items, block_aggregate, scan_op);
                            }
                        }
                        else
                        {
                            block_scan.ExclusiveScan(items, output_items, block_aggregate, scan_op, init_value);
                            block_aggregate = scan_op(init_value, block_aggregate);
                        }
                        $if(!is_last_tile & thread_id().x == 0u)
                        {
                            tile_state_viewer.SetInclusive(0, block_aggregate);
                        };
                    }
                    $else
                    {
                        auto temp_storage = new SmemType<TilePrefixTempStorage<Type4Byte>>{1};
                        TilePrefixCallbackOp<Type4Byte, ScanOp, TileState> prefix_op(
                            tile_state_viewer, temp_storage, scan_op, tile_id);
                        BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                        if constexpr(is_inclusive)
                        {
                            block_scan.InclusiveScan(items, output_items, scan_op, prefix_op);
                        }
                        else
                        {
                            block_scan.ExclusiveScan(items, output_items, scan_op, prefix_op);
                        }
                    };

                    $if(is_last_tile)
                    {
                        BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, STORE_ALGORITHM, WARP_SYNC>(s_data).Store(
                            output_items, d_out, tile_start, num_remaining);
                    }
                    $else
                    {
                        BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, STORE_ALGORITHM, WARP_SYNC>(s_data).Store(
                            output_items, d_out, tile_start);

                    };
        }

        template <typename InputT, typename TransformOp, typename TileState>
        static void build_scan_transform(size_t                 shared_mem_size,
                                          TransformOp            transform,
                                          TileState&              tile_state_viewer,
                                          BufferVar<InputT>&      d_in,
                                          BufferVar<Type4Byte>&   d_out,
                                          Var<Type4Byte>&         init_value,
                                          UInt                    num_elements)
        {
            set_block_size(BLOCK_SIZE);
            UInt tile_id    = block_id().x;
            UInt tile_items = UInt(ITEMS_PER_THREAD) * block_size_x();
            UInt tile_start = tile_id * tile_items;

            UInt num_remaining = num_elements - tile_start;
            Bool is_last_tile  = num_remaining <= tile_items;

            ArrayVar<InputT, ITEMS_PER_THREAD> input_items;
            SmemTypePtr<InputT> input_shared = new SmemType<InputT>{shared_mem_size};
            SmemTypePtr<Type4Byte> output_shared = new SmemType<Type4Byte>{shared_mem_size};
            $if(is_last_tile)
            {
                BlockLoad<InputT, BLOCK_SIZE, ITEMS_PER_THREAD, LOAD_ALGORITHM, WARP_SYNC>(input_shared)
                    .Load(d_in, input_items, tile_start, num_remaining, InputT{});
            }
            $else
            {
                BlockLoad<InputT, BLOCK_SIZE, ITEMS_PER_THREAD, LOAD_ALGORITHM, WARP_SYNC>(input_shared)
                    .Load(d_in, input_items, tile_start);
            };

            ArrayVar<Type4Byte, ITEMS_PER_THREAD> items;
            for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
            {
                UInt local_index = thread_id().x * UInt(ITEMS_PER_THREAD) + UInt(item);
                items[item] = Type4Byte{};
                $if(local_index < num_remaining)
                {
                    items[item] = select(Type4Byte{}, Type4Byte{1}, transform(input_items[item], tile_start + local_index));
                };
            }

            ArrayVar<Type4Byte, ITEMS_PER_THREAD> output_items;
            auto scan_op = [](Var<Type4Byte> a, Var<Type4Byte> b) { return a + b; };
            $if(tile_id == 0u)
            {
                Var<Type4Byte> block_aggregate;
                BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                block_scan.ExclusiveScan(items, output_items, block_aggregate, scan_op, init_value);
                block_aggregate = block_aggregate + init_value;
                $if(!is_last_tile & thread_id().x == 0u)
                {
                    tile_state_viewer.SetInclusive(0, block_aggregate);
                };
            }
            $else
            {
                auto temp_storage = new SmemType<TilePrefixTempStorage<Type4Byte>>{1};
                TilePrefixCallbackOp<Type4Byte, decltype(scan_op), TileState> prefix_op(
                    tile_state_viewer, temp_storage, scan_op, tile_id);
                BlockScan<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD> block_scan;
                block_scan.ExclusiveScan(items, output_items, scan_op, prefix_op);
            };

            $if(is_last_tile)
            {
                BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, STORE_ALGORITHM, WARP_SYNC>(output_shared)
                    .Store(output_items, d_out, tile_start, num_remaining);
            }
            $else
            {
                BlockStore<Type4Byte, BLOCK_SIZE, ITEMS_PER_THREAD, STORE_ALGORITHM, WARP_SYNC>(output_shared)
                    .Store(output_items, d_out, tile_start);
            };
        }
    };
};  // namespace details
}  // namespace luisa::parallel_primitive
