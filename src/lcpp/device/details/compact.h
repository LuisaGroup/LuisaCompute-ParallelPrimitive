#pragma once

#include <lcpp/block/block_load.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/common/utils.h>
#include <lcpp/device/details/single_pass_scan_operator.h>
#include <lcpp/runtime/core.h>

namespace luisa::parallel_primitive::details
{
using namespace luisa::compute;

enum class CompactSelection
{
    Predicate,
    Flagged,
    Adjacent,
};

// Keep ranks in registers and exchange selected items in shared memory so that
// both the input load and the compacted output store remain coalesced.
template <size_t BLOCK_THREADS, size_t ITEMS_PER_THREAD>
class CompactModule : public LuisaModule
{
    Device                                                           m_device;
    U<Shader<1, Buffer<ulong>, uint>>                                m_init;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<Resource>> m_shaders;

  public:
    static constexpr size_t TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    void create(Device& device) { m_device = device; }

    template <bool PARTITION = false, NumericT T>
    void If(CommandList&             cmdlist,
            BufferView<uint>         temp_storage,
            BufferView<T>            input,
            BufferView<T>            output,
            BufferView<uint>         count,
            size_t                   num_items,
            const Callable<bool(T)>& predicate)
    {
        dispatch<CompactSelection::Predicate, PARTITION, false>(cmdlist,
                                                                temp_storage,
                                                                input,
                                                                input,
                                                                output,
                                                                count,
                                                                count,
                                                                num_items,
                                                                predicate,
                                                                predicate.function().hash());
    }

    template <bool PARTITION = false, NumericT T, NumericT FlagT>
    void Flagged(CommandList&                 cmdlist,
                 BufferView<uint>             temp_storage,
                 BufferView<T>                input,
                 BufferView<FlagT>            flags,
                 BufferView<T>                output,
                 BufferView<uint>             count,
                 size_t                       num_items,
                 const Callable<bool(FlagT)>& predicate)
    {
        dispatch<CompactSelection::Flagged, PARTITION, false>(cmdlist,
                                                              temp_storage,
                                                              input,
                                                              flags,
                                                              output,
                                                              count,
                                                              count,
                                                              num_items,
                                                              predicate,
                                                              predicate.function().hash());
    }

    template <bool STORE_OFFSETS = false, NumericT T>
    void Unique(CommandList&                cmdlist,
                BufferView<uint>            temp_storage,
                BufferView<T>               input,
                BufferView<T>               output,
                BufferView<uint>            count,
                BufferView<uint>            offsets,
                size_t                      num_items,
                const Callable<bool(T, T)>& equality)
    {
        dispatch<CompactSelection::Adjacent, false, STORE_OFFSETS>(cmdlist,
                                                                   temp_storage,
                                                                   input,
                                                                   input,
                                                                   output,
                                                                   count,
                                                                   offsets,
                                                                   num_items,
                                                                   equality,
                                                                   equality.function().hash());
    }

  private:
    template <CompactSelection SELECTION, bool PARTITION, bool STORE_OFFSETS, NumericT T, NumericT FlagT, typename SelectionOp>
    void dispatch(CommandList&      cmdlist,
                  BufferView<uint>  temp_storage,
                  BufferView<T>     input,
                  BufferView<FlagT> flags,
                  BufferView<T>     output,
                  BufferView<uint>  count,
                  BufferView<uint>  offsets,
                  size_t            num_items,
                  SelectionOp       selection,
                  uint64_t          selection_hash)
    {
        static_assert(BLOCK_THREADS % WARP_SIZE == 0u);
        LUISA_ASSERT(m_device.backend_name() == "cuda", "Fused compaction requires the CUDA backend.");
        size_t tiles       = ceil_div(num_items, TILE_ITEMS);
        size_t state_uints = (tiles + WARP_SIZE) * sizeof(ulong) / sizeof(uint);
        LUISA_ASSERT(temp_storage.size() >= state_uints, "Fused compaction temporary storage is too small.");
        auto tile_state = temp_storage.subview(0u, state_uints).template as<ulong>();
        if(!m_init)
        {
            lazy_compile(m_device,
                         m_init,
                         [](BufferVar<ulong> state, UInt num_tiles) noexcept
                         { InitializePackedWardStatus(num_tiles, state); });
        }
        // The leading look-back window needs all padding lanes even for one tile.
        cmdlist << (*m_init)(tile_state, static_cast<uint>(tiles)).dispatch(std::max(tiles, size_t{WARP_SIZE}));

        using Kernel =
            Shader<1, Buffer<ulong>, Buffer<T>, Buffer<FlagT>, Buffer<T>, Buffer<uint>, Buffer<uint>, uint>;
        auto key       = luisa::format("compact+{}+{}+{}+{}+{}+{}",
                                 static_cast<uint>(SELECTION),
                                 PARTITION,
                                 STORE_OFFSETS,
                                 Type::of<T>()->description(),
                                 Type::of<FlagT>()->description(),
                                 selection_hash);
        auto shader_it = m_shaders.find(key);
        if(shader_it == m_shaders.end())
        {
            U<Kernel> shader = nullptr;
            lazy_compile(
                m_device,
                shader,
                [selection](BufferVar<ulong> state,
                            BufferVar<T>     input_buffer,
                            BufferVar<FlagT> flag_buffer,
                            BufferVar<T>     output_buffer,
                            BufferVar<uint>  count_buffer,
                            BufferVar<uint>  offset_buffer,
                            UInt             num_elements) noexcept
                {
                    set_block_size(BLOCK_THREADS);
                    UInt      tile        = block_id().x;
                    UInt      tile_start  = tile * UInt(TILE_ITEMS);
                    UInt      valid_items = min(UInt(TILE_ITEMS), num_elements - tile_start);
                    Shared<T> exchange{TILE_ITEMS};
                    ArrayVar<T, ITEMS_PER_THREAD> items;
                    BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE, true> load{
                        &exchange};
                    $if(valid_items == UInt(TILE_ITEMS))
                    {
                        load.Load(input_buffer, items, tile_start);
                    }
                    $else
                    {
                        load.Load(input_buffer, items, tile_start, valid_items, T{});
                    };

                    ArrayVar<FlagT, ITEMS_PER_THREAD> flag_items;
                    if constexpr(SELECTION == CompactSelection::Flagged)
                    {
                        BlockLoad<FlagT, BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE, true> flag_load;
                        flag_load.Load(flag_buffer, flag_items, tile_start, valid_items, FlagT{});
                    }
                    Var<T> predecessor;
                    if constexpr(SELECTION == CompactSelection::Adjacent && STORE_OFFSETS)
                    {
                        UInt lane = warp_lane_id();
                        predecessor = warp_read_lane(items[ITEMS_PER_THREAD - 1u], max(lane, 1u) - 1u);
                        UInt local = thread_id().x * UInt(ITEMS_PER_THREAD);
                        $if(lane == 0u & local < valid_items & tile_start + local > 0u)
                        {
                            predecessor = input_buffer.read(tile_start + local - 1u);
                        };
                    }
                    ArrayVar<uint, ITEMS_PER_THREAD> selected;
                    for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        UInt local     = thread_id().x * UInt(ITEMS_PER_THREAD) + UInt(item);
                        selected[item] = 0u;
                        $if(local < valid_items)
                        {
                            Bool keep;
                            if constexpr(SELECTION == CompactSelection::Predicate)
                            {
                                keep = selection(items[item]);
                            }
                            else if constexpr(SELECTION == CompactSelection::Flagged)
                            {
                                keep = selection(flag_items[item]);
                            }
                            else
                            {
                                keep = true;
                                $if(tile_start + local > 0u)
                                {
                                    if(item > 0u)
                                    {
                                        keep = !selection(items[item - 1u], items[item]);
                                    }
                                    else
                                    {
                                        if constexpr(STORE_OFFSETS)
                                        {
                                            keep = !selection(predecessor, items[item]);
                                        }
                                        else
                                        {
                                            keep = !selection(input_buffer.read(tile_start + local - 1u),
                                                              items[item]);
                                        }
                                    }
                                };
                            }
                            selected[item] = select(0u, 1u, keep);
                        };
                    }

                    ArrayVar<uint, ITEMS_PER_THREAD> ranks;
                    UInt                             aggregate;
                    auto sum = [](UInt lhs, UInt rhs) { return lhs + rhs; };
                    BlockScan<uint, BLOCK_THREADS, ITEMS_PER_THREAD> scan;
                    scan.ExclusiveSum(selected, ranks, aggregate);
                    PackedScanTileStateViewer<uint> viewer{state};
                    Shared<uint>                    prefix_storage{1u};
                    $if(tile == 0u)
                    {
                        $if(thread_id().x == 0u)
                        {
                            prefix_storage.write(0u, 0u);
                            viewer.SetInclusive(0, aggregate);
                        };
                    }
                    $else
                    {
                        auto prefix_temp = new SmemType<TilePrefixTempStorage<uint>>{1u};
                        $if(thread_id().x < UInt(WARP_SIZE))
                        {
                            TilePrefixCallbackOp<uint, decltype(sum), PackedScanTileStateViewer<uint>> prefix_op{
                                viewer, prefix_temp, sum, tile};
                            UInt prefix = prefix_op(aggregate);
                            $if(thread_id().x == 0u)
                            {
                                prefix_storage.write(0u, prefix);
                            };
                        };
                    };
                    sync_block();
                    UInt prefix = prefix_storage.read(0u);

                    // Reuse the load exchange after every warp has finished reading it.
                    Shared<uint> offset_exchange{STORE_OFFSETS ? TILE_ITEMS : 1u};
                    for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        UInt local = thread_id().x * UInt(ITEMS_PER_THREAD) + UInt(item);
                        $if(selected[item] != 0u)
                        {
                            exchange.write(ranks[item], items[item]);
                            if constexpr(STORE_OFFSETS)
                            {
                                offset_exchange.write(ranks[item], tile_start + local);
                            }
                        };
                        if constexpr(PARTITION)
                        {
                            $if(local < valid_items & selected[item] == 0u)
                            {
                                exchange.write(aggregate + local - ranks[item], items[item]);
                            };
                        }
                    }
                    sync_block();
                    for(size_t item = 0u; item < ITEMS_PER_THREAD; ++item)
                    {
                        UInt local = thread_id().x + UInt(item * BLOCK_THREADS);
                        $if(local < aggregate)
                        {
                            output_buffer.write(prefix + local, exchange.read(local));
                            if constexpr(STORE_OFFSETS)
                            {
                                offset_buffer.write(prefix + local, offset_exchange.read(local));
                            }
                        };
                        if constexpr(PARTITION)
                        {
                            $if(local >= aggregate & local < valid_items)
                            {
                                UInt destination = num_elements - 1u - tile_start + prefix - (local - aggregate);
                                output_buffer.write(destination, exchange.read(local));
                            };
                        }
                    }
                    $if(tile_start + valid_items == num_elements & thread_id().x == 0u)
                    {
                        count_buffer.write(0u, prefix + aggregate);
                    };
                });
            m_shaders.try_emplace(key, std::move(shader));
            shader_it = m_shaders.find(key);
        }
        auto shader = reinterpret_cast<Kernel*>(&(*shader_it->second));
        cmdlist << (*shader)(tile_state, input, flags, output, count, offsets, static_cast<uint>(num_items))
                       .dispatch(tiles * BLOCK_THREADS);
    }
};
}  // namespace luisa::parallel_primitive::details
