/*
 * @Author: Ligo 
 * @Date: 2025-09-19 14:24:07 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 17:31:05
 */

#pragma once
#include <algorithm>
#include <lcpp/common/grid_even_shared.h>
#include <luisa/core/mathematics.h>
#include <luisa/dsl/local.h>
#include <limits>
#include <luisa/core/basic_traits.h>
#include <luisa/ast/type.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/struct.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/var.h>
#include <cstddef>
#include <utility>
#include <lcpp/runtime/core.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>
#include <lcpp/common/thread_operators.h>
#include <lcpp/block/block_reduce.h>
#include <lcpp/block/block_scan.h>
#include <lcpp/block/block_load.h>
#include <lcpp/block/block_store.h>
#include <lcpp/block/block_discontinuity.h>
#include <lcpp/warp/warp_reduce.h>
#include <lcpp/device/details/reduce.h>
#include <lcpp/device/details/reduce_by_key.h>
namespace luisa::parallel_primitive
{

using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE,
          size_t WARP_NUMS = details::WARP_SIZE,
          size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD,
          // Force the CUDA redux variant when true; the default also selects
          // it automatically after create() when the backend is CUDA.
          bool   CUDA_WARP_REDUX = false,
          // Use the larger CUDA tile for large scalar reductions while
          // retaining the caller's policy for small inputs and other backends.
          bool   AUTO_CUDA_TUNED_POLICY = true>
class DeviceReduce : public LuisaModule
{
  private:
    uint m_block_size = BLOCK_SIZE;
    uint m_warp_nums  = WARP_NUMS;

    uint   m_shared_mem_size = 0;
    Device m_device;

    bool m_created = false;

    template <size_t PolicyBlockSize, size_t PolicyItemsPerThread>
    [[nodiscard]] static constexpr size_t policy_shared_mem_size() noexcept
    {
        constexpr size_t tile_items = PolicyBlockSize * PolicyItemsPerThread;
        return tile_items + tile_items / WARP_NUMS;
    }

  public:
    DeviceReduce()  = default;
    ~DeviceReduce() = default;

#ifndef NDEBUG
    Stream* m_debug_stream;  // bind debug stream for sync
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

    /// Temp storage bytes for Reduce / Sum / Min / Max / TransformReduce
    template <typename Type4Byte>
    static size_t GetTempStorageBytes(size_t num_item)
    {
        return reduce_temp_storage_bytes<Type4Byte>(num_item);
    }

    /// Temp storage bytes for ArgMin / ArgMax
    /// Needs: IndexValuePairT<Type4Byte> temp for reduce + d_in_kv(num_item) + d_out_kv(1)
    template <typename Type4Byte>
    static size_t GetArgTempStorageBytes(size_t num_item)
    {
        using IVP    = IndexValuePairT<Type4Byte>;
        size_t bytes = 0;
        bytes += reduce_temp_storage_bytes<IVP>(num_item);  // reduce temp
        bytes = align_up_temp_storage(bytes);
        bytes += num_item * sizeof(IVP);  // d_in_kv
        bytes = align_up_temp_storage(bytes);
        bytes += 1 * sizeof(IVP);  // d_out_kv
        return bytes;
    }

    /// Temp storage bytes for ReduceByKey
    template <typename KeyType, typename ValueType>
    static size_t GetReduceByKeyTempStorageBytes(size_t num_elements)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        size_t num_tiles = std::max<size_t>(1u, ceil_div(num_elements, ITEMS_PER_THREAD * BLOCK_SIZE));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t bytes = 0;
        bytes += tile_count * sizeof(uint);  // tile_states
        bytes = align_up_temp_storage(bytes);
        bytes += tile_count * sizeof(FlagValuePairT);  // tile_partial
        bytes = align_up_temp_storage(bytes);
        bytes += tile_count * sizeof(FlagValuePairT);  // tile_inclusive
        return bytes;
    }

    // ============================================================
    // Dispatch APIs (CUB-style: caller provides temp_storage)
    // ============================================================

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                size_t                num_item,
                ReduceOp              reduce_op,
                Type4Byte             initial_value)
    {
        LUISA_ASSERT(num_item <= std::numeric_limits<uint>::max(), "Too many items for DeviceReduce.");
        if(num_item == 0u)
        {
            write_scalar(cmdlist, d_out, initial_value);
            return;
        }
        auto temp_view = temp_storage
                             .subview(0, bytes_to_uint_count(reduce_temp_storage_bytes<Type4Byte>(num_item)))
                             .template as<Type4Byte>();

        lcpp_check(reduce_array_recursive<Type4Byte>(
                       cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, initial_value, IdentityOp()),
                   cmdlist,
                   debug_stream());
    }

    template <NumericT Type4Byte, typename ReduceOp>
    void Reduce(CommandList&                           cmdlist,
                BufferView<uint>                       temp_storage,
                BufferView<IndexValuePairT<Type4Byte>> d_in,
                BufferView<IndexValuePairT<Type4Byte>> d_out,
                size_t                                 num_item,
                ReduceOp                               reduce_op,
                IndexValuePairT<Type4Byte>             initial_value)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        LUISA_ASSERT(num_item <= std::numeric_limits<uint>::max(), "Too many items for DeviceReduce.");
        if(num_item == 0u)
        {
            write_scalar(cmdlist, d_out, initial_value);
            return;
        }
        auto temp_view =
            temp_storage.subview(0, bytes_to_uint_count(reduce_temp_storage_bytes<IVP>(num_item))).template as<IVP>();

        lcpp_check(reduce_array_recursive<IVP>(
                       cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, initial_value, IdentityOp()),
                   cmdlist,
                   debug_stream());
    }


    template <NumericT Type4Byte>
    void Sum(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, SumOp(), Type4Byte(0));
    }

    template <NumericT Type4Byte>
    void Min(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, MinOp(), std::numeric_limits<Type4Byte>::max());
    }

    template <NumericT Type4Byte>
    void Max(CommandList&          cmdlist,
             BufferView<uint>      temp_storage,
             BufferView<Type4Byte> d_in,
             BufferView<Type4Byte> d_out,
             size_t                num_item)
    {
        Reduce(cmdlist, temp_storage, d_in, d_out, num_item, MaxOp(), std::numeric_limits<Type4Byte>::lowest());
    }

    template <NumericT Type4Byte>
    void ArgMin(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        if(num_item == 0u)
        {
            write_scalar(cmdlist, d_out, std::numeric_limits<Type4Byte>::max());
            write_scalar(cmdlist, d_index_out, 1u);
            return;
        }
        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = bytes_to_uint_count(reduce_temp_storage_bytes<IVP>(num_item));
        auto reduce_temp =
            temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // d_in_kv
        size_t in_kv_uint_count = bytes_to_uint_count(num_item * sizeof(IVP));
        auto   d_in_kv =
            temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // d_out_kv
        size_t out_kv_uint_count = bytes_to_uint_count(sizeof(IVP));
        auto   d_out_kv =
            temp_storage.subview(offset_bytes / sizeof(uint), out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv, num_item), cmdlist, debug_stream());

        Reduce(cmdlist,
               temp_storage.subview(0, reduce_temp_uint_count),
               d_in_kv,
               d_out_kv,
               num_item,
               ArgMinOp(),
               IVP{std::numeric_limits<uint>::max(), std::numeric_limits<Type4Byte>::max()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<Type4Byte>(cmdlist, d_out_kv, d_out, d_index_out), cmdlist, debug_stream());
    }

    template <NumericT Type4Byte, typename CompareOp>
    void ArgMin(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item,
                CompareOp             compare_op)
    {
        dispatch_arg_extremum<true>(cmdlist, temp_storage, d_in, d_out, d_index_out, num_item, std::move(compare_op));
    }

    template <NumericT Type4Byte>
    void ArgMax(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item)
    {
        using IVP = IndexValuePairT<Type4Byte>;
        if(num_item == 0u)
        {
            write_scalar(cmdlist, d_out, std::numeric_limits<Type4Byte>::lowest());
            write_scalar(cmdlist, d_index_out, 1u);
            return;
        }
        size_t offset_bytes = 0;
        // reduce temp
        size_t reduce_temp_uint_count = bytes_to_uint_count(reduce_temp_storage_bytes<IVP>(num_item));
        auto reduce_temp =
            temp_storage.subview(offset_bytes / sizeof(uint), reduce_temp_uint_count).template as<IVP>();
        offset_bytes += reduce_temp_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // d_in_kv
        size_t in_kv_uint_count = bytes_to_uint_count(num_item * sizeof(IVP));
        auto   d_in_kv =
            temp_storage.subview(offset_bytes / sizeof(uint), in_kv_uint_count).template as<IVP>();
        offset_bytes += in_kv_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // d_out_kv
        size_t out_kv_uint_count = bytes_to_uint_count(sizeof(IVP));
        auto   d_out_kv =
            temp_storage.subview(offset_bytes / sizeof(uint), out_kv_uint_count).template as<IVP>();

        // construct key value pair
        lcpp_check(arg_construct<Type4Byte>(cmdlist, d_in, d_in_kv, num_item), cmdlist, debug_stream());

        Reduce(cmdlist,
               temp_storage.subview(0, reduce_temp_uint_count),
               d_in_kv,
               d_out_kv,
               num_item,
               ArgMaxOp(),
               IVP{std::numeric_limits<uint>::max(), std::numeric_limits<Type4Byte>::lowest()});

        // copy result to d_out and d_index_out
        lcpp_check(arg_assign<Type4Byte>(cmdlist, d_out_kv, d_out, d_index_out), cmdlist, debug_stream());
    }

    template <NumericT Type4Byte, typename CompareOp>
    void ArgMax(CommandList&          cmdlist,
                BufferView<uint>      temp_storage,
                BufferView<Type4Byte> d_in,
                BufferView<Type4Byte> d_out,
                BufferView<uint>      d_index_out,
                size_t                num_item,
                CompareOp             compare_op)
    {
        dispatch_arg_extremum<false>(cmdlist, temp_storage, d_in, d_out, d_index_out, num_item, std::move(compare_op));
    }


    template <NumericT KeyType, NumericT ValueType, typename ReduceOp>
    void ReduceByKey(CommandList&          cmdlist,
                     BufferView<uint>      temp_storage,
                     BufferView<KeyType>   d_keys_in,
                     BufferView<ValueType> d_values_in,
                     BufferView<KeyType>   d_unique_out,
                     BufferView<ValueType> g_aggregates_out,
                     BufferView<uint>      g_num_runs_out,
                     ReduceOp              reduce_op,
                     size_t                num_elements)
    {
        using FlagValuePairT = KeyValuePair<int, ValueType>;
        auto ptr = luisa::new_with_allocator<luisa::uint>(0u);  // retain until the cmdlist commit
        cmdlist << g_num_runs_out.copy_from(luisa::span<luisa::uint>{ptr, 1u});
        cmdlist.add_dtor_callback([ptr] { luisa::delete_with_allocator(ptr); });
        if(num_elements == 0u)
        {
            return;
        }

        size_t num_tiles = std::max<size_t>(1u, ceil_div(num_elements, ITEMS_PER_THREAD * m_block_size));
        size_t tile_count = details::WARP_SIZE + num_tiles;

        size_t offset_bytes = 0;
        // tile_states: tile_count * uint
        auto tile_states = temp_storage.subview(offset_bytes / sizeof(uint), tile_count);
        offset_bytes += tile_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // tile_partial: tile_count * FlagValuePairT
        size_t partial_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_partial =
            temp_storage.subview(offset_bytes / sizeof(uint), partial_uint_count).template as<FlagValuePairT>();
        offset_bytes += partial_uint_count * sizeof(uint);
        offset_bytes = align_up_temp_storage(offset_bytes);

        // tile_inclusive: tile_count * FlagValuePairT
        size_t inclusive_uint_count = bytes_to_uint_count(tile_count * sizeof(FlagValuePairT));
        auto   tile_inclusive =
            temp_storage.subview(offset_bytes / sizeof(uint), inclusive_uint_count).template as<FlagValuePairT>();

        lcpp_check(reduce_by_key_array<KeyType, ValueType>(
                       cmdlist, tile_states, tile_partial, tile_inclusive, d_keys_in, d_values_in, d_unique_out, g_aggregates_out, g_num_runs_out, reduce_op, num_elements),
                   cmdlist,
                   debug_stream());
    }


    template <typename Type4Byte, typename ReduceOp, typename TransformOp>
    void TransformReduce(CommandList&          cmdlist,
                         BufferView<uint>      temp_storage,
                         BufferView<Type4Byte> d_in,
                         BufferView<Type4Byte> d_out,
                         size_t                num_item,
                         ReduceOp              reduce_op,
                         TransformOp           transform_op,
                         Type4Byte             init)
    {
        LUISA_ASSERT(num_item <= std::numeric_limits<uint>::max(), "Too many items for DeviceReduce.");
        if(num_item == 0u)
        {
            write_scalar(cmdlist, d_out, init);
            return;
        }
        auto temp_view = temp_storage
                             .subview(0, bytes_to_uint_count(reduce_temp_storage_bytes<Type4Byte>(num_item)))
                             .template as<Type4Byte>();
        lcpp_check(reduce_array_recursive<Type4Byte, ReduceOp, TransformOp>(
                       cmdlist, temp_view, d_in, d_out, num_item, 0, 0, reduce_op, init, transform_op),
                   cmdlist,
                   debug_stream());
    }

  private:
    template <typename Type>
    [[nodiscard]] static size_t reduce_temp_storage_bytes(size_t num_items)
    {
        size_t bytes       = 0u;
        size_t level_items = num_items;
        bool   first_level = true;
        do
        {
            size_t num_blocks =
                std::max<size_t>(1u, ceil_div(level_items, static_cast<size_t>(BLOCK_SIZE * ITEMS_PER_THREAD)));
            if(!first_level)
            {
                bytes = align_up_temp_storage(bytes);
            }
            bytes += num_blocks * sizeof(Type);
            level_items = num_blocks;
            first_level = false;
        } while(level_items > 1u);
        return bytes;
    }

    template <bool FIND_MIN, NumericT Type4Byte, typename CompareOp>
    void dispatch_arg_extremum(CommandList&          cmdlist,
                               BufferView<uint>      temp_storage,
                               BufferView<Type4Byte> d_in,
                               BufferView<Type4Byte> d_out,
                               BufferView<uint>      d_index_out,
                               size_t                num_item,
                               CompareOp             compare_op)
    {
        static_cast<void>(temp_storage);
        LUISA_ASSERT(m_created, "DeviceReduce::create must be called before dispatch.");
        LUISA_ASSERT(num_item <= std::numeric_limits<uint>::max(), "Too many items for DeviceReduce.");
        LUISA_ASSERT(d_out.size() >= 1u, "DeviceReduce extremum output view is empty.");
        LUISA_ASSERT(d_index_out.size() >= 1u, "DeviceReduce extremum index output view is empty.");
        if(num_item == 0u)
        {
            write_scalar(cmdlist,
                         d_out,
                         FIND_MIN ? std::numeric_limits<Type4Byte>::max() :
                                    std::numeric_limits<Type4Byte>::lowest());
            write_scalar(cmdlist, d_index_out, 1u);
            return;
        }
        LUISA_ASSERT(d_in.size() >= num_item, "DeviceReduce extremum input view is too small.");

        using ArgExtremumShader = Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<uint>, uint>;
        auto key       = luisa::format("device_arg_{}+{}+{}",
                                 FIND_MIN ? "min" : "max",
                                 Type::of<Type4Byte>()->description(),
                                 get_binary_op_hash<Type4Byte>(compare_op));
        auto shader_it = ms_arg_extremum_map.find(key);
        if(shader_it == ms_arg_extremum_map.end())
        {
            U<ArgExtremumShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [compare_op](BufferVar<Type4Byte> input,
                                      BufferVar<Type4Byte> value_output,
                                      BufferVar<uint>      index_output,
                                      UInt                 num_items) noexcept
                         {
                             set_block_size(details::WARP_SIZE);
                             $if(dispatch_id().x == 0u)
                             {
                                 Var<Type4Byte> best_value = input.read(0u);
                                 UInt           best_index = 0u;
                                 $for(index, 1u, num_items, 1u)
                                 {
                                     Var<Type4Byte> candidate = input.read(index);
                                     Bool           candidate_is_better;
                                     if constexpr(FIND_MIN)
                                     {
                                         candidate_is_better = compare_op(candidate, best_value);
                                     }
                                     else
                                     {
                                         candidate_is_better = compare_op(best_value, candidate);
                                     }
                                     $if(candidate_is_better)
                                     {
                                         best_value = candidate;
                                         best_index = index;
                                     };
                                 };
                                 value_output.write(0u, best_value);
                                 index_output.write(0u, best_index);
                             };
                         });
            ms_arg_extremum_map.try_emplace(key, std::move(shader));
            shader_it = ms_arg_extremum_map.find(key);
        }
        auto shader = reinterpret_cast<ArgExtremumShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile DeviceReduce custom extremum shader.");
        cmdlist << (*shader)(d_in, d_out, d_index_out, static_cast<uint>(num_item)).dispatch(1u);
    }

    template <NumericT Type4Byte>
    [[nodiscard]] int arg_construct(CommandList&                           cmdlist,
                                    BufferView<Type4Byte>                  d_in,
                                    BufferView<IndexValuePairT<Type4Byte>> d_kv_out,
                                    size_t                                 num_items) noexcept
    {
        using ArgReduce          = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgConstructShader = ArgReduce::ArgConstructShaderT;
        auto key = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_construct_it = ms_arg_construct_map.find(key);
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            auto shader = ArgReduce().compile_arg_construct_shader(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_arg_construct_map.try_emplace(key, std::move(shader));
            ms_arg_construct_it = ms_arg_construct_map.find(key);
        }
        if(ms_arg_construct_it == ms_arg_construct_map.end())
        {
            return -1;
        }
        auto ms_arg_construct_ptr = reinterpret_cast<ArgConstructShader*>(&(*ms_arg_construct_it->second));
        if(!ms_arg_construct_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_arg_construct_ptr)(d_in, d_kv_out).dispatch(num_items);
        return 0;
    }

    template <NumericT Type4Byte>
    [[nodiscard]] int arg_assign(CommandList&                           cmdlist,
                                 BufferView<IndexValuePairT<Type4Byte>> d_kv_in,
                                 BufferView<Type4Byte>                  d_value_out,
                                 BufferView<uint>                       d_index_out) noexcept
    {
        using ArgReduce       = details::ArgReduce<Type4Byte, BLOCK_SIZE>;
        using ArgAssignShader = ArgReduce::ArgAssignShaderT;
        auto key              = luisa::string{luisa::compute::Type::of<Type4Byte>()->description()};
        auto ms_arg_assign_it = ms_arg_assign_map.find(key);
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            auto shader = ArgReduce().compile_arg_assign_shader(m_device);
            if(!shader)
            {
                return -1;
            }
            ms_arg_assign_map.try_emplace(key, std::move(shader));
            ms_arg_assign_it = ms_arg_assign_map.find(key);
        }
        if(ms_arg_assign_it == ms_arg_assign_map.end())
        {
            return -1;
        }
        auto ms_arg_assign_ptr = reinterpret_cast<ArgAssignShader*>(&(*ms_arg_assign_it->second));
        if(!ms_arg_assign_ptr)
        {
            return -1;
        }
        cmdlist << (*ms_arg_assign_ptr)(d_kv_in, d_value_out, d_index_out).dispatch(d_index_out.size());
        return 0;
    }

    template <bool USE_CUDA_WARP_REDUX,
              size_t PolicyBlockSize = BLOCK_SIZE,
              size_t PolicyItemsPerThread = ITEMS_PER_THREAD,
              NumericTOrKeyValuePairT Type,
              typename ReduceOp,
              typename TransformOp = IdentityOp>
    [[nodiscard]] int reduce_array_recursive_impl(luisa::compute::CommandList& cmdlist,
                                                  BufferView<Type>             temp_storage,
                                                  BufferView<Type>             arr_in,
                                                  BufferView<Type>             arr_out,
                                                  uint                         num_items,
                                                  uint                         offset,
                                                  uint                         level,
                                                  ReduceOp                     reduce_op,
                                                  Type                         init,
                                                  TransformOp transform_op = IdentityOp()) noexcept
    {
        constexpr uint policy_block_size = static_cast<uint>(PolicyBlockSize);
        constexpr uint policy_items_per_thread = static_cast<uint>(PolicyItemsPerThread);
        uint tile_items = policy_block_size * policy_items_per_thread;
        uint num_tiles  = std::max(1u, ceil_div(num_items, tile_items));

        // reduce_device_occupancy × subscription_factor
        // reduce_device_occupancy = sm_occupancy × sm_count
        // sm_occupancy = 8 sm_count = 128, subscription_factor = 5
        constexpr auto max_blocks = PolicyBlockSize * 8 * 5;

        using ReduceShader = details::ReduceModule<Type, PolicyBlockSize, PolicyItemsPerThread, CUDA_WARP_REDUX>;
        using ReduceKernel           = ReduceShader::ReduceShaderKernel;
        using ReduceSingleTileShader = ReduceShader::ReduceSingleTileShaderKernel;
        constexpr auto shader_shared_mem_size = policy_shared_mem_size<PolicyBlockSize, PolicyItemsPerThread>();

        size_t           size_elements     = temp_storage.size() - offset;
        BufferView<Type> temp_buffer_level = temp_storage.subview(offset, size_elements);

        if(num_tiles > 1)
        {
            auto key = get_type_and_op_desc<Type>(reduce_op, transform_op)
                       + luisa::format("+policy:{}x{}", PolicyBlockSize, PolicyItemsPerThread);
            auto ms_reduce_it = ms_reduce_map.find(key);
            if(ms_reduce_it == ms_reduce_map.end())
            {
                auto shader = ReduceShader().compile(m_device, shader_shared_mem_size, reduce_op, transform_op);
                ms_reduce_map.try_emplace(key, std::move(shader));
                ms_reduce_it = ms_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceKernel*>(&(*ms_reduce_it->second));

            GridEvenShared even_share;
            even_share.DispatchInit(num_items, max_blocks, tile_items);
            cmdlist << (*ms_reduce_ptr)(arr_in, temp_buffer_level, num_items, even_share)
                           .dispatch(policy_block_size * num_tiles);
            uint next_level_offset = static_cast<uint>(aligned_temp_storage_count<Type>(num_tiles));
            lcpp_check(reduce_array_recursive_impl<USE_CUDA_WARP_REDUX, PolicyBlockSize, PolicyItemsPerThread, Type>(
                           cmdlist,
                           temp_buffer_level,
                           temp_buffer_level,
                           arr_out,
                           num_tiles,
                           next_level_offset,
                           level + 1,
                           reduce_op,
                           init,
                           IdentityOp()),
                       cmdlist,
                       debug_stream());
        }
        else
        {
            auto key = get_type_and_op_desc<Type>(reduce_op, transform_op)
                       + luisa::format("+policy:{}x{}", PolicyBlockSize, PolicyItemsPerThread);
            auto ms_reduce_it = ms_single_reduce_map.find(key);
            if(ms_reduce_it == ms_single_reduce_map.end())
            {
                auto shader = ReduceShader().compile_single_tile(m_device, shader_shared_mem_size, reduce_op, transform_op);
                ms_single_reduce_map.try_emplace(key, std::move(shader));
                ms_reduce_it = ms_single_reduce_map.find(key);
            }
            auto ms_reduce_ptr = reinterpret_cast<ReduceSingleTileShader*>(&(*ms_reduce_it->second));
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(arr_in, arr_out, num_items, init).dispatch(policy_block_size);
        }
        return 0;
    };

    template <NumericTOrKeyValuePairT Type, typename ReduceOp, typename TransformOp = IdentityOp>
    [[nodiscard]] int reduce_array_recursive(luisa::compute::CommandList& cmdlist,
                                             BufferView<Type>             temp_storage,
                                             BufferView<Type>             arr_in,
                                             BufferView<Type>             arr_out,
                                             uint                         num_items,
                                             uint                         offset,
                                             uint                         level,
                                             ReduceOp                     reduce_op,
                                             Type                         init,
                                             TransformOp transform_op = IdentityOp()) noexcept
    {
        // CUB's integer-sum path uses the SM native redux instruction.  Keep
        // that optimization in a CUDA-only shader variant; other backends
        // retain the portable shuffle implementation.
        if constexpr(CUDA_WARP_REDUX)
        {
            return reduce_array_recursive_impl<true, BLOCK_SIZE, ITEMS_PER_THREAD, Type>(
                cmdlist, temp_storage, arr_in, arr_out, num_items, offset, level, reduce_op, init, transform_op);
        }
        else if(m_device.backend_name() == "cuda")
        {
            if constexpr(AUTO_CUDA_TUNED_POLICY && NumericT<Type> && sizeof(Type) == sizeof(uint))
            {
                if(num_items >= 2u * 1024u * 1024u &&
                   BLOCK_SIZE == details::BLOCK_SIZE && ITEMS_PER_THREAD == details::ITEMS_PER_THREAD)
                {
                    return reduce_array_recursive_impl<true, details::BLOCK_SIZE, 8u, Type>(
                        cmdlist, temp_storage, arr_in, arr_out, num_items, offset, level, reduce_op, init, transform_op);
                }
            }
            return reduce_array_recursive_impl<true, BLOCK_SIZE, ITEMS_PER_THREAD, Type>(
                cmdlist, temp_storage, arr_in, arr_out, num_items, offset, level, reduce_op, init, transform_op);
        }
        else
        {
            return reduce_array_recursive_impl<false, BLOCK_SIZE, ITEMS_PER_THREAD, Type>(
                cmdlist, temp_storage, arr_in, arr_out, num_items, offset, level, reduce_op, init, transform_op);
        }
    };


    template <NumericT KeyType, NumericT ValueType, typename ReduceOp, typename FlagValuePairT = KeyValuePair<int, ValueType>>
    [[nodiscard]] int reduce_by_key_array(luisa::compute::CommandList& cmdlist,
                                          BufferView<uint>             tile_states,
                                          BufferView<FlagValuePairT>   tile_partial,
                                          BufferView<FlagValuePairT>   tile_inclusive,
                                          BufferView<KeyType>          keys_in,
                                          BufferView<ValueType>        values_in,
                                          BufferView<KeyType>          unique_out,
                                          BufferView<ValueType>        aggregated_out,
                                          BufferView<uint>             num_runs_out,
                                          ReduceOp                     reduce_op,
                                          uint                         num_items) noexcept
    {
        uint tile_items = m_block_size * ITEMS_PER_THREAD;
        uint num_tiles  = std::max(1u, ceil_div(num_items, tile_items));


        using ReduceByKey = details::ReduceByKeyModule<KeyType, ValueType, BLOCK_SIZE, ITEMS_PER_THREAD>;
        using ReduceByKeyTileStateInitKernel = ReduceByKey::ScanTileStateInitKernel;
        using ReduceByKeyKernel              = ReduceByKey::ReduceByKeyKernel;

        // init
        auto init_key                   = get_type_and_op_desc<KeyType, ValueType>();
        auto ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        if(ms_scan_tile_state_init_it == ms_scan_tile_state_init_map.end())
        {
            auto shader = ReduceByKey().compile_scan_tile_state_init(m_device);
            if(!shader)
            {
                return -1;  // shader create failed
            }
            ms_scan_tile_state_init_map.try_emplace(init_key, std::move(shader));
            ms_scan_tile_state_init_it = ms_scan_tile_state_init_map.find(init_key);
        }
        auto ms_scan_tile_state_init_ptr =
            reinterpret_cast<ReduceByKeyTileStateInitKernel*>(&(*ms_scan_tile_state_init_it->second));
        cmdlist << (*ms_scan_tile_state_init_ptr)(tile_states, num_tiles).dispatch(num_tiles * m_block_size);
        // reduce by key
        auto key                 = get_type_and_op_desc<KeyType, ValueType>(reduce_op);
        auto ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        if(ms_reduce_by_key_it == ms_reduce_by_key_map.end())
        {
            auto shader = ReduceByKey().compile(m_device, m_shared_mem_size, reduce_op);
            ms_reduce_by_key_map.try_emplace(key, std::move(shader));
            ms_reduce_by_key_it = ms_reduce_by_key_map.find(key);
        }
        auto ms_reduce_by_key_ptr = reinterpret_cast<ReduceByKeyKernel*>(&(*ms_reduce_by_key_it->second));

        cmdlist << (*ms_reduce_by_key_ptr)(
                       tile_states, tile_partial, tile_inclusive, keys_in, values_in, unique_out, aggregated_out, num_runs_out, num_items)
                       .dispatch(m_block_size * num_tiles);
        return 0;
    };


    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_single_reduce_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_transform_reduce_map;
    // for arg reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_construct_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_assign_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_arg_extremum_map;
    // for reduce by key
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_by_key_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_scan_tile_state_init_map;

    template <typename T>
    static void write_scalar(CommandList& cmdlist, BufferView<T> output, T value)
    {
        auto ptr = luisa::new_with_allocator<T>(value);
        cmdlist << output.copy_from(luisa::span<T>{ptr, 1u});
        cmdlist.add_dtor_callback([ptr] { luisa::delete_with_allocator(ptr); });
    }
};
}  // namespace luisa::parallel_primitive
