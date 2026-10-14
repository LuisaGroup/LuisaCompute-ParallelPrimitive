/*
 * @Author: Ligo 
 * @Date: 2025-09-29 11:30:37 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-06 17:58:06
 */
#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/utils.h>
#include <lcpp/runtime/core.h>
#include <lcpp/warp/details/warp_scan_smem.h>
#include <lcpp/warp/details/warp_scan_shlf.h>

namespace luisa::parallel_primitive
{
namespace details
{
    template <typename ScanOp, typename T>
    concept WarpScanCallable =
        requires(ScanOp op, const compute::Var<T>& lhs, const compute::Var<T>& rhs) { op(lhs, rhs); };
}  // namespace details

enum class WarpScanAlgorithm
{
    WARP_SHUFFLE       = 0,
    WARP_SHARED_MEMORY = 1
};
template <typename Type4Byte, size_t WARP_SIZE = 32, WarpScanAlgorithm WarpScanMethod = WarpScanAlgorithm::WARP_SHUFFLE>
class WarpScan : public LuisaModule
{
  public:
    WarpScan()
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            m_shared_mem = new SmemType<Type4Byte>{1024u};
        };
    }
    WarpScan(SmemTypePtr<Type4Byte> shared_mem)
        : m_shared_mem(shared_mem)
    {
    }
    ~WarpScan() = default;

  public:
    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output, ScanOp scan_op)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.ExclusiveScan(
                thread_data, exclusive_output, scan_op);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.ExclusiveScan(thread_data, exclusive_output, scan_op);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.ExclusiveScan(
                thread_data, exclusive_output, warp_aggregate, scan_op);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.ExclusiveScan(
                thread_data, exclusive_output, warp_aggregate, scan_op);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.ExclusiveScan(
                thread_data, exclusive_output, scan_op, initial_value);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.ExclusiveScan(
                thread_data, exclusive_output, scan_op, initial_value);
        }
    }

    template <typename ScanOp>
    void ExclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       exclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.ExclusiveScan(
                thread_data, exclusive_output, warp_aggregate, scan_op, initial_value);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.ExclusiveScan(
                thread_data, exclusive_output, warp_aggregate, scan_op, initial_value);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_in, Var<Type4Byte>& inclusive_output, ScanOp scan_op)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.InclusiveScan(thread_in, inclusive_output, scan_op);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.InclusiveScan(thread_in, inclusive_output, scan_op);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.InclusiveScan(
                thread_data, inclusive_output, warp_aggregate, scan_op);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.InclusiveScan(
                thread_data, inclusive_output, warp_aggregate, scan_op);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.InclusiveScan(
                thread_data, inclusive_output, scan_op, initial_value);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.InclusiveScan(
                thread_data, inclusive_output, scan_op, initial_value);
        }
    }

    template <typename ScanOp>
    void InclusiveScan(const Var<Type4Byte>& thread_data,
                       Var<Type4Byte>&       inclusive_output,
                       Var<Type4Byte>&       warp_aggregate,
                       ScanOp                scan_op,
                       const Var<Type4Byte>& initial_value)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.InclusiveScan(
                thread_data, inclusive_output, warp_aggregate, scan_op, initial_value);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.InclusiveScan(
                thread_data, inclusive_output, warp_aggregate, scan_op, initial_value);
        }
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void InclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       inclusive_output,
                              ScanOp                scan_op,
                              compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = raw_output;
        };
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void InclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       inclusive_output,
                              const Var<Type4Byte>& initial_value,
                              ScanOp                scan_op,
                              compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = scan_op(initial_value, raw_output);
        };
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void InclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       inclusive_output,
                              ScanOp                scan_op,
                              compute::UInt         valid_items,
                              Var<Type4Byte>&       warp_aggregate)
    {
        Var<Type4Byte> raw_output;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = raw_output;
        };
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void InclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       inclusive_output,
                              const Var<Type4Byte>& initial_value,
                              ScanOp                scan_op,
                              compute::UInt         valid_items,
                              Var<Type4Byte>&       warp_aggregate)
    {
        Var<Type4Byte> raw_output;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = scan_op(initial_value, raw_output);
        };
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ExclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       exclusive_output,
                              ScanOp                scan_op,
                              compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        assign_partial_exclusive(raw_output, exclusive_output, scan_op, valid_items);
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ExclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       exclusive_output,
                              const Var<Type4Byte>& initial_value,
                              ScanOp                scan_op,
                              compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        assign_partial_exclusive(raw_output, exclusive_output, initial_value, scan_op, valid_items);
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ExclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       exclusive_output,
                              ScanOp                scan_op,
                              compute::UInt         valid_items,
                              Var<Type4Byte>&       warp_aggregate)
    {
        Var<Type4Byte> raw_output;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        assign_partial_exclusive(raw_output, exclusive_output, scan_op, valid_items);
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ExclusiveScanPartial(const Var<Type4Byte>& input,
                              Var<Type4Byte>&       exclusive_output,
                              const Var<Type4Byte>& initial_value,
                              ScanOp                scan_op,
                              compute::UInt         valid_items,
                              Var<Type4Byte>&       warp_aggregate)
    {
        Var<Type4Byte> raw_output;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        assign_partial_exclusive(raw_output, exclusive_output, initial_value, scan_op, valid_items);
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ScanPartial(const Var<Type4Byte>& input,
                     Var<Type4Byte>&       inclusive_output,
                     Var<Type4Byte>&       exclusive_output,
                     ScanOp                scan_op,
                     compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = raw_output;
        };
        assign_partial_exclusive(raw_output, exclusive_output, scan_op, valid_items);
    }

    template <typename ScanOp>
        requires details::WarpScanCallable<ScanOp, Type4Byte>
    void ScanPartial(const Var<Type4Byte>& input,
                     Var<Type4Byte>&       inclusive_output,
                     Var<Type4Byte>&       exclusive_output,
                     const Var<Type4Byte>& initial_value,
                     ScanOp                scan_op,
                     compute::UInt         valid_items)
    {
        Var<Type4Byte> raw_output;
        Var<Type4Byte> warp_aggregate;
        partial_inclusive(input, raw_output, warp_aggregate, scan_op, valid_items);
        $if(logical_lane() < valid_items)
        {
            inclusive_output = scan_op(initial_value, raw_output);
        };
        assign_partial_exclusive(raw_output, exclusive_output, initial_value, scan_op, valid_items);
    }

    // sum
    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output)
    {
        ExclusiveScan(
            thread_data,
            exclusive_output,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    void ExclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& exclusive_output, Var<Type4Byte>& warp_aggregate)
    {
        ExclusiveScan(
            thread_data,
            exclusive_output,
            warp_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_output)
    {
        InclusiveScan(
            thread_data,
            inclusive_output,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    void InclusiveSum(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_output, Var<Type4Byte>& warp_aggregate)
    {
        InclusiveScan(
            thread_data,
            inclusive_output,
            warp_aggregate,
            [](const Var<Type4Byte>& a, const Var<Type4Byte>& b) { return a + b; },
            Type4Byte(0));
    }

    [[nodiscard]] Var<Type4Byte> Broadcast(const Var<Type4Byte>& input, compute::UInt source_lane)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            return details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.Broadcast(input, source_lane);
        }
        else
        {
            compute::UInt lane_id   = compute::warp_lane_id();
            compute::UInt warp_base = lane_id / compute::UInt(WARP_SIZE) * compute::UInt(WARP_SIZE);
            return WarpReadLane(input, warp_base + source_lane);
        }
    }

    template <typename ScanOp>
    void Scan(const Var<Type4Byte>& thread_data, Var<Type4Byte>& inclusive_output, Var<Type4Byte>& exclusive_output, ScanOp scan_op)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.Scan(
                thread_data, inclusive_output, exclusive_output, scan_op);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.Scan(thread_data, inclusive_output, exclusive_output, scan_op);
        }
    }

    template <typename ScanOp>
    void Scan(const Var<Type4Byte>& thread_data,
              Var<Type4Byte>&       inclusive_output,
              Var<Type4Byte>&       exclusive_output,
              ScanOp                scan_op,
              const Var<Type4Byte>& initial_value)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.Scan(
                thread_data, inclusive_output, exclusive_output, scan_op, initial_value);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.Scan(
                thread_data, inclusive_output, exclusive_output, scan_op, initial_value);
        }
    }

  private:
    template <typename ScanOp>
    void partial_inclusive(const Var<Type4Byte>& input,
                           Var<Type4Byte>&       inclusive_output,
                           Var<Type4Byte>&       warp_aggregate,
                           ScanOp                scan_op,
                           compute::UInt         valid_items)
    {
        if constexpr(WarpScanMethod == WarpScanAlgorithm::WARP_SHARED_MEMORY)
        {
            details::WarpScanSmem<Type4Byte, WARP_SIZE>{m_shared_mem}.InclusiveScanPartial(
                input, inclusive_output, warp_aggregate, scan_op, valid_items);
        }
        else
        {
            details::WarpScanShfl<Type4Byte, WARP_SIZE>{}.InclusiveScanPartial(
                input, inclusive_output, warp_aggregate, scan_op, valid_items);
        }
    }

    template <typename ScanOp>
    static void assign_partial_exclusive(const Var<Type4Byte>& inclusive_output,
                                         Var<Type4Byte>&       exclusive_output,
                                         ScanOp,
                                         compute::UInt valid_items)
    {
        compute::UInt  lane     = logical_lane();
        Var<Type4Byte> previous = previous_lane(inclusive_output);
        $if(lane < valid_items & lane > 0u)
        {
            exclusive_output = previous;
        };
    }

    template <typename ScanOp>
    static void assign_partial_exclusive(const Var<Type4Byte>& inclusive_output,
                                         Var<Type4Byte>&       exclusive_output,
                                         const Var<Type4Byte>& initial_value,
                                         ScanOp                scan_op,
                                         compute::UInt         valid_items)
    {
        compute::UInt  lane     = logical_lane();
        Var<Type4Byte> previous = previous_lane(inclusive_output);
        $if(lane < valid_items)
        {
            $if(lane == 0u)
            {
                exclusive_output = initial_value;
            }
            $else
            {
                exclusive_output = scan_op(initial_value, previous);
            };
        };
    }

    static Var<Type4Byte> previous_lane(const Var<Type4Byte>& inclusive_output)
    {
        compute::UInt physical_lane = compute::warp_lane_id();
        compute::UInt base = physical_lane / compute::UInt(WARP_SIZE) * compute::UInt(WARP_SIZE);
        compute::UInt source_lane = compute::select(base, physical_lane - 1u, logical_lane() > 0u);
        return WarpReadLane(inclusive_output, source_lane);
    }

    static compute::UInt logical_lane()
    {
        return compute::warp_lane_id() % compute::UInt(WARP_SIZE);
    }

    SmemTypePtr<Type4Byte> m_shared_mem = nullptr;
};
}  // namespace luisa::parallel_primitive
