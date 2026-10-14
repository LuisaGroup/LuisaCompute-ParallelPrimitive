#include "cub_misc.h"

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_find.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/stream_ref>

static_assert(CUB_VERSION == 300500, "This benchmark requires CCCL/CUB 3.5.0.");

namespace {
struct Transform {
    __host__ __device__ int operator()(int value) const { return value * 3 + 1; }
};
struct Even {
    __host__ __device__ bool operator()(int value) const { return value % 2 == 0; }
};
struct Match {
    __host__ __device__ bool operator()(int value) const { return value == 777777; }
};
}
extern "C" int cub_misc_version() { return CUB_VERSION; }

extern "C" cudaError_t cub_misc_dispatch(CubMiscOp operation, void *temp, size_t *bytes,
                                         const CubMiscArgs *args, cudaStream_t stream) {
    const auto &a = *args;
    switch (operation) {
        case CubMiscOp::transform:
            return cub::DeviceTransform::Transform(temp, *bytes, a.input, a.output, a.num_items,
                                                   Transform{}, cuda::stream_ref{stream});
        case CubMiscOp::find:
            return cub::DeviceFind::FindIf(temp, *bytes, a.input, a.count, Match{}, a.num_items,
                                          cuda::stream_ref{stream});
        case CubMiscOp::select:
            return cub::DeviceSelect::If(temp, *bytes, a.input, a.output, a.count, a.num_items,
                                        Even{}, cuda::stream_ref{stream});
        case CubMiscOp::unique:
            return cub::DeviceSelect::Unique(temp, *bytes, a.input, a.output, a.count, a.num_items,
                                            cuda::stream_ref{stream});
        case CubMiscOp::partition:
            return cub::DevicePartition::Flagged(temp, *bytes, a.input, a.flags, a.output, a.count,
                                                a.num_items, cuda::stream_ref{stream});
        case CubMiscOp::merge: {
            const int left = a.num_items / 2;
            return cub::DeviceMerge::MergeKeys(temp, *bytes, a.input, left, a.second_input,
                                              a.num_items - left, a.output, cuda::std::less<>{}, stream);
        }
        case CubMiscOp::histogram:
            return cub::DeviceHistogram::HistogramEven(temp, *bytes, a.input, a.aux_output,
                                                       257, 0, 256, a.num_items, stream);
        case CubMiscOp::run_length_encode:
            return cub::DeviceRunLengthEncode::Encode(temp, *bytes, a.input, a.output,
                                                       a.aux_output, a.count, a.num_items, stream);
        case CubMiscOp::segment_sum:
            return cub::DeviceSegmentedReduce::Sum(temp, *bytes, a.input, a.output, a.num_segments,
                                                  a.offsets, a.offsets + 1, stream);
        case CubMiscOp::adjacent_difference:
            return cub::DeviceAdjacentDifference::SubtractLeftCopy(temp, *bytes, a.input,
                                                                   a.output, a.num_items,
                                                                   cuda::std::minus<>{}, stream);
    }
    return cudaErrorInvalidValue;
}
