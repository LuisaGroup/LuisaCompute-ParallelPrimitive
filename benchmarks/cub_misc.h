#pragma once

#include <cuda_runtime_api.h>

enum class CubMiscOp : int {
    transform,
    find,
    select,
    unique,
    partition,
    merge,
    histogram,
    run_length_encode,
    segment_sum,
    adjacent_difference
};

struct CubMiscArgs {
    const int *input;
    const int *second_input;
    const unsigned *flags;
    const unsigned *offsets;
    int *output;
    unsigned *count;
    unsigned *aux_output;
    int num_items;
    int num_segments;
};

extern "C" int cub_misc_version();
extern "C" cudaError_t cub_misc_dispatch(CubMiscOp operation, void *temp, size_t *bytes,
                                         const CubMiscArgs *args, cudaStream_t stream);
