# RLE and SegmentReduce CUDA optimization

Date: 2026-10-19. GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
Driver: 590.48.01, CUDA driver API 13010. CUDA toolkit/NVRTC: 12.8.
CCCL/CUB: 3.5.0, version code 300500, compiled for sm_120.
LCPP host compiler: GCC 13.3.0, Release/C++20.

Baseline: main at `80f8f3248094c4a2ab4501bdcd9b10f682d9804f`.
Both baseline and optimized executables use LuisaCompute
`79abe365f4dd6a97d04969cedb87c97127eb2c58` and the same CUB build settings.
The older local SDK lacked the interop API required by this main branch;
the compatible SDK was built separately without modifying the original checkout.

## Result

SegmentReduce's offset-based int32 Sum is approximately equal to CUB at 4M
items for segment sizes 256, 4096 and 16384, and about 9% faster for size 32.
The 4M RLE run16 case is also approximately equal to CUB.

RLE is not generally at parity. All-unique input improves from 24.528 to
22.488 us at 4M and from 12.264 to 9.072 us at 1M. At 1M, run16 still has
about 17% median overhead, and all-unique input about 20%. Long-run RLE is
approximately unchanged; its 4M median increases by 0.104 us, within overlapping
run-to-run ranges. No improvement is claimed for that distribution.

These are GPU microbenchmarks, not application end-to-end measurements or
claims about other GPUs, types, backends or input distributions.

## Measurement

For each size and implementation version, five independent processes take
60 alternating LCPP/CUB samples per case. Baseline and optimized processes
also alternate order. Both libraries use the same context, stream and input.
Each CUDA-event window covers a graph batch of four complete operations;
times are divided by four. All initialization and auxiliary kernels are included.
JIT compilation, allocation, H2D transfer, graph capture and 16 graph warmups
finish before sampling.

Every process validates CPU/LCPP/CUB counts, values and lengths before and
after timing. All 140 case executions in the 20-process comparison pass.
Desktop GPU activity was present and was not stopped. Raw sample min/max
outliers are retained. No profiler runs contribute to the tables below.

Times are medians of the five process medians. CUB time is from optimized
processes. Ratios are medians of the five per-process LCPP/CUB median ratios,
not ratios of the separately summarized time columns. Lower ratios are better.

### 4,194,304 items

| Operation | Before LCPP (us) | After LCPP (us) | CUB (us) | Before LCPP/CUB | After LCPP/CUB | After range, 5 runs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RLE, run1 | 24.528 | 22.488 | 19.512 | 1.2619x | 1.1566x | 1.1469-1.1661x |
| RLE, run16 | 12.776 | 12.248 | 12.216 | 1.0458x | 1.0026x | 0.9993-1.0033x |
| RLE, run4096 | 11.696 | 11.800 | 11.224 | 1.0436x | 1.0478x | 1.0418-1.0843x |
| SegmentReduce.Sum, seg32 | 83.360 | 74.688 | 82.456 | 1.0086x | 0.9058x | 0.9023-0.9061x |
| SegmentReduce.Sum, seg256 | 11.704 | 10.744 | 10.696 | 1.0951x | 1.0030x | 0.9963-1.0060x |
| SegmentReduce.Sum, seg4096 | 4.576 | 4.064 | 4.048 | 1.1310x | 1.0040x | 0.9960-1.0099x |
| SegmentReduce.Sum, seg16384 | 4.536 | 4.072 | 4.072 | 1.1250x | 1.0000x | 0.9942-1.0020x |

### 1,048,576 items

| Operation | Before LCPP (us) | After LCPP (us) | CUB (us) | Before LCPP/CUB | After LCPP/CUB | After range, 5 runs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RLE, run1 | 12.264 | 9.072 | 7.608 | 1.6120x | 1.2040x | 1.1420-1.2051x |
| RLE, run16 | 6.784 | 6.504 | 5.560 | 1.2028x | 1.1749x | 1.0899-1.1842x |
| RLE, run4096 | 6.152 | 6.112 | 5.560 | 1.1017x | 1.1022x | 1.0926-1.1044x |
| SegmentReduce.Sum, seg32 | 21.944 | 19.688 | 21.856 | 1.0103x | 0.9050x | 0.8895-0.9060x |
| SegmentReduce.Sum, seg256 | 4.040 | 3.512 | 3.560 | 1.1323x | 0.9865x | 0.9757-1.0069x |
| SegmentReduce.Sum, seg4096 | 2.480 | 1.984 | 2.280 | 1.0915x | 0.8759x | 0.8000-0.9656x |
| SegmentReduce.Sum, seg16384 | 4.024 | 2.488 | 2.536 | 1.5868x | 0.9811x | 0.9662-0.9968x |

The 1M seg4096 CUB timings vary substantially between processes. Use the
4M results as the stronger evidence for parity rather than treating the
exact 1M speedup as a stable architecture-wide result.

## Implementation

- RLE retains the existing three-kernel pipeline and unchanged compaction
  algorithm. Its default compaction policy changes from 384x15 to 256x15;
  explicit block sizes and nondefault item counts remain honored.
- Aligned contiguous offset/length views use uint4 loads and stores in the
  lengths kernel, with scalar handling of the final one to three runs.
  Small inputs keep at least three logical tail lanes active. Unaligned
  views retain the original scalar implementation. Temporary-storage sizing,
  wide-key fallback, in-place fallback and NonTrivialRuns are unchanged.
- SegmentReduce's CUDA int32/uint32 Sum uses warp_active_sum for per-warp
  reduction and a parallel first-warp reduction of warp totals. Invalid
  partial-segment lanes are zeroed. The SDK provides the pre-SM80 fallback.
- Default CUDA integer Sum consumes 16 items per thread instead of four,
  reducing the tile-loop count for long segments. Custom policies, other
  operators/types/backends and the fixed-size small-segment path retain
  their existing behavior.

Offset-only key reload, fused interior lengths, sparse-tile bypass and
warp-per-tile correction experiments were discarded because they regressed
other RLE distributions. No changes to the shared compaction implementation
are retained.

## Profiling evidence

Nsight Systems traces are separate from benchmark sampling. Each entry below
is the median of 85 kernel launches, including warmup and sampled graph nodes,
in one 4M process. These medians exclude graph gaps and must not be summed
to reproduce an event-window median.

| RLE distribution | Compact before/after (us) | Lengths before/after (us) |
| --- | ---: | ---: |
| run1 | 16.064 / 14.944 | 7.392 / 6.560 |
| run16 | 10.720 / 9.984 | 1.248 / 1.248 |
| run4096 | 9.728 / 9.920 | 0.864 / 1.024 |

The dense-output lengths pass improves, while long-run kernels do not.
RLE still launches three kernels versus CUB's two; SegmentReduce remains one
kernel on both sides. This is consistent with residual small-workload RLE
overhead, but does not isolate all remaining stalls or bandwidth limits.

CUDA driver function attributes and the occupancy calculator, outside timing:

| Kernel | Block/grid at 4M | Registers/thread | Static shared bytes | Maximum blocks/SM |
| --- | --- | ---: | ---: | ---: |
| RLE compact, before | 384/729 | 61 | 46144 | 2 |
| RLE compact, after | 256/1093 | 73 | 30768 | 3 |
| CUB RLE main | 256/1171 | 64 | 28688 | 3 |
| RLE lengths, before | 256/729 | 16 | 0 | 6 |
| RLE lengths, after | 256/1093 | 22 | 0 | 6 |
| SegmentReduce seg256, before and after | 256/16384 | 40 | 64 | 6 |
| CUB SegmentReduce seg256 | 256/16384 | 40 | 256 | 6 |

These are compiled resource limits, not achieved occupancy. Hardware counters
were not collected; RmProfilingAdminOnly remains 1. No current Nsight Compute
DRAM bandwidth, achieved occupancy or warp-stall attribution is claimed.

## Verification

All six focused targets were rebuilt and run on CUDA:

| Target | Result |
| --- | --- |
| device_run_length_encode_test | 979 assertions, 6 tests |
| device_segment_reduce | 126 assertions, 9 tests |
| device_select_test | 209 assertions, 7 tests |
| device_partition_test | 96 assertions, 6 tests |
| device_reduce_test | 93 assertions, 14 tests |
| pr6_review_regression_test | 136 checks, 0 failures |

Total: 1503 assertions plus 136 PR6 checks. New tests cover tiny packet tails,
old/new tile boundaries, poisoned scratch, mixed dense/sparse runs, uint/float
keys, guarded subviews, custom block policies, ragged/empty segments, nonzero
initial values, uint/float/ulong fallback and fixed-size full tiles/tails.

RLE and SegmentReduce each pass Compute Sanitizer memcheck, synccheck and
racecheck: zero errors, zero reported race hazards/warnings.

Additional CPU/LCPP/CUB checks pass all 17 miscellaneous benchmark cases at
1M and 4M. Seven targeted cases also pass at 1,048,613, 4,194,317, 4 and 6 items.
These extra processes are correctness smoke checks, not additional statistical
performance claims. git diff --check and the benchmark helper syntax check pass.

Full repository build/test, other backends and application end-to-end timing
were not run.

## Reproduce

Use the SDK/compiler versions above. The tested XMake configuration disables
GUI, Python, OSL and non-CUDA backends, with no LLVM/TVM stack:

```sh
xmake f -y -m release --cc=gcc-13 --cxx=g++-13 --ld=g++-13 --sh=g++-13 \
  --lc_enable_gui=false --lc_enable_imgui=false --lc_enable_py=false \
  --lc_enable_osl=false --lc_vk_backend=false --lc_dx_backend=false \
  --lc_metal_backend=false --lc_cuda_backend=true --lc_enable_tests=false \
  --lc_tile_tirx_bridge=false --lc_tvm_stack=false
xmake build -j2 lc-backend-cuda
xmake build -j2 lc-nvrtc
xmake project -k compile_commands build
node benchmarks/rebuild_cuda_cub_misc.cjs
build/linux/x86_64/release/cuda_cub_misc 1048576 60 rle-segmentreduce
build/linux/x86_64/release/cuda_cub_misc 4194304 60 rle-segmentreduce
xmake build -j2 device_run_length_encode_test
xmake build -j2 device_segment_reduce
build/linux/x86_64/release/device_run_length_encode_test --backend=cuda
build/linux/x86_64/release/device_segment_reduce --backend=cuda
```

The helper's `--lcpp DIR` selects frozen baseline headers without changing the
active checkout. `--reuse-cub 1` only reuses an object whose recorded paths,
compiler version, flags and source fingerprint match. Default behavior rebuilds
the CUB object. Omit the case filter to run all 17 cases.

## Saved evidence

Local evidence root:
`/home/ligo/.codex/worktrees/rle-segmentreduce-perf/lc_parallel_primitive/build/rle-segmentreduce-20261019`.
The build directory is ignored; these artifacts are local, not repository files.

- `environment.json`: source/toolchain/environment snapshot.
- `validated_sweep/benchmark_results.json`: executable hashes, five-process
  summaries and all parsed observations. Raw logs are alongside it.
- `measure.cjs`: local alternating before/after repeat-run driver; it accepts
  LCPP_RESULTS_DIR to preserve independent measurement sets.
- `validation_results.json`, `validate.cjs` and `*_correctness.log`: validation
  commands and outputs. `rle_*check.log` and `segment_*check.log` contain
  sanitizer summaries; `*_stdout.log` contains the associated test output.
- `validated_baseline_rle_trace.nsys-rep`, `validated_after_rle_trace.nsys-rep`,
  corresponding SQLite exports and `*.kernel_summary.json`: kernel timelines.
- `trace_kernels.cjs`: local SQLite census, with explicit launch-count checks.
- `validated_baseline_launch_4m.log`, `validated_after_launch_4m.log`: driver
  function attributes and theoretical block residency.

Experimental logs elsewhere in this directory are not the retained result.
