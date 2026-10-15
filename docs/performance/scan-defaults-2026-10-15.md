# Default and segmented scan performance, 2026-10-15

These tables describe the first optimization pass. The subsequent [long segmented scan follow-up](segmented-scan-long-2026-10-15.md) reaches CUB parity at 4M/4096 and improves larger segments further.

Measured library commit: `ae0d0f1c66d8060aea0eee49c6d799bad2a39dd8`.
Frozen before version: `ee963595708fa7029533d81fa557afc4f83a202e`.
LuisaCompute: `fac75f60bc1f8925559852e002dc261e47dd1f4d`, including merged PR #279.

Linux, RTX 5090, driver 590.48.01, CUDA toolkit 12.8, GCC 13.3, CUB 3.5.0 (`300500`), native `sm_120`.
An existing Isaac Sim process remained active throughout. These are shared-device, hot-buffer results.

## CUDA GPU graph results

4,194,304 int32 elements. Each side captures four public operations on the same CUDA context/stream; events time graph replay and the result is divided by four.
Allocation, compilation, transfers, warmup, and CPU reference checks are outside the measured interval.
Each configuration runs in five independent processes per version, with 31 alternating LCPP/CUB sample pairs per process. Before/after order alternates between rounds.
Times below are the median of five per-process medians. Speedup is the median of the five corresponding before/after ratios.

| Operation | Before (us) | After (us) | CUB (us) | Speedup | After/CUB |
| --- | ---: | ---: | ---: | ---: | ---: |
| InclusiveSum | 45.416 | 10.160 | 10.160 | 4.45x | 1.004x |
| InclusiveSegmentedSum, 256 items/segment | 53.176 | 8.120 | 11.232 | 6.55x | 0.720x |
| InclusiveSegmentedSum, 4096 items/segment | 996.416 | 14.192 | 7.584 | 69.78x | 1.873x |

The ordinary scan is within run-to-run variation of CUB. Short segments are faster in this workload. At this first-pass commit, long segments take about 1.87 times CUB despite the large improvement.
The long-segment baseline varied from 886.824 to 1093.584 us; the optimized version varied from 14.176 to 14.296 us.

## Host submission and completion

Both sides time 100 enqueues plus one final stream synchronization and divide by 100. This includes submission and completion overhead, and is not a pure GPU kernel measurement.
Vulkan and CUB use UUID-matched devices. Required interop rejects fallback.

| Operation | Backend | Before (us) | After (us) | CUB (us) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| InclusiveSum | CUDA | 63.119 | 15.303 | 12.133 | 4.13x |
| InclusiveSum | Vulkan DSL | 95.763 | 96.922 | 11.542 | 1.00x |
| Segmented sum, 256/segment | Vulkan DSL | 80.334 | 27.534 | 13.824 | 2.95x |
| Segmented sum, 4096/segment | Vulkan DSL | 1080.099 | 34.748 | 10.597 | 31.51x |
| InclusiveSum | Vulkan CUDA interop | 76.354 | 28.728 | 12.062 | 2.71x |

Native Vulkan ordinary scan retains its portable policy; its performance did not improve in this change. SegmentedScan uses the native DSL on both backends and has no CUDA interop implementation.

## Implementation

- Default ordinary 4-byte CUDA/interop scans select 320 threads x 25 items, warp-transpose I/O, and warp-local synchronization. The old native CUDA default already used packed tile state; packed state alone is not a new improvement.
- Native Vulkan ordinary scans, wider values, transform/adjacent scans, scan-by-key, and explicitly customized policies keep their existing dispatch.
- Built-in int32/uint32/float sums choose one warp per segment for short average input views, or one block per segment for larger averages. Both handle arbitrary ragged lengths and exact in-place operation. The average is a performance heuristic, not a maximum-length requirement.
- The default warp strategy uses no shared memory or block barriers. The block strategy uses 260 bytes of shared warp totals and three uniform barriers per 2048-element tile.
- General segmented operators and wider types retain serial left-to-right evaluation. Floating-point sum association changes can change rounding.
- Dispatches split at 65535 workgroups using a segment-base argument and complete offset views. Vulkan uses its default subgroup width without requiring optional subgroup-size-control.
- BlockScan shuffle initial-value overloads now compose the seed exactly once while preserving the raw block aggregate. The frozen old version reproduced eight initial-value failures in the new ordinary tests.

## Validation and profiling

- `scan_optimization_regression_test`: 507 checks each on native CUDA and Vulkan; 520 checks on required ordinary Vulkan CUDA interop.
- Coverage includes int/uint/float/double, both tile boundaries, guarded exact-size scratch, initial values, noncommutative operations, in-place scans, independent output offsets, empty segments, ragged long outliers under the short-segment heuristic, custom legal block sizes, and both dispatch split boundaries.
- Existing scan and segmented scan suites pass on CUDA and Vulkan. PR #6 regressions pass 136 native checks per backend and 294 required-interop checks.
- CUDA Compute Sanitizer: 507 checks with zero memcheck errors; segmented racecheck: 162 checks, zero hazards/errors/warnings.
- All 110 before/after comparison processes passed CPU output checks, across 1M/4M graph workloads and the host modes above.
- Separate Nsight traces confirm ordinary scan changes from 1093 blocks x 256 threads to 525 x 320; short segmented scan from 64 x 256 to 2048 x 256; long segmented scan from 4 x 256 to 1024 x 256. The former segmented implementation assigned one thread to each segment.
- Hardware performance counters and CPU instruction sampling were unavailable. No DRAM-bandwidth or achieved-occupancy claim is made. Other GPU architectures and Windows/DX/Metal were not measured.

Full commands, binary/runtime/header hashes, raw paired samples, per-run telemetry, CPU references, and Nsight reports are retained in the local `build/scan-optimization/` evidence directory. Profile runs are excluded from the timing tables.
