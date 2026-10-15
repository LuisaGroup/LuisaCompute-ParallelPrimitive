# Long segmented scan follow-up, 2026-10-15

Measured code: `c20ef138809873ad1db2e7b9fb871d343e84c0c4`.
Before: `a38a11bda15a40c36b0edc64338b23ae6652e5d7`, the previous cooperative implementation.
LuisaCompute: `fac75f60bc1f8925559852e002dc261e47dd1f4d`.

The long int32 sum now performs one warp scan of per-thread register aggregates per tile, with padded shared-memory transposes for coalesced input and output.
On the measured RTX 5090, 4096-element segments reach CUB's performance and 16384-element segments are faster than CUB. Short segments retain their previous performance.

## Measurement

Linux, RTX 5090, driver 590.48.01, CUDA toolkit 12.8, GCC 13.3, CUB 3.5.0 (`300500`), native `sm_120`.
An existing Isaac Sim process remained active. These are shared-device, hot-buffer measurements.

Each configuration has five independent processes per version, with 31 alternating LCPP/CUB sample pairs in each process. Before/after order alternates between rounds.
CUDA events time graph replay containing four public operations and divide by four; both libraries use the same CUDA context and stream.
Allocation, compilation, transfers, warmup, and CPU output checks are outside timing.
The expanded baseline harness uses the same protocol and frozen headers from the before commit; only the accepted segment lengths and policy labels changed.

Times are the median of five process medians. Speedup and After/CUB are the median of the five corresponding per-process ratios, so they need not equal the ratio of the table's median times.

| int32 elements | Items/segment | Before (us) | After (us) | CUB (us) | Speedup | After/CUB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 256 | 3.568 | 3.576 | 4.080 | 0.998x | 0.873x |
| 1,048,576 | 4096 | 6.632 | 3.536 | 5.048 | 1.876x | 0.705x |
| 1,048,576 | 16384 | 20.480 | 6.728 | 13.304 | 3.044x | 0.506x |
| 4,194,304 | 256 | 8.152 | 8.160 | 11.280 | 0.999x | 0.723x |
| 4,194,304 | 4096 | 13.816 | 7.600 | 7.704 | 1.818x | 0.992x |
| 4,194,304 | 16384 | 21.464 | 9.208 | 15.376 | 2.331x | 0.599x |

For 4M/4096, the five after medians range from 7.496 to 7.664 us; CUB ranges from 7.560 to 7.752 us. The difference is within variation, so this is parity rather than a demonstrated win.

## Host submission and completion

Both sides enqueue 100 operations and perform one final synchronization, dividing elapsed host time by 100. These results include submission and completion overhead.
Vulkan and CUB use UUID-matched devices; segmented scan runs through the native Vulkan DSL.

| Backend | Items/segment, 4M int32 | Before (us) | After (us) | CUB (us) | Speedup | After/CUB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 4096 | 14.404 | 8.251 | 8.237 | 1.745x | 1.002x |
| Vulkan | 256 | 17.284 | 16.802 | 12.910 | 1.029x | 1.303x |
| Vulkan | 4096 | 22.913 | 16.790 | 8.842 | 1.384x | 1.899x |
| Vulkan | 16384 | 36.037 | 20.476 | 17.043 | 1.738x | 1.203x |

Native Vulkan improves on long segments but remains slower than CUB in these host measurements. CUDA parity does not establish Vulkan parity.

## Implementation and resource checks

- The default long kernel uses 256 threads x 8 items, a 2048-element tile, and 8480 bytes of shared memory. Each thread reduces and expands its eight contiguous values in registers; only the thread aggregate participates in the warp scan.
- Input/output transposes use `physical = logical + logical / 32`. Consecutive 32-element shared rows receive a one-word pad. An unpadded register implementation remained around 14.7 us at 4M/4096; padding and the selected tile size removed that regression. Hardware bank-conflict counters were unavailable, so timing alone does not quantify their contribution.
- CUDA uses two warp fences and two uniform block barriers per tile. Vulkan uses uniform block barriers for both transposes, totaling four per tile. The block barriers also protect shared-memory reuse and exact in-place operation.
- Blocks above 256 threads or shared allocations exceeding the device limit use the existing arbitrary-length warp implementation. A 1024-thread test reproduced `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` before the register-headroom fallback and passes after it.
- Short sums, general operators, wider values, output-offset semantics, zero scratch, dispatch splitting, and ordinary scan dispatch remain unchanged. Parallel floating-point sums may round differently from serial evaluation.

Independent Nsight Systems traces, excluded from timing tables, confirm these 4M/4096 resources:

| Kernel | Grid x block | Registers/thread | Static shared (bytes) | Local bytes/thread | Average kernel time (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LCPP before | 1024 x 256 | 48 | 260 | 0 | 13.854 |
| LCPP after | 1024 x 256 | 64 | 8480 | 0 | 7.073 |
| CUB after comparison | 1024 x 128 | 56 | 6176 | 0 | 7.169 |

The before profile is retained from the first pass. These resource fields describe allocations, not achieved occupancy or measured DRAM traffic.

## Validation

- Final `scan_optimization_regression_test`: CUDA 520 checks, Vulkan 507 checks, required ordinary Vulkan CUDA interop 520 checks; zero failures.
- CUDA memcheck: all 520 checks, zero errors. Segmented racecheck: 175 checks, zero hazards, errors, or warnings.
- Existing segmented scan suite rebuilt against the final header: 155 assertions on each of CUDA and Vulkan, all passed.
- Added ragged lengths around 8192 and 16384, a 32769-element multi-tile segment, and the 1024-thread fallback case. Existing coverage retains seeds, inclusive/exclusive modes, exact in-place operation, independent output gaps, empty segments, float/double, non-associative fallback, and dispatch split boundaries.
- All 100 before/after benchmark processes passed CPU output checks. Their header, executable, and runtime hashes were verified against the frozen baseline and measured code.

Full commands, raw paired samples, telemetry, frozen tuning experiments, hashes, sanitizer logs, and Nsight reports are retained locally in `build/long-segment-optimization/`. Other GPU architectures and Windows/DX/Metal were not measured.
