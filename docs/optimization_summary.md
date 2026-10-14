# lcpp Optimization Campaign — Final Summary

Date: 2026-10-14
Hardware: NVIDIA GeForce RTX 5060 (8 GB, Blackwell sm_120), Windows, MSVC 2026, CUDA 13.4 (CUB/CCCL 3.4)
Benchmark: `tests/lcpp_cub_bench.cpp` (xmake target `lcpp_cub_bench` + nvcc-compiled
`lcpp_cub_bench_kernels` CUB baseline), 4,194,304 items, median of 10 reps, results
verified **exactly** against a host reference and CUB on every run.
Measurement: cuda = CUDA events on the LC stream; dx/vk = amortized wall clock
(`HostSyncTimer`, batch=10; `LCPP_BENCH_BATCH=1` reproduces the epoch-0/1 mode).

## Results (median ms, lcpp vs CUB)

| op | backend | epoch 0 | final | speedup | lcpp/CUB: epoch 0 → final |
|---|---|---|---|---|---|
| DeviceReduce.Sum | cuda | 0.0325 | 0.0223 | 1.46x | 1.55x → **1.11x** |
| DeviceScan.InclusiveSum | cuda | 0.1860 | 0.0774 | 2.40x | 2.32x → **0.96x** |
| DeviceRadixSort.SortKeys | cuda | 0.7666 | 0.4285 | 1.79x | 2.24x → **1.00x** |
| DeviceReduce.Sum | dx | 0.1748¹ | 0.0434 | 4.0x¹ | 3.89x¹ → **0.93x** |
| DeviceScan.InclusiveSum | dx | 0.2937¹ | 0.1005 | 2.9x¹ | 3.81x¹ → **1.24x** |
| DeviceRadixSort.SortKeys | dx | 0.9481¹ | 0.6718 | 1.4x¹ | 2.76x¹ → **1.58x** |
| DeviceReduce.Sum | vk | 0.1481¹ | 0.0362 | 4.1x¹ | 3.17x¹ → **0.78x** |
| DeviceScan.InclusiveSum | vk | 0.3246¹ | 0.0921 | 3.5x¹ | 4.12x¹ → **1.15x** |
| DeviceRadixSort.SortKeys | vk | 3.8178¹ | 2.4836 | 1.54x¹ | 9.02x¹ → **5.86x** |

¹ epoch-0 dx/vk numbers used per-op `synchronize()` timing (~100+ µs backend
round-trip per op); final numbers use amortized timing — see
`docs/epoch1_host_overhead.md`. Same-backend before/after comparisons for dx/vk
are therefore indicative; the cuda column is directly comparable.

**Headline: on CUDA, all three primitives now run at CUB parity (0.96x–1.11x),
with scan and radix ~2.4x / ~1.8x faster than at epoch 0.**

## Epochs

| epoch | focus | key change | result (cuda) | doc |
|---|---|---|---|---|
| 0 | baseline + harness | bench target, CUB wrappers, vk/dx portability fixes | — | `epoch0_baseline.md` |
| 1 | host dispatch | string shader-cache keys → O(1) pointer-identity keys (`shader_cache.h`); fixed latent KEY_ONLY shader collision | host dispatch 0.25→0.16 µs (−35%) | `epoch1_host_overhead.md` |
| 2 | scan kernel | coalesced striped I/O via shared-memory transpose (blocked register layout preserved for order-correct BlockScan); policy 256×8→256×15 | 0.186→0.079 ms (2.4x), 0.96x CUB | `epoch2_scan_kernel.md` |
| 3 | radix sort | policy 256×4→256×16 (tile 4096, 4x fewer blocks → shorter lookback, 4x smaller lookback resets) | 0.767→0.428 ms (1.8x), 1.00x CUB | `epoch3_radix_sort.md` |
| 3.5 | review fix | sizeof-aware tile clamp `radix_effective_items_v` keeps default policy DX-safe for >4-byte keys (+ ulong test) | no perf change | `epoch3_radix_sort.md` addendum |
| 4 | reduce kernel | coalesced striped `ConsumeFullTile` loads; policy 256×4→256×8; ReduceByKey decoupled to its own DX-floor tile | 0.032→0.022 ms (1.46x), 1.11x CUB | `epoch4_reduce_kernel.md` |

## Verification (final state)
- Full regression `scripts/run_regression.sh`: **21/21 green** (7 tests ×
  cuda/dx/vk), incl. 16.7M-assert scan test and 67-assert radix test with the new
  ulong default-policy case — `docs/final_regression.log`.
- Ragged sizes (1,000,003 / 65,537 items) verified exactly on cuda/dx/vk.
- Full `xmake build` (all targets) OK.

## What remains (documented, out of scope)
- **vk radix sort on the XIR→SPIR-V path (5.9x CUB)**: per-phase breakdown shows
  the one-sweep kernel itself is ~90% of the time; its volatile lookback
  spin-polling and MatchAny rank are expensive on that codegen route — a backend
  characteristic, not a policy knob (epoch-3 doc). **Update (epoch 5):** with
  `lc_vk_backend_use_xir_spirv=false` (legacy HLSL/DXC path) the same kernel runs
  3.5x faster (0.71 ms, 1.68x CUB), so the gap is specific to the XIR→SPIR-V
  codegen, not the algorithm. Both vk routes now pass the full suite.
- dx/vk residual ratios for scan/radix partly reflect cross-backend comparison
  (CUB runs natively on CUDA on the same GPU) plus dx/vk dispatch overhead.
- `LCPP_BENCH_GRAPH=1` (CUDA graph capture) exists for submit-overhead-free
  cuda measurements.
