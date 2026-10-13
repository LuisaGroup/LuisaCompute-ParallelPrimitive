# CUDA and CUB parity performance receipt

Date: 2026-10-13
GPU: NVIDIA GeForce RTX 5090, compute capability 12.0
Driver: 13010
CUB: CCCL/CUB v3.5.0 (`300500` in the benchmark output)

## Changes under test

- CUDA scan has an opt-in `DeviceScanCudaParity` policy: 320 threads, 25 items per thread, warp-transpose load/store, warp-local synchronization, and a packed 64-bit tile status/value word.
- CUDA key-only radix sort uses the CUB-shaped 384-thread, 20-items-per-thread onesweep policy for 32-bit keys and reuses the rank workspace for the key tile.
- CUDA reduce and the block/warp helpers use the tuned shuffle and synchronization paths in this change set.

Portable defaults remain unchanged. The measurements below are for the CUDA backend only.

## 320 x 25 benchmark

The benchmark uses signed `int32` reduce/scan and `uint32` key-only radix sort on the same CUDA stream and context. It warms up before measurement, alternates LCPP and CUB, runs 60 repetitions, and validates every output. CUDA events surround a graph launch batch of four operations, so the values are GPU execution time per operation. JIT, allocation, and H2D setup are outside the timed region; tile-state initialization kernels and graph inter-kernel scheduling are inside it. The 4M table is the range across five independent runs; the ratio is `LCPP / CUB`.

### 4,194,304 items, five runs

| Operation | LCPP median across runs (us) | CUB median across runs (us) | Paired ratio median | Ratio range |
| --- | ---: | ---: | ---: | ---: |
| Reduce | 5.072 | 4.672 | 1.0856x | 1.0463x-1.0912x |
| Scan | 10.264 | 10.136 | 1.0158x | 1.0118x-1.0256x |
| Radix key-only | 88.104 | 100.208 | 0.8792x | 0.8778x-0.8852x |

The policy sweep for 4M scan gave `2.0126x` at 256x32, `1.4535x` at 512x16, `1.0632x` at 320x26, and `1.0110x` at 320x25. These are exploratory runs with 20 or 30 repetitions; the five-run table above is the authoritative result.

### 1,048,576 items, one 60-run replay

| Operation | LCPP (us) | CUB (us) | Ratio |
| --- | ---: | ---: | ---: |
| Reduce | 3.008 | 2.592 | 1.1605x |
| Scan | 5.400 | 4.696 | 1.1499x |
| Radix key-only | 45.608 | 42.712 | 1.0678x |

The 1M replay is included to keep the remaining small-input gap visible. These numbers do not claim universal parity across other types, input sizes, or backends.

## Nsight Compute conclusion

Report: `/tmp/lcpp_launch_scan320x25_4m.ncu-rep`
Command: `ncu --set launch --target-processes all --force-overwrite -o /tmp/lcpp_launch_scan320x25_4m build/linux/x86_64/release/lcpp_cub_bench_scan320x25 4194304 5`

This is a launch-metric report. Hardware performance counters were not collected, so it cannot establish DRAM bandwidth or warp-stall attribution.

| Kernel | Block/grid | Registers | Shared per block | Static shared | Waves/SM | Occupancy limits (register/shared/warp) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LCPP scan `kernel_main` | 320 / 525 | 48 | 33.120 KiB | 32.096 KiB | 1.03 | 4 / 3 / 4 |
| CUB scan `DeviceScanKernel` | 384 / 497 | 80 | 34.832 KiB | 33.808 KiB | 1.46 | 2 / 2 / 4 |
| LCPP radix `kernel_main` | 384 / 547 | 64 | 32.820 KiB | 31.796 KiB | 1.61 | 2 / 3 / 4 |
| CUB radix `DeviceRadixSortOnesweepKernel` | 384 / 547 | 80 | 32.768 KiB | 31.744 KiB | 1.61 | 2 / 3 / 4 |

The scan policy lowers register and shared-memory pressure enough for the launch calculator to admit three shared-memory-limited CTAs per SM, versus two for the CUB-shaped 384x22 launch. This is a resource-limit inference, not an achieved-occupancy measurement. Radix has the same launch shape and wave metric as CUB while using 16 fewer registers per thread.

Static PTX inspection of the generated kernels found four `ld.relaxed.gpu.u64` and three `st.relaxed.gpu.u64` instructions in the packed scan path, and one `ld.relaxed.gpu.u32` plus two `st.relaxed.gpu.u32` instructions in the radix lookback path. Neither path contains `membar.gl`. These are static instruction counts, not dynamic execution counts.

## CUDA correctness

Each target was rebuilt with XMake and run with `--backend=cuda` on the same RTX 5090:

| Target | Result |
| --- | --- |
| `device_scan_test` | 16,781,335 assertions, 10 tests passed |
| `device_reduce_test` | 93 assertions, 14 tests passed |
| `device_radix_sort_one_sweep` | 66 assertions, 3 tests passed |
| `device_radix_sort_interface_test` | 311 assertions, 6 tests passed |
| `block_exchange_shuffle_test` | 1,408 assertions, 1 test passed |

This receipt covers focused CUDA builds and tests. A full repository build, CTest matrix, Compute Sanitizer run, and full Nsight hardware-counter collection were not rerun in this change.
