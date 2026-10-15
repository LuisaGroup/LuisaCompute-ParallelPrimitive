# DX backend known gaps (post-merge verification, 2026-10-14)

After merging `origin/main` (CUB/CCCL v3.5.0 parity) into local `main`
(epoch0-6 perf tuning), the full 40-binary test suite was built with xmake
and run on all three backends:

| Backend | Passed | Failed |
|---------|--------|--------|
| cuda    | 40/40  | 0      |
| vk      | 40/40  | 0      |
| dx      | 34/40  | 6 (all pre-existing legacy-HLSL gaps, see below) |

The 6 DX failures are **not merge regressions**. Each failing test (or the
failing sub-case) is new from `origin/main`, which validated CUDA only
(see `docs/performance/cuda-cub-parity-2026-10-13.md`); the vendored
LuisaCompute HLSL codegen was not touched by the merge. Each failure maps to
a genuine capability gap of the legacy-HLSL DX codegen or a hardware limit:

| Test | Failure | Missing feature / limit |
|------|---------|-------------------------|
| `device_copy_test` | `LUISA_NOT_IMPLEMENTED()` at `function_codegen.cpp:1987` | `CallOp::BINDLESS_BUFFER_WRITE` (and PACK/UNPACK) not implemented in legacy-HLSL lowering |
| `device_memcpy_test` | same | same: raw byte-buffer ops unavailable on DX |
| `device_scan_test` | "Bitwise cast not implemented for type 'double'" (`hlsl_codegen.cpp:340`) | no FLOAT64 arm in the HLSL `bit_cast` switch (double sub-case is remote-new) |
| `device_partition_test` | `..._preserves_wide_values_and_boolean_flags` wrong results | 64-bit (`ulong`) value movement + bool flags not lowered correctly by legacy-HLSL |
| `device_select_test` | `..._preserves_wide_values_and_boolean_flags` wrong results | same wide-value + bool limitation |
| `device_radix_sort_interface_test` | `shared_size <= 32768` assert (`hlsl_codegen.cpp:739`) | DX/SM group-shared memory is capped at 32 KB; the interface test's block policy requests more |

Note: the epoch5 claim "vk legacy-HLSL (DXC) path fully supported" refers to
the **Vulkan** backend's DXC path (green at 40/40 here), not the DX device
backend.

Real fixes would be backend features in vendored LuisaCompute (HLSL
double/64-bit bitcast, bindless byte writes, SM 6.10 64 KB shared), not
lcpp changes; alternatively the affected sub-cases can be tagged/skipped on
`--backend=dx` in the test harness.
