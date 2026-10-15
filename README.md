# LCPP - LuisaCompute Parallel Primitives

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)

A high-performance GPU parallel primitives library built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), providing thread-level to device-level parallel operations. LCPP is a header-only C++20 library inspired by CUDA CUB (Cooperative primitives for CUDA C++).

## Features

LCPP provides a comprehensive set of parallel primitives organized in a hierarchical architecture:

```
Device Level  → Reduce, Scan, Sort, Select, Partition, Find, Transform, Copy, RLE, SpMV
       ↓
Agent Layer   → Algorithm policy management (e.g., OneSweepSmallKeyTunedPolicy)
       ↓
Block Level   → Reduce, Scan, Load/Store, Exchange, Sort, Histogram, Shuffle
       ↓
Warp Level    → Reduce, Scan, Exchange, Load/Store, MergeSort (32 threads)
       ↓
Thread Level  → Reduce, Scan, Search, Sort, Load, Store
```

### Key Features

- **Header-only library** - Easy integration without compilation
- **Multi-backend support** - Leverages LuisaCompute's backend abstraction (CUDA, Metal, DirectX, Vulkan)
- **Lazy compilation** - Shaders are compiled on-demand using the `lazy_compile` macro
- **Flexible algorithms** - Multiple algorithm implementations (SHARED_MEMORY, WARP_SHUFFLE)
- **Type-safe** - Modern C++20 with strong type checking
- **Portable semantics first** - Optimized policies where available, with portable fallbacks

## Build

### Prerequisites

- C++20 compatible compiler
- CMake 3.26+ or XMake 3.0.6+
- LuisaCompute (prepared with XMake or automatically fetched by CMake)

### XMake (Recommended)

```bash
# Clone LuisaCompute's next branch, or safely fast-forward an existing checkout
xmake prepare-lc

# Configure and build
xmake f -m release
xmake

# Build tests
xmake build -g test

# Run tests (CUDA is the Linux default)
xmake run block_level_test
xmake run warp_level_test
xmake run device_reduce_test
xmake run device_scan_test
xmake run device_segment_reduce
xmake run device_radix_sort_one_sweep

# Select Vulkan explicitly
xmake run device_scan_test --backend=vk
```

`xmake prepare-lc` only updates a clean `LuisaCompute/` checkout on its `next`
branch with the official `origin` URL. It refuses dirty, divergent, wrong-branch,
or wrong-remote checkouts instead of overwriting local work. Regular `xmake`
commands do not access the network when `lc_use_xrepo=false`.

The vendored XMake configuration enables LuisaCompute's native XIR-to-SPIR-V
path for Vulkan. Keep `lc_vk_backend_use_xir_spirv=true`: the complete LCPP
surface includes bindless destination-buffer writes that are not implemented by
the legacy HLSL-to-SPIR-V path.

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=ON
cmake --build .

# Run tests
ctest
# Or run individual tests
./tests/block_level_test
./tests/device_scan_test
```

## Benchmarking, diagnostics and the vk-CUDA interop path

`tests/run_regression.sh` is replaced by `scripts/run_regression.sh` (legacy
cuda/dx/vk matrix + `LCPP_INTEROP` modes; see `--help` inside the script for
`--legacy-only` / `--interop-only` / `--policy-only`).

`lcpp_cub_bench` measures `DeviceReduce` / `DeviceScan` / `DeviceRadixSort`
against a same-run CUB baseline on the same GPU and prints CSV rows
(`operation,items,lcpp_median_ms,lcpp_min_ms,lcpp_max_ms,cub_*,lcpp_over_cub`).
Run `lcpp_cub_bench --help` for the full knob list; the two that matter when
comparing numbers:

- `LCPP_BENCH_BATCH=N` — number of op invocations, each in its own
  `CommandList` and commit, bracketed by **one** `synchronize()` pair; the
  reported time is the wall time divided by `N`. `N=1` exposes the raw per-op
  host floor (~120–150 us of vk submit + completion per op); `N>=100`
  amortizes it and exposes GPU throughput. **Throughput tables must be taken
  at `N>=100`** — a batch-1 "vk 0.15 ms vs cuda 0.02 ms" row is the host floor,
  not a codegen gap.
- `LCPP_INTEROP` — vk only. `1`/`all` (all modules), `auto` (the measured
  per-op recommendation: reduce/segment/radix on the interop path, scan on the
  DSL path), `a,b,c` (explicit list), `0`/unset (pure-vulkan DSL everywhere).
  See `src/lcpp/runtime/interop_dispatch.h` for the contract, and
  `.kimix_cache/epoch_C_report.md` for the measurements behind `auto`.

`LCPP_BENCH_POLL=1` observes completion by polling a stream event instead of
`synchronize()`; `LCPP_BENCH_INQUEUE_LIMIT=0` turns the vk backend's depth-2
in-flight throttle off (measured: a large regression on WDDM, see
`--help`); `LCPP_BENCH_CL_OPS=K` measures K invocations recorded into **one**
caller `CommandList` over K distinct buffer sets — the pattern that hides the
per-op submit floor, because the vk command-reorder pass merges such
non-aliasing iterations into barrier-free layers.

`host_enqueue_probe` is the host-side cost microscope: recording cost per call
(`[backend] [items] [reps] [calls_per_cl]`), `--commit` for the per-commit
encode+submit cost, `--micro` for a host-only DSL-vs-interop encoder
micro-benchmark, and a `KernelLauncher` ABI packing self-check that runs on
every invocation.

`policy_tune` is the compile-time policy sweep harness (Epoch D): the one-sweep
radix tile and the reduce tile are template parameters, so the harness
instantiates a fixed set of them and measures the requested subset on the DSL
path and — on vk, when `LCPP_INTEROP` selects the module — on the interop path,
interleaved round-robin, with a bit-exactness check against a host reference for
every case.  `--help`-style usage is in the file header; rows are
`PT,<op>,<tile>,<path>,<items>,<median_ms>,<min_ms>,<max_ms>,<rounds>,<ok>`.
Knobs: `--op=radix|reduce|all`, `--items=N`, `--reps=R`, `--batch=B`
(`B=1` floor, `B>=100` throughput), `--cl-ops=K` (K ops in one CommandList, the
Epoch-C C3 pattern), `--tiles=`, `--reduce-tiles=`, `--dsl-max-items=`
(the vk/dx HLSL codegen rejects > 32 KiB of static shared memory, which a
keys-only radix one-sweep tile reaches above 31 items/thread).

### Per-op policy revisited under the CUDA codegen (Epoch D)

The interop path compiles kernels with the cuda backend, which is not bound by
the 32 KiB HLSL/DX shared-memory floor that fixed the portable policy defaults.
Epoch D re-measured the two candidates that floor had blocked
(`.kimix_cache/epoch_D_report.md`): the radix one-sweep tile is already optimal
at `256x16` for the cuda codegen (tile 32 is 0.5 % better but needs interop
gating, tile 24/40 are worse), so it is unchanged; the plain narrow-integral
reduce may use `256x16` on the interop path only
(`REDUCE_INTEROP_ITEMS_PER_THREAD` in `src/lcpp/device/device_reduce.h`), worth
1.5 % at 16M items on vk/`LCPP_INTEROP=auto` and a tie at 4M.  The gate excludes
8-byte `IndexValuePairT` (ArgMin/ArgMax tie-breaking), floating-point reductions
(non-associative) and transforming reductions (whose recursion structure would
change), so every other backend and tile keeps its existing kernels and results.

## Quick Start

```cpp
#include <lcpp/parallel_primitive.h>
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

// Create device and stream
Context context{argv[0]};
Device device = context.create_device("cuda");
Stream stream = device.create_stream();

// Device-level reduce example
DeviceReduce<> device_reduce;
device_reduce.create(device, &stream);

constexpr size_t item_count = 4096u;
Buffer<int> input = device.create_buffer<int>(item_count);
Buffer<int> output = device.create_buffer<int>(1u);
size_t temp_bytes = DeviceReduce<>::GetTempStorageBytes<int>(item_count);
Buffer<uint> temp_storage =
    device.create_buffer<uint>((temp_bytes + sizeof(uint) - 1u) / sizeof(uint));

CommandList cmdlist;
device_reduce.Sum(cmdlist, temp_storage.view(), input.view(), output.view(), item_count);
stream << cmdlist.commit() << synchronize();
```

## Implemented Features

### ✅ Thread Level
- [x] **ThreadReduce** - Full and partial fixed-array reductions
- [x] **ThreadScan** - Full and partial inclusive/exclusive fixed-array scans
- [x] **ThreadSearch** - Lower/upper-bound and merge-path searches
- [x] **ThreadSort** - Stable key and key-value odd-even sorting
- [x] **ThreadLoad / ThreadStore** - Cache-modifier-compatible scalar access

### ✅ Warp Level (32 threads)
- [x] **WarpReduce** - Scalar/array, valid-item, and head/tail-segmented reductions
- [x] **WarpReduceBatched** - Independent batched reductions with direct, striped, and blocked layouts
- [x] **WarpScan** - Full and partial inclusive/exclusive scans and broadcast
- [x] **WarpExchange** - Data exchange patterns within a warp
- [x] **WarpLoad / WarpStore** - Blocked, striped, and transpose I/O
- [x] **WarpMergeSort** - Stable key and key-value merge sort

### ✅ Block Level (typically 256 threads)
- [x] **BlockReduce** - Block-level reduction with SHARED_MEMORY and WARP_SHUFFLE algorithms
- [x] **BlockScan** - Inclusive/exclusive scans, aggregates, initial values, and prefix callbacks
- [x] **BlockLoad** - Efficient block-wide data loading (DIRECT, STRIPED modes)
- [x] **BlockStore** - Efficient block-wide data storing
- [x] **BlockExchange** - Data rearrangement within a block
- [x] **BlockRadixRank** - Ranking operations for radix sort
- [x] **BlockRadixSort** - Numeric/decomposed key and pair sorting with bit ranges and layout conversion
- [x] **BlockDiscontinuity** - Flag head/tail discontinuities in sequences
- [x] **BlockAdjacentDifference** - Left/right differences and flags
- [x] **BlockHistogram** - Sort and atomic histogram policies
- [x] **BlockMergeSort** - Stable blocked and striped merge sort
- [x] **BlockRunLengthDecode** - Block-wide run decoding
- [x] **BlockShuffle** - Offset, up, and down shuffles

### ✅ Device Level
- [x] **DeviceAdjacentDifference** - In-place/copy left and right adjacent differences
- [x] **DeviceBatchedTopK** - Per-segment min/max key and pair selection over flat buffers
- [x] **DeviceCopy** - Typed batched copies and multidimensional layout conversion
- [x] **DeviceFind** - First-match search, batched bounds, and sorted-query bound variants
- [x] **DeviceFor** - Mutable/copy iteration, bulk dispatch, and multidimensional layouts
- [x] **DeviceHistogram** - Even/range histogram computation
- [x] **DeviceMemcpy** - Batched raw-byte copies over bindless source and destination ranges
- [x] **DeviceMerge** - Sorted key and key-value pair merging with custom ordering
- [x] **DeviceMergeSort** - Copy/in-place key and pair sorting with stable variants
- [x] **DevicePartition** - Flagged, two-way predicate, and three-way predicate partitioning
- [x] **DeviceRadixSort** - Numeric/decomposed key and pair sorting with copy and DoubleBuffer flows
- [x] **DeviceReduce** - Reductions, custom arg comparison, transform-reduce, and reduce-by-key
- [x] **DeviceRunLengthEncode** - Full and non-trivial run encoding
- [x] **DeviceScan** - Inclusive/exclusive, initialized-inclusive, and by-key scans
- [x] **DeviceSelect** - Flagged/FlaggedIf, predicate, unique, and unique-by-key copy/in-place selection
- [x] **DeviceSegmentReduce** - Segmented reduction operations
- [x] **DeviceSegmentedScan** - Inclusive/exclusive scans with shared or independent output offsets
  - Built-in 32-bit sums cooperate within a warp or block; general operators retain serial left-to-right evaluation.
- [x] **DeviceSegmentedRadixSort** - Numeric bit-range copy and DoubleBuffer sorting within segments
- [x] **DeviceSegmentedSort** - Ordered/stable copy and DoubleBuffer sorting within segments
- [x] **DeviceTopK** - Min/max key and pair selection with numeric or decomposed keys
- [x] **DeviceTransform** - Fill, generate, conditional, tuple-input, and tuple-output transforms
- [x] **DeviceSpmv** - CSR sparse-matrix vector multiplication (retained LCPP extension)

## CUB Comparison Coverage

This matrix is derived from the public semantic method and data-flow families in
CCCL/CUB v3.5.0 at release commit
`28101b07413743e1dcca8ca8af1cfc63e8cff2d1`. It records the portable
LuisaCompute surface implemented here.

### Completion Status

Portable semantic coverage of the pinned public surface is complete:

- [x] Add the aggregate-header device families that were not represented:
  `DeviceBatchedTopK`, `DeviceFind`, `DeviceMemcpy`, `DeviceMerge`, `DeviceTopK`, and
  `DeviceTransform`.
- [x] Add the v3.5 sorted-query `DeviceFind` variants and the portable
  flat-buffer `DeviceBatchedTopK` key/pair method groups.
- [x] Evaluate the public families outside the aggregate header and implement
  the portable ones, including `thread_search`, `thread_sort`,
  `WarpReduceBatched`, and `DeviceSegmentedScan`.
- [x] Audit existing families overload-by-overload and implement every distinct
  portable method/data-flow group, including copy/in-place and DoubleBuffer
  forms, decomposed radix keys, multidimensional layouts, initialized scans,
  partial collectives, custom comparators, and three-way partitioning.
- [x] Cover temporary-storage queries, empty inputs, aliasing, range boundaries,
  stability, selector updates, and custom comparison behavior where applicable.
- [x] Export every implemented family through `lcpp/parallel_primitive.h` and
  register all 40 GPU behavior tests in both XMake and CMake.

This is semantic portability, not CUDA source compatibility. LCPP deliberately
collapses CUB's legacy/environment/stream wrappers and two-call scratch-query
protocol into LuisaCompute modules, `CommandList`, and explicit storage queries.
It does not claim arbitrary CUDA iterator compatibility, 64-bit dispatch,
architecture-specific performance parity, or identical overload signatures.
Some portable fallback algorithms favor clarity and correctness over CUB-level
asymptotic performance.

The v3.5 `DeviceBatchedTopK` adapter intentionally uses LuisaCompute buffer
views plus begin/end offsets and packed `segment * k` output. CUDA iterator
arrays, `cuda::args` deferred problem sizes, execution environments, and CUB's
architecture-specific tuning remain outside the portable contract.

### Verification

- Public aggregate: 6 thread, 7 warp, 13 block, and 22 device headers. The
  device set contains the 20 v3.5 aggregate families, public
  `DeviceSegmentedScan`, and the retained `DeviceSpmv` extension.
- Validation snapshot: Linux x86_64, NVIDIA GeForce RTX 5090, LuisaCompute
  `next` at `9ff7419332340683402993e3b5b57f186767e4c0` (2026-10-12).
- Forced XMake ordinary Release, CUDA: 40/40 executables, 140 tests,
  16,809,376 assertions.
- Forced XMake SAFE Release, CUDA: 40/40 executables, 140 tests,
  16,809,376 assertions.
- Forced XMake ordinary Release, Vulkan with native XIR-to-SPIR-V: 40/40
  executables, 140 tests, 16,809,376 assertions.
- Forced XMake SAFE Release, Vulkan with native XIR-to-SPIR-V: 40/40
  executables, 140 tests, 16,809,376 assertions.
- Independent CMake CUDA Release build and CTest: 40/40 tests passed.

Both Vulkan rows were run with `LUISA_VULKAN_VALIDATION=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and `LUISA_DUMP_SOURCE=1`. These
guards reject compatibility-route fallback and force fresh JIT code generation.

This snapshot does not claim runtime validation for Metal or DirectX. The
portable API surface is shared, but those backends still require their own
build and GPU test runs.

### Priority 1: High-Priority Device Operations

#### DeviceSelect
- [x] `DeviceSelect::Flagged` - Select items based on selection flags
- [x] `DeviceSelect::FlaggedIf` - Select items by applying a predicate to flags
- [x] `DeviceSelect::If` - Select items based on predicate
- [x] `DeviceSelect::Unique` - Select unique items from input sequence
- [x] `DeviceSelect::UniqueByKey` - Select unique keys from key-value pairs
- [x] Copy and exact in-place data flows where provided by the pinned contract

#### DevicePartition
- [x] `DevicePartition::Flagged` - Partition based on selection flags
- [x] `DevicePartition::If` - Two-way and three-way predicate partitioning

#### DeviceReduce Extensions
- [x] `DeviceReduce::ReduceByKey` - Reduce segments defined by keys
- [x] `DeviceReduce::ArgMin` - Find minimum value and its index
- [x] `DeviceReduce::ArgMax` - Find maximum value and its index
- [x] Custom strict-order comparators for argument reductions

#### DeviceScan Extensions
- [x] `DeviceScan::InclusiveScanByKey` - Segmented inclusive scan with keys
- [x] `DeviceScan::ExclusiveScanByKey` - Segmented exclusive scan with keys
- [x] `DeviceScan::InclusiveScanInit` - Inclusive scan with an explicit initial value
- [x] `DeviceScanCudaParity` - CUDA packed warp-transpose policy tuned for CUB parity (`320` threads, `25` items/thread)
- [x] `DeviceScan<>` automatically selects this policy for ordinary 4-byte CUDA and Vulkan CUDA interop scans; other backends and scan variants retain their portable policies
- [x] Custom key equality and distinct key/value types

### Priority 2: Sorting and Merging

#### DeviceRadixSort
- [x] Numeric and decomposed key/pair sorting in ascending and descending order
- [x] Copy and DoubleBuffer data flows with full/partial bit ranges and selector updates

#### DeviceMergeSort
- [x] `DeviceMergeSort::SortKeys` - Stable merge sort for keys
- [x] `DeviceMergeSort::SortPairs` - Stable merge sort for key-value pairs
- [x] `DeviceMergeSort::StableSortKeys` - Guaranteed stable sort
- [x] `DeviceMergeSort::StableSortPairs` - Guaranteed stable sort for pairs
- [x] `SortKeysCopy`, `StableSortKeysCopy`, and `SortPairsCopy` data flows

#### DeviceSegmentedSort
- [x] `DeviceSegmentedRadixSort` - Copy/DoubleBuffer radix ordering within segments
- [x] `DeviceSegmentedSort` - Copy/DoubleBuffer ordered and stable sorting within segments

#### BlockMergeSort
- [x] `BlockMergeSort::Sort` - Block-level stable merge sort
- [x] `BlockMergeSort::SortBlockedToStriped` - Sort with layout transformation

### Priority 3: Advanced Operations

#### DeviceRunLengthEncode
- [x] `DeviceRunLengthEncode::Encode` - Run-length encoding
- [x] `DeviceRunLengthEncode::NonTrivialRuns` - Encode runs with length > 1

#### DeviceAdjacentDifference
- [x] `DeviceAdjacentDifference::SubtractLeft` - In-place left subtraction
- [x] `DeviceAdjacentDifference::SubtractLeftCopy` - Copy with left subtraction
- [x] `DeviceAdjacentDifference::SubtractRight` - In-place right subtraction
- [x] `DeviceAdjacentDifference::SubtractRightCopy` - Copy with right subtraction

#### DeviceSpmv (Sparse Matrix Operations)
- [x] `DeviceSpmv::CsrMV` - Sparse matrix-vector multiplication (CSR format)

#### DeviceCopy
- [x] `DeviceCopy::Batched` - Batched memory copy operations
- [x] `DeviceCopy::Copy` - Multidimensional row/column-major layout conversion

#### DeviceMemcpy
- [x] `DeviceMemcpy::Batched` - Bindless raw-byte range copies

### Priority 4: Block-Level Extensions

#### BlockRadixSort
- [x] `BlockRadixSort::Sort` - Complete radix sort (not just ranking)
- [x] `BlockRadixSort::SortDescending` - Descending radix sort
- [x] `BlockRadixSort::SortBlockedToStriped` - Sort with layout change
- [x] Numeric and decomposed custom keys with full and partial bit ranges

#### BlockHistogram
- [x] `BlockHistogram::Composite` - Block-level histogram computation
- [x] `BlockHistogram::Init` - Initialize histogram bins

#### BlockAdjacentDifference
- [x] `BlockAdjacentDifference::SubtractLeft` - Block-level left subtraction
- [x] `BlockAdjacentDifference::SubtractRight` - Block-level right subtraction
- [x] `BlockAdjacentDifference::FlagHeads` - Flag heads of segments
- [x] `BlockAdjacentDifference::FlagTails` - Flag tails of segments

#### BlockRunLengthDecode
- [x] `BlockRunLengthDecode::RunLengthDecode` - Decode run-length encoded data

#### BlockShuffle
- [x] `BlockShuffle::Up` - Shuffle data up within block
- [x] `BlockShuffle::Down` - Shuffle data down within block
- [x] `BlockShuffle::Offset` - Shuffle data by offset

### Priority 5: Warp-Level Extensions

#### WarpMergeSort
- [x] `WarpMergeSort::Sort` - Warp-level stable merge sort

#### WarpLoad/WarpStore
- [x] `WarpLoad` - Optimized warp-level data loading
- [x] `WarpStore` - Optimized warp-level data storing

#### WarpScan Extensions
- [x] `WarpScan::Broadcast` - Broadcast value across warp
- [x] Full and partial fixed-array inclusive/exclusive scan groups

#### WarpReduce Extensions
- [x] `WarpReduce::HeadSegmentedReduce` - Segmented reduce with head flags
- [x] `WarpReduce::TailSegmentedReduce` - Segmented reduce with tail flags
- [x] Fixed-array full and partial reduction groups

### Priority 6: Thread-Level Extensions

#### ThreadReduce / ThreadScan
- [x] Fixed-array full and partial reduction and scan groups

#### ThreadSearch / ThreadSort
- [x] Lower/upper-bound and merge-path search utilities
- [x] Stable odd-even key and key/value sorting

#### ThreadLoad/ThreadStore
- [x] `ThreadLoad` - Thread-level load with cache modifiers (CG, CA, CS, CV)
- [x] `ThreadStore` - Thread-level store with cache modifiers (CG, CS, WT, WB)

### Algorithm Improvements
- [x] Add SHARED_MEMORY implementations for Warp operations
- [x] Add TRANSPOSE and WARP_TRANSPOSE modes for BlockLoad
- [x] Add VECTORIZE mode for BlockLoad/BlockStore
- [ ] Optimize policies for different GPU architectures
- [ ] Add support for 64-bit indexing for large data

## Architecture

### Core Design Patterns

1. **LuisaModule Base Class** - All parallel operations inherit from `LuisaModule`, providing unified type definitions and shader compilation interface
2. **Lazy Shader Compilation** - Shaders are compiled on first use via `lazy_compile` macro
3. **Policy-Based Configuration** - Algorithm strategies configured through Policy template parameters
4. **Multiple Algorithm Support** - Block operations support both `SHARED_MEMORY` and `WARP_SHUFFLE` algorithms

### Directory Structure

```
src/lcpp/
├── device/           # Device-level operations
│   ├── details/      # Implementation details
│   └── device_*.h    # Public interfaces
├── agent/            # Algorithm policies and tuning
├── block/            # Block-level operations (256 threads)
│   └── detail/       # Implementation details
├── warp/             # Warp-level operations (32 threads)
│   └── details/      # Implementation details
├── thread/           # Thread-level operations
├── common/           # Common utilities and type traits
└── runtime/          # LuisaModule base and core definitions
```

### Key Files

- `src/lcpp/parallel_primitive.h` - Main header including all primitives
- `src/lcpp/runtime/core.h` - `LuisaModule` base class and `lazy_compile` macro
- `src/lcpp/agent/policy.h` - Tuning policies and parameters

## Code Style

- **Standard**: LLVM style with modifications (see `.clang-format`)
- **Line Width**: 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Language**: C++20

## Dependencies

- **LuisaCompute** - GPU compute framework (automatically fetched via CMake FetchContent)
- **boost_ut** - Unit testing framework
- **cpptrace** - Stack tracing utility

## Documentation

For detailed API documentation, see the header files in `src/lcpp/`. Each primitive class is documented with:
- Template parameters
- Member functions
- Algorithm options
- Usage examples

## Contributing

Contributions are welcome! Please feel free to:
1. Pick an item from the TODO list
2. Implement the feature following existing code patterns
3. Add tests in the `tests/` directory
4. Submit a pull request

### Implementation Guidelines

- Follow the existing class hierarchy (inherit from `LuisaModule`)
- Use `lazy_compile` for shader compilation
- Support multiple algorithm variants when applicable
- Add comprehensive tests
- Document public APIs

## References

- [LuisaCompute Documentation](https://github.com/LuisaGroup/LuisaCompute)
- [CUDA CUB Library](https://nvidia.github.io/cccl/cub/)
- [CUDA CUB Documentation](https://docs.nvidia.com/cuda/cub/index.html)

## License

LCPP is distributed under the [Apache License, Version 2.0](LICENSE).

The license applies to this repository's source and documentation. Third-party
dependencies, including LuisaCompute and CCCL/CUB, retain their own licenses
and notices.

## Acknowledgments

- Built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) by LuisaGroup
- Inspired by NVIDIA's [CUB library](https://github.com/NVIDIA/cccl)

---

**Sources:**
- [CUB :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cub/index.html)
- [CUB — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cub/)
