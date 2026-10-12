# LCPP - LuisaCompute Parallel Primitives

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
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
- [x] **DeviceCopy** - Typed batched copies and multidimensional layout conversion
- [x] **DeviceFind** - First-match search and batched lower/upper bounds
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
- [x] **DeviceSegmentedRadixSort** - Numeric bit-range copy and DoubleBuffer sorting within segments
- [x] **DeviceSegmentedSort** - Ordered/stable copy and DoubleBuffer sorting within segments
- [x] **DeviceTopK** - Min/max key and pair selection with numeric or decomposed keys
- [x] **DeviceTransform** - Fill, generate, conditional, tuple-input, and tuple-output transforms
- [x] **DeviceSpmv** - CSR sparse-matrix vector multiplication (retained LCPP extension)

## CUB Comparison Coverage

This matrix is derived from the public semantic method and data-flow families in
CCCL/CUB v3.4.0 at commit
`a5630ac57111e1aaef8cfe0069ede566c3d15f60`. It records the portable
LuisaCompute surface implemented here.

### Completion Status

Portable semantic coverage of the pinned public surface is complete:

- [x] Add the aggregate-header device families that were not represented:
  `DeviceFind`, `DeviceMemcpy`, `DeviceMerge`, `DeviceTopK`, and
  `DeviceTransform`.
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
  register all 39 GPU behavior tests in both XMake and CMake.

This is semantic portability, not CUDA source compatibility. LCPP deliberately
collapses CUB's legacy/environment/stream wrappers and two-call scratch-query
protocol into LuisaCompute modules, `CommandList`, and explicit storage queries.
It does not claim arbitrary CUDA iterator compatibility, 64-bit dispatch,
architecture-specific performance parity, or identical overload signatures.
Some portable fallback algorithms favor clarity and correctness over CUB-level
asymptotic performance.

APIs introduced by later CCCL/CUB releases are not implicitly included here;
the comparison baseline must be pinned again before extending this roadmap.

### Verification

- Public aggregate: 6 thread, 7 warp, 13 block, and 21 device headers. The
  device set contains the 19 pinned aggregate families, public
  `DeviceSegmentedScan`, and the retained `DeviceSpmv` extension.
- Validation snapshot: Linux x86_64, NVIDIA GeForce RTX 5090, LuisaCompute
  `next` at `9ff7419332340683402993e3b5b57f186767e4c0` (2026-08-30).
- Forced XMake ordinary Release, CUDA: 39/39 executables, 136 tests,
  16,809,342 assertions.
- Forced XMake SAFE Release, CUDA: 39/39 executables, 136 tests,
  16,809,342 assertions.
- Forced XMake ordinary Release, Vulkan with native XIR-to-SPIR-V: 39/39
  executables, 136 tests, 16,809,342 assertions.
- Forced XMake SAFE Release, Vulkan with native XIR-to-SPIR-V: 39/39
  executables, 136 tests, 16,809,342 assertions.
- Independent CMake CUDA Release build and CTest: 39/39 tests passed.

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

[Specify your license here]

## Acknowledgments

- Built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) by LuisaGroup
- Inspired by NVIDIA's [CUB library](https://github.com/NVIDIA/cccl)

---

**Sources:**
- [CUB :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cub/index.html)
- [CUB — CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cub/)
