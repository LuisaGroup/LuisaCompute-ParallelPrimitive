# LCPP - LuisaCompute Parallel Primitives

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)

A high-performance GPU parallel primitives library built on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), providing thread-level to device-level parallel operations. LCPP is a header-only C++20 library inspired by CUDA CUB (Cooperative primitives for CUDA C++).

## Features

LCPP provides a comprehensive set of parallel primitives organized in a hierarchical architecture:

```
Device Level  → Reduce, Scan, Sort, Select, Partition, RLE, Copy, SpMV
       ↓
Agent Layer   → Algorithm policy management (e.g., OneSweepSmallKeyTunedPolicy)
       ↓
Block Level   → Reduce, Scan, Load/Store, Exchange, Sort, Histogram, Shuffle
       ↓
Warp Level    → Reduce, Scan, Exchange, Load/Store, MergeSort (32 threads)
       ↓
Thread Level  → Reduce, Scan, Load, Store
```

### Key Features

- **Header-only library** - Easy integration without compilation
- **Multi-backend support** - Leverages LuisaCompute's backend abstraction (CUDA, Metal, DirectX, Vulkan)
- **Lazy compilation** - Shaders are compiled on-demand using the `lazy_compile` macro
- **Flexible algorithms** - Multiple algorithm implementations (SHARED_MEMORY, WARP_SHUFFLE)
- **Type-safe** - Modern C++20 with strong type checking
- **Performance-focused** - Optimized policies and tuning parameters

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

# Run tests
xmake run block_level_test
xmake run warp_level_test
xmake run device_reduce_test
xmake run device_scan_test
xmake run device_segment_reduce
xmake run device_radix_sort_one_sweep
```

`xmake prepare-lc` only updates a clean `LuisaCompute/` checkout on its `next`
branch with the official `origin` URL. It refuses dirty, divergent, wrong-branch,
or wrong-remote checkouts instead of overwriting local work. Regular `xmake`
commands do not access the network when `lc_use_xrepo=false`.

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
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
- [x] **ThreadReduce** - Thread-level reduction operations
- [x] **ThreadScan** - Thread-level scan (prefix sum) operations
- [x] **ThreadLoad / ThreadStore** - Cache-modifier-compatible scalar access

### ✅ Warp Level (32 threads)
- [x] **WarpReduce** - Warp-level reduction (Sum, Min, Max, custom operators)
- [x] **WarpScan** - Warp-level inclusive/exclusive scan
- [x] **WarpExchange** - Data exchange patterns within a warp
- [x] **WarpLoad / WarpStore** - Blocked, striped, and transpose I/O
- [x] **WarpMergeSort** - Stable key and key-value merge sort

### ✅ Block Level (typically 256 threads)
- [x] **BlockReduce** - Block-level reduction with SHARED_MEMORY and WARP_SHUFFLE algorithms
- [x] **BlockScan** - Block-level inclusive/exclusive scan with prefix callback support
- [x] **BlockLoad** - Efficient block-wide data loading (DIRECT, STRIPED modes)
- [x] **BlockStore** - Efficient block-wide data storing
- [x] **BlockExchange** - Data rearrangement within a block
- [x] **BlockRadixRank** - Ranking operations for radix sort
- [x] **BlockRadixSort** - Ascending/descending key and pair sorting
- [x] **BlockDiscontinuity** - Flag head/tail discontinuities in sequences
- [x] **BlockAdjacentDifference** - Left/right differences and flags
- [x] **BlockHistogram** - Sort and atomic histogram policies
- [x] **BlockMergeSort** - Stable blocked and striped merge sort
- [x] **BlockRunLengthDecode** - Block-wide run decoding
- [x] **BlockShuffle** - Offset, up, and down shuffles

### ✅ Device Level
- [x] **DeviceReduce** - Device-wide reduction (Sum, Min, Max, custom operators)
- [x] **DeviceScan** - Device-wide inclusive/exclusive scan with decoupled look-back
- [x] **DeviceRadixSort** - OneSweep key/pair sorting with ascending, descending, and bit-range interfaces
- [x] **DeviceMergeSort** - Stable key and key-value merge sorting
- [x] **DeviceSegmentedSort** - Stable sorting within non-overlapping segments
- [x] **DeviceSegmentedRadixSort** - Bit-range radix ordering within segments
- [x] **DeviceSpmv** - CSR sparse-matrix vector multiplication
- [x] **DeviceSegmentReduce** - Segmented reduction operations
- [x] **DeviceHistogram** - Histogram computation
- [x] **DeviceFor** - Parallel for-loop utilities
- [x] **DeviceSelect** - Flagged, predicate, unique, and unique-by-key selection
- [x] **DevicePartition** - Flagged and predicate partitioning

## CUB Comparison Coverage

This repository-scoped matrix is based on the interface families selected from
CCCL/CUB v3.4.0. It records the portable LuisaCompute surface implemented here;
it is not a claim of source-level or exhaustive API parity with every CUB release.

### Completion Status

The repository-selected roadmap below is complete. Exhaustive parity with all
public headers and overloads at CCCL/CUB v3.4.0
(`a5630ac57111e1aaef8cfe0069ede566c3d15f60`) is a separate goal and is not yet
complete:

- [ ] Add the aggregate-header device families that are not represented yet:
  `DeviceFind`, `DeviceMemcpy`, `DeviceMerge`, `DeviceTopK`, and
  `DeviceTransform`.
- [ ] Evaluate the public families outside the aggregate header and implement
  the portable ones, including `thread_search`, `thread_sort`,
  `WarpReduceBatched`, and `DeviceSegmentedScan`.
- [ ] Audit and implement remaining overload parity in existing families, such
  as `DeviceFor::{Bulk,ForEachCopy,ForEachCopyN}`,
  `DeviceSelect::FlaggedIf`,
  `DeviceScan::InclusiveScanInit`, and merge-sort copy variants.

APIs introduced by later CCCL/CUB releases are not implicitly included here;
the comparison baseline must be pinned again before extending this roadmap.

### Priority 1: High-Priority Device Operations

#### DeviceSelect
- [x] `DeviceSelect::Flagged` - Select items based on selection flags
- [x] `DeviceSelect::If` - Select items based on predicate
- [x] `DeviceSelect::Unique` - Select unique items from input sequence
- [x] `DeviceSelect::UniqueByKey` - Select unique keys from key-value pairs

#### DevicePartition
- [x] `DevicePartition::Flagged` - Partition based on selection flags
- [x] `DevicePartition::If` - Partition based on predicate

#### DeviceReduce Extensions
- [x] `DeviceReduce::ReduceByKey` - Reduce segments defined by keys
- [x] `DeviceReduce::ArgMin` - Find minimum value and its index
- [x] `DeviceReduce::ArgMax` - Find maximum value and its index

#### DeviceScan Extensions
- [x] `DeviceScan::InclusiveScanByKey` - Segmented inclusive scan with keys
- [x] `DeviceScan::ExclusiveScanByKey` - Segmented exclusive scan with keys
- [x] Custom key equality and distinct key/value types

### Priority 2: Sorting and Merging

#### DeviceMergeSort
- [x] `DeviceMergeSort::SortKeys` - Stable merge sort for keys
- [x] `DeviceMergeSort::SortPairs` - Stable merge sort for key-value pairs
- [x] `DeviceMergeSort::StableSortKeys` - Guaranteed stable sort
- [x] `DeviceMergeSort::StableSortPairs` - Guaranteed stable sort for pairs

#### DeviceSegmentedSort
- [x] `DeviceSegmentedRadixSort` - Stable radix ordering within non-overlapping segments
- [x] `DeviceSegmentedSort` - Stable general sorting within non-overlapping segments

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

### Priority 4: Block-Level Extensions

#### BlockRadixSort
- [x] `BlockRadixSort::Sort` - Complete radix sort (not just ranking)
- [x] `BlockRadixSort::SortDescending` - Descending radix sort
- [x] `BlockRadixSort::SortBlockedToStriped` - Sort with layout change

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

#### WarpReduce Extensions
- [x] `WarpReduce::HeadSegmentedReduce` - Segmented reduce with head flags
- [x] `WarpReduce::TailSegmentedReduce` - Segmented reduce with tail flags

### Priority 6: Thread-Level Extensions

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
