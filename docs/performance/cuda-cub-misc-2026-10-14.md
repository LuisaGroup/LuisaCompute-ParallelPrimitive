# CUDA and CUB miscellaneous primitives

Date: 2026-10-14. GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
Driver: 13010. CCCL/CUB: v3.5.0, version code 300500.
Backend: CUDA. Both implementations use the same context, stream, input, and operation semantics.

## Result

The large gaps in Find, Select, Partition, Merge, Unique, and run-length encoding have been reduced. At 4M items, Transform and Select are approximately equal to CUB, while Find and AdjacentDifference are faster in this workload. Partition, Merge, Unique, and RLE still have approximately 3-8% median overhead. SegmentReduce remains approximately 10-12% slower at 4M and was not changed in this optimization.

At 1M, Select and Unique retain approximately 10% overhead, and RLE retains approximately 19%. These measurements do not establish parity for other distributions, types, sizes, APIs, GPUs, or backends.

## Measurement

Each size has five independent benchmark processes, each with 60 alternating samples per implementation. A sample times a CUDA graph batch of four operations with CUDA events; the reported time is per operation. JIT compilation, allocation, H2D transfer, graph capture, and warmup finish before sampling. Every initialization, auxiliary kernel, and graph scheduling interval inside the operation is included.

All 13 cases pass CPU/LCPP/CUB validation before and after sampling in all ten processes. Measurements run without a profiler. Background desktop GPU activity was present and was not stopped; all sample outliers remain in the raw logs. The tables show the median of the five per-run medians and the median/range of the five paired ratios. Small differences with overlapping sample ranges should be read as approximate parity.

### 4,194,304 items

| Operation | LCPP median (us) | CUB median (us) | Paired LCPP/CUB | Ratio range, 5 runs | Kernel nodes LCPP/CUB |
| --- | ---: | ---: | ---: | --- | --- |
| Transform.Transform | 6.624 | 6.640 | 0.9976x | 0.9761x-1.0024x | 1/1 |
| AdjacentDifference.SubtractLeftCopy | 6.648 | 7.184 | 0.9254x | 0.9242x-0.9723x | 1/1 |
| Find.FindIf.head | 4.120 | 4.584 | 0.8972x | 0.8871x-0.9023x | 2/2 |
| Find.FindIf.tail | 4.096 | 4.576 | 0.8958x | 0.8920x-1.0090x | 2/2 |
| Find.FindIf.miss | 4.072 | 4.600 | 0.8870x | 0.8852x-0.8942x | 2/2 |
| Select.If.50pct | 9.696 | 9.704 | 1.0000x | 0.9836x-1.0066x | 2/2 |
| Partition.Flagged.25pct | 13.872 | 13.272 | 1.0439x | 1.0423x-1.0802x | 2/2 |
| Merge.MergeKeys | 17.864 | 17.368 | 1.0299x | 1.0286x-1.0313x | 1/2 |
| Histogram.HistogramEven.256bins | 8.080 | 8.288 | 0.9768x | 0.9406x-0.9863x | 2/2 |
| Select.Unique.run16 | 10.744 | 10.136 | 1.0556x | 1.0504x-1.0726x | 2/2 |
| RunLengthEncode.Encode.run16 | 12.824 | 11.856 | 1.0816x | 1.0580x-1.0838x | 3/2 |
| SegmentReduce.Sum.seg256 | 11.784 | 10.712 | 1.1001x | 1.0985x-1.1162x | 1/1 |
| SegmentReduce.Sum.seg4096 | 4.568 | 4.072 | 1.1238x | 1.1051x-1.1400x | 1/1 |

### 1,048,576 items

| Operation | LCPP median (us) | CUB median (us) | Paired LCPP/CUB | Ratio range, 5 runs | Kernel nodes LCPP/CUB |
| --- | ---: | ---: | ---: | --- | --- |
| Transform.Transform | 2.552 | 2.528 | 1.0095x | 1.0000x-1.0127x | 1/1 |
| AdjacentDifference.SubtractLeftCopy | 2.904 | 3.032 | 0.9713x | 0.8417x-0.9838x | 1/1 |
| Find.FindIf.head | 2.536 | 2.520 | 1.0000x | 0.9808x-1.0095x | 2/2 |
| Find.FindIf.tail | 2.536 | 2.536 | 0.9969x | 0.9937x-1.0063x | 2/2 |
| Find.FindIf.miss | 2.528 | 2.544 | 0.9968x | 0.9906x-0.9969x | 2/2 |
| Select.If.50pct | 5.600 | 5.096 | 1.0989x | 1.0063x-1.1022x | 2/2 |
| Partition.Flagged.25pct | 6.640 | 6.400 | 1.0197x | 1.0088x-1.0520x | 2/2 |
| Merge.MergeKeys | 6.640 | 8.168 | 0.8129x | 0.8100x-0.8149x | 1/2 |
| Histogram.HistogramEven.256bins | 4.576 | 5.088 | 0.8994x | 0.8216x-0.9134x | 2/2 |
| Select.Unique.run16 | 5.592 | 5.096 | 1.0973x | 1.0943x-1.1022x | 2/2 |
| RunLengthEncode.Encode.run16 | 6.656 | 5.608 | 1.1871x | 1.1826x-1.1929x | 3/2 |
| SegmentReduce.Sum.seg256 | 4.072 | 3.560 | 1.1483x | 1.0993x-1.1608x | 1/1 |
| SegmentReduce.Sum.seg4096 | 2.464 | 2.064 | 1.1992x | 1.1711x-1.2101x | 1/1 |

### Comparison with the pre-fusion 4M baseline

The baseline is one 30-sample process, recorded immediately before these changes. Its paired ratios are useful for locating the large changes, but it is not a five-process statistical baseline.

| Operation | Before LCPP/CUB | After LCPP/CUB |
| --- | ---: | ---: |
| Find, head | 2.0035x | 0.8972x |
| Select.If, 50% | 2.5791x | 1.0000x |
| Partition.Flagged, 25% | 1.8953x | 1.0439x |
| MergeKeys | 3.2099x | 1.0299x |
| Unique, runs of 16 | 2.4577x | 1.0556x |
| RLE, runs of 16 | 3.1305x | 1.0816x |

## Implementation

- Out-of-place Select.If, Select.Flagged, and Unique fuse predicate/adjacency evaluation, scan, look-back, and scatter for eligible 32-bit inputs and flags. Ranks stay in registers; a shared-memory exchange produces coalesced output. The default CUDA configuration is 384 threads by 15 items. Other widths retain flags materialization and a uint32 scan; 64-bit values and boolean flags are covered by regression tests.
- Partition shares the compaction implementation. Rejected items use the reverse output index directly, retaining CUB's selected-first/reverse-rejected ordering.
- RLE uses fused head compaction to produce unique values and run offsets, followed by a coalesced offsets-to-lengths kernel. Its count includes all three kernels. The existing temporary-storage query still reserves the portable workspace; the CUDA fast path does not stream all those reserved arrays.
- Merge searches both tile boundaries in parallel, loads the slices into shared memory, searches locally, and writes with warp-transpose stores. The CUDA fast path uses 256 threads by 8 items for the default configuration.
- Find processes 16 coalesced items per thread and combines candidate indices within a warp before the global atomic minimum.
- Transform and AdjacentDifference use aligned 128-bit loads/stores for eligible 32-bit scalar inputs. Tail and unaligned cases retain guarded scalar handling.
- Histogram processes 16 items per thread in its contiguous 256-bin CUDA specialization, reducing the number of global flushes.
- Callables that use builtin indices, warp collectives, atomics, shared memory, or external functions retain the unbatched pointwise execution. Out-of-place compaction requires separate input/output resources.
- Compaction initializes the complete 32-lane leading look-back padding, including inputs with fewer than 32 tiles. Tests poison scratch memory before reuse.

The two-kernel RLE experiments were discarded: 384x15, 256x8, and CUB-matching 256x14 variants were slower than the retained three-kernel path. The 256x14 variant had the same grid and theoretical block residency as CUB but took about 1.38x its execution time. Kernel count and launch policy alone therefore do not explain the remaining RLE gap.

## Resource evidence

Hardware-counter collection with Nsight Compute still fails with ERR_NVGPUCTRPERM; /proc/driver/nvidia/params reports RmProfilingAdminOnly: 1. No achieved occupancy, DRAM bandwidth, or warp-stall attribution is claimed.

The benchmark's optional LCPP_BENCH_LAUNCH mode reads CUDA driver function attributes and the occupancy calculator outside sampling. These are compiled resource limits, not achieved occupancy.

| Main kernel | LCPP block/grid | CUB block/grid | Registers LCPP/CUB | Static shared bytes LCPP/CUB | Maximum blocks/SM LCPP/CUB |
| --- | --- | --- | --- | --- | --- |
| Select.If | 384/729 | 384/729 | 56/66 | 23104/23040 | 3/2 |
| Unique | 384/729 | 384/729 | 68/72 | 23104/23040 | 2/2 |
| Partition.Flagged | 384/729 | 448/852 | 66/40 | 46144/19712 | 2/3 |
| MergeKeys | 256/2048 | 512/547 | 38/40 | 16392/30800 | 5/3 |
| RLE head compaction / CUB reduce-by-key | 384/729 | 256/1171 | 66/64 | 46144/28688 | 2/3 |
| SegmentReduce, seg256 | 256/16384 | 256/16384 | 40/40 | 64/256 | 6/6 |

Select removes an entire positions-array pass and now has the same main-kernel launch shape as CUB. SegmentReduce has the same block/grid, registers, and theoretical block limit as CUB, so those resource limits alone do not account for its remaining gap. Partition and RLE still use more shared memory per block; whether that limits achieved occupancy requires counter-enabled profiling.

## Reproduce

```sh
node benchmarks/rebuild_cuda_cub_misc.cjs
build/linux/x86_64/release/cuda_cub_misc 1048576 60
build/linux/x86_64/release/cuda_cub_misc 4194304 60
LCPP_BENCH_LAUNCH=1 build/linux/x86_64/release/cuda_cub_misc 4194304 5
```

Optional arguments: item count, repetition count, and case-name substring.
LCPP_BENCH_BATCH controls graph batching; LCPP_BENCH_GRAPH=0 selects direct submission.
Raw execution timings from profiler runs are not comparable with the unprofiled tables.

Baseline log: /tmp/cuda_cub_misc_before_fused_4m.log.
Final logs: /tmp/cuda_cub_misc_delivery_1m_r1.log through r5.log and /tmp/cuda_cub_misc_delivery_4m_r1.log through r5.log.
Driver attributes: /tmp/cuda_cub_misc_delivery_launch_4m.log.
Counter-permission failure: /tmp/lcpp_misc_rle_256x8_launch_4m.log.

## Verification

Eight focused targets were rebuilt and passed on CUDA: Select (167 assertions), Partition (54), RLE (757), Find (55), Merge (59), Transform (71), AdjacentDifference (29), and Histogram (296). Total: 1,488 assertions.

The new cases cover partial tiles, more than 32 tiles, empty/all-selected tiles, long runs spanning tiles, single-item runs, skewed and duplicate merge inputs, descending order, subviews, poisoned scratch, 64-bit values, boolean flags, and pointwise builtin-index/warp-collective semantics.

Compute Sanitizer memcheck passed Select, RLE, and Merge with zero errors. Select also passed synccheck with zero errors. Logs are /tmp/device_select_test_delivery_memcheck.log, /tmp/device_run_length_encode_test_delivery_memcheck.log, /tmp/device_merge_test_delivery_memcheck.log, and /tmp/device_select_test_delivery_synccheck.log.

git diff --check passed. A full repository build/test, other-backend validation, and an application end-to-end benchmark were not run.

## PR #5 review follow-up

The Select.If and two-way Partition.If fallbacks now materialize pointwise flags when the predicate cannot be batched. Scan and scatter consume those same flags instead of invoking the predicate in different execution contexts. The wide-input fallback also reuses flags, so side effects run once per item. Eligible pure 32-bit predicates retain the fused CUDA path.

The predicate-flag shader caches in these two modules include captured resource bindings in their keys. LuisaCompute's structural function hash excludes binding payloads; using it alone could reuse a shader bound to a previously released buffer when an otherwise identical atomic predicate captured a new buffer.

Two new regression tests cover dispatch_id/dispatch_size, warp prefix sums, and stateful atomic predicates for both 32-bit and 64-bit inputs at 1, 33, and 4,099 items. They check selected counts, complete stable selection or selected-first/reverse-rejected partition output, and one atomic invocation per item. Captured buffers change across input sizes. The unchanged tests failed on the original implementation with count/output/call-count assertions and an invalid-memory-access abort. After the fixes, Select passed 209 assertions in 7 tests and Partition passed 96 assertions in 6 tests, totaling 305 assertions. The initial eight-target results above describe the pre-review performance commit; this follow-up rebuilt and reran these two affected targets.

Compute Sanitizer memcheck and synccheck both passed both targets with zero errors. Before/after logs are /tmp/lcpp_pr5_select_before.log, /tmp/lcpp_pr5_partition_before.log, /tmp/lcpp_pr5_select_after.log, and /tmp/lcpp_pr5_partition_after.log. Sanitizer logs are /tmp/lcpp_pr5_select_memcheck.log, /tmp/lcpp_pr5_partition_memcheck.log, /tmp/lcpp_pr5_select_synccheck.log, and /tmp/lcpp_pr5_partition_synccheck.log.

The benchmark was rebuilt after the fixes. One additional unprofiled 4M, 60-sample process passed all 13 CPU/LCPP/CUB case validations. Select.If measured 0.9690x CUB and Partition.Flagged measured 1.0780x; both retain two kernel nodes. This replay is a smoke check, not a replacement for the five-process tables above. Its log is /tmp/lcpp_pr5_benchmark_after_4m.log. Full repository build/test, other backends, and end-to-end application performance were not rerun for this follow-up.
