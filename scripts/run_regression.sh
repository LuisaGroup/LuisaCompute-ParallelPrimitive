#!/bin/bash
# Regression runner: all lcpp tests x {cuda, dx, vk}
cd "$(dirname "$0")/.."
export PATH="$PWD/LuisaCompute/bin/release:$PATH"
BIN=build/windows/x64/release
TESTS="block_level_test warp_level_test decoupled_look_back device_reduce_test device_scan_test device_segment_reduce device_radix_sort_one_sweep"
for t in $TESTS; do
  for b in cuda dx vk; do
    if [ "$t" = "block_level_test" ] || [ "$t" = "warp_level_test" ]; then
      out=$($BIN/$t.exe --backend=$b 2>&1 | tail -1)
    else
      out=$($BIN/$t.exe --backend=$b 2>&1 | tail -1)
    fi
    echo "$t $b :: $out"
  done
done
