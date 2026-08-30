include(FetchContent)

# boost ut
FetchContent_Declare(
        boost_ut
        GIT_REPOSITORY "https://github.com/boost-ext/ut.git"
        GIT_TAG "master"
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(boost_ut)

# cpptrace
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY "https://github.com/jeremy-rifkin/cpptrace"
        GIT_TAG "main"
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cpptrace)

function(lcpp_add_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE lcpp Boost::ut cpptrace::cpptrace)
    add_test(NAME ${test_name} COMMAND ${test_name} "--ctx=$<TARGET_FILE_DIR:luisa-compute-core>")
#    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "LUISA_EXPERIMENTAL_LLVM_CODEGEN=1")
endfunction()

lcpp_add_test(block_level_test)
lcpp_add_test(block_exchange_shuffle_test)
lcpp_add_test(block_io_test)
lcpp_add_test(block_adjacent_difference_test)
lcpp_add_test(block_collective_policy_test)
lcpp_add_test(block_discontinuity_test)
lcpp_add_test(block_histogram_test)
lcpp_add_test(block_merge_sort_test)
lcpp_add_test(block_radix_rank_test)
lcpp_add_test(block_radix_sort_test)
lcpp_add_test(block_run_length_decode_test)
lcpp_add_test(thread_level_test)
lcpp_add_test(decoupled_look_back)
lcpp_add_test(device_reduce_test)
lcpp_add_test(device_adjacent_difference_test)
lcpp_add_test(device_copy_test)
lcpp_add_test(device_for_test)
lcpp_add_test(device_find_test)
lcpp_add_test(device_histogram_test)
lcpp_add_test(device_merge_sort_test)
lcpp_add_test(device_segmented_sort_test)
lcpp_add_test(device_segmented_scan_test)
lcpp_add_test(device_spmv_test)
lcpp_add_test(device_partition_test)
lcpp_add_test(device_run_length_encode_test)
lcpp_add_test(device_scan_test)
lcpp_add_test(device_select_test)
lcpp_add_test(device_segment_reduce)
lcpp_add_test(device_radix_sort_one_sweep)
lcpp_add_test(device_radix_sort_interface_test)
lcpp_add_test(warp_level_test)
lcpp_add_test(warp_shared_collectives_test)
lcpp_add_test(warp_io_test)
lcpp_add_test(warp_exchange_test)
lcpp_add_test(warp_merge_sort_test)
