add_requires("boost_ut","cpptrace")
local function add_test_target(file_name)
    target(file_name)
        set_kind("binary")
        add_files(file_name..".cpp")
        set_languages("c++20")
        add_deps("lcpp")
        add_packages("boost_ut","cpptrace")
        on_load(function(target)
            if has_config('lc_use_xrepo') then
                target:add('packages', "luisa-compute", {public = true})
            else
                target:add('deps', 'lc-runtime', 'lc-dsl')
            end

            if target:is_plat('macosx') then
                target:add('defines', '__APPLE__')
            end
        end)
        -- add run path for luisa-compute
        on_config(function (target)
            if has_config('lc_use_xrepo') then
                if not os.exists(target:targetdir()) then
                    os.mkdir(target:targetdir())
                end
                if is_host("windows") then
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dll"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc.exe"), target:targetdir())
                elseif is_host("macosx") then
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dylib"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc"), target:targetdir())
                else
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.so"), target:targetdir())
                    os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/luisa_nvrtc"), target:targetdir())
                end
            end
        end)
    target_end()
end

add_test_target("block_level_test")
add_test_target("block_exchange_shuffle_test")
add_test_target("block_io_test")
add_test_target("block_adjacent_difference_test")
add_test_target("block_collective_policy_test")
add_test_target("block_discontinuity_test")
add_test_target("block_histogram_test")
add_test_target("block_merge_sort_test")
add_test_target("block_radix_rank_test")
add_test_target("block_radix_sort_test")
add_test_target("block_run_length_decode_test")
add_test_target("thread_level_test")
add_test_target("warp_level_test")
add_test_target("warp_shared_collectives_test")
add_test_target("warp_io_test")
add_test_target("warp_exchange_test")
add_test_target("warp_merge_sort_test")
add_test_target("decoupled_look_back")
add_test_target("device_reduce_test")
add_test_target("device_adjacent_difference_test")
add_test_target("device_copy_test")
add_test_target("device_for_test")
add_test_target("device_find_test")
add_test_target("device_histogram_test")
add_test_target("device_merge_sort_test")
add_test_target("device_merge_test")
add_test_target("device_memcpy_test")
add_test_target("device_segmented_sort_test")
add_test_target("device_segmented_scan_test")
add_test_target("device_spmv_test")
add_test_target("device_transform_test")
add_test_target("device_partition_test")
add_test_target("device_run_length_encode_test")
add_test_target("device_scan_test")
add_test_target("device_select_test")
add_test_target("device_segment_reduce")
add_test_target("device_radix_sort_one_sweep")
add_test_target("device_radix_sort_interface_test")
