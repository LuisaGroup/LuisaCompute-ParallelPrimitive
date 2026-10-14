/*
 * @Author: Ligo
 * @Date: 2025-11-12 14:42:25
 * @Last Modified time: 2026-08-30
 */
#pragma once

#include <lcpp/common/type_trait.h>
#include <lcpp/runtime/core.h>

#include <array>
#include <limits>
#include <type_traits>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;

template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
class DeviceHistogram : public LuisaModule
{
  private:
    Device m_device;
    bool   m_created = false;

    luisa::unordered_map<luisa::string, luisa::shared_ptr<Resource>> m_reset_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<Resource>> m_even_shaders;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<Resource>> m_range_shaders;

    static constexpr size_t FAST_EVEN_BINS = 256u;

  public:
    DeviceHistogram()  = default;
    ~DeviceHistogram() = default;

    void create(Device& device)
    {
        m_device  = device;
        m_created = true;
    }

    [[nodiscard]] static constexpr size_t GetTempStorageBytes(size_t, size_t) noexcept
    {
        return 0u;
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void HistogramEven(CommandList& cmdlist,
                       BufferView<uint>,
                       BufferView<SampleT>  samples,
                       BufferView<CounterT> histogram,
                       size_t               num_levels,
                       LevelT               lower_level,
                       LevelT               upper_level,
                       size_t               num_samples)
    {
        histogram_even(cmdlist, samples, histogram, num_levels, lower_level, upper_level, num_samples, 1u, num_samples, 1u, 0u);
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void HistogramEven(CommandList& cmdlist,
                       BufferView<uint>,
                       BufferView<SampleT>  samples,
                       BufferView<CounterT> histogram,
                       size_t               num_levels,
                       LevelT               lower_level,
                       LevelT               upper_level,
                       size_t               num_row_samples,
                       size_t               num_rows,
                       size_t               row_stride_bytes)
    {
        histogram_even(cmdlist,
                       samples,
                       histogram,
                       num_levels,
                       lower_level,
                       upper_level,
                       num_row_samples,
                       num_rows,
                       row_stride_in_samples<SampleT>(row_stride_bytes),
                       1u,
                       0u);
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void HistogramRange(CommandList& cmdlist,
                        BufferView<uint>,
                        BufferView<SampleT>  samples,
                        BufferView<CounterT> histogram,
                        BufferView<LevelT>   levels,
                        size_t               num_levels,
                        size_t               num_samples)
    {
        histogram_range(cmdlist, samples, histogram, levels, num_levels, num_samples, 1u, num_samples, 1u, 0u);
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void HistogramRange(CommandList& cmdlist,
                        BufferView<uint>,
                        BufferView<SampleT>  samples,
                        BufferView<CounterT> histogram,
                        BufferView<LevelT>   levels,
                        size_t               num_levels,
                        size_t               num_row_samples,
                        size_t               num_rows,
                        size_t               row_stride_bytes)
    {
        histogram_range(cmdlist,
                        samples,
                        histogram,
                        levels,
                        num_levels,
                        num_row_samples,
                        num_rows,
                        row_stride_in_samples<SampleT>(row_stride_bytes),
                        1u,
                        0u);
    }

    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void MultiHistogramEven(CommandList& cmdlist,
                            BufferView<uint>,
                            BufferView<SampleT>                                          samples,
                            const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                            const std::array<uint, NUM_ACTIVE_CHANNELS>&                 num_levels,
                            const std::array<LevelT, NUM_ACTIVE_CHANNELS>& lower_levels,
                            const std::array<LevelT, NUM_ACTIVE_CHANNELS>& upper_levels,
                            size_t                                         num_pixels)
    {
        size_t row_stride_samples =
            checked_multiply(num_pixels, NUM_CHANNELS, "Histogram pixel count overflows.");
        multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            cmdlist, samples, histograms, num_levels, lower_levels, upper_levels, num_pixels, 1u, row_stride_samples);
    }

    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void MultiHistogramEven(CommandList& cmdlist,
                            BufferView<uint>,
                            BufferView<SampleT>                                          samples,
                            const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                            const std::array<uint, NUM_ACTIVE_CHANNELS>&                 num_levels,
                            const std::array<LevelT, NUM_ACTIVE_CHANNELS>& lower_levels,
                            const std::array<LevelT, NUM_ACTIVE_CHANNELS>& upper_levels,
                            size_t                                         num_row_pixels,
                            size_t                                         num_rows,
                            size_t                                         row_stride_bytes)
    {
        multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(cmdlist,
                                                                samples,
                                                                histograms,
                                                                num_levels,
                                                                lower_levels,
                                                                upper_levels,
                                                                num_row_pixels,
                                                                num_rows,
                                                                row_stride_in_samples<SampleT>(row_stride_bytes));
    }

    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void MultiHistogramRange(CommandList& cmdlist,
                             BufferView<uint>,
                             BufferView<SampleT> samples,
                             const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                             const std::array<BufferView<LevelT>, NUM_ACTIVE_CHANNELS>& levels,
                             const std::array<uint, NUM_ACTIVE_CHANNELS>&               num_levels,
                             size_t                                                     num_pixels)
    {
        size_t row_stride_samples =
            checked_multiply(num_pixels, NUM_CHANNELS, "Histogram pixel count overflows.");
        multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            cmdlist, samples, histograms, levels, num_levels, num_pixels, 1u, row_stride_samples);
    }

    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void MultiHistogramRange(CommandList& cmdlist,
                             BufferView<uint>,
                             BufferView<SampleT> samples,
                             const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                             const std::array<BufferView<LevelT>, NUM_ACTIVE_CHANNELS>& levels,
                             const std::array<uint, NUM_ACTIVE_CHANNELS>&               num_levels,
                             size_t num_row_pixels,
                             size_t num_rows,
                             size_t row_stride_bytes)
    {
        multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            cmdlist, samples, histograms, levels, num_levels, num_row_pixels, num_rows, row_stride_in_samples<SampleT>(row_stride_bytes));
    }

  private:
    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void multi_histogram_even(CommandList&        cmdlist,
                              BufferView<SampleT> samples,
                              const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                              const std::array<uint, NUM_ACTIVE_CHANNELS>&   num_levels,
                              const std::array<LevelT, NUM_ACTIVE_CHANNELS>& lower_levels,
                              const std::array<LevelT, NUM_ACTIVE_CHANNELS>& upper_levels,
                              size_t                                         num_row_pixels,
                              size_t                                         num_rows,
                              size_t                                         row_stride_samples)
    {
        static_assert(NUM_ACTIVE_CHANNELS <= NUM_CHANNELS);
        for(size_t channel = 0u; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
            histogram_even(cmdlist,
                           samples,
                           histograms[channel],
                           num_levels[channel],
                           lower_levels[channel],
                           upper_levels[channel],
                           num_row_pixels,
                           num_rows,
                           row_stride_samples,
                           NUM_CHANNELS,
                           channel);
        }
    }

    template <size_t NUM_CHANNELS, size_t NUM_ACTIVE_CHANNELS, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void multi_histogram_range(CommandList&        cmdlist,
                               BufferView<SampleT> samples,
                               const std::array<BufferView<CounterT>, NUM_ACTIVE_CHANNELS>& histograms,
                               const std::array<BufferView<LevelT>, NUM_ACTIVE_CHANNELS>& levels,
                               const std::array<uint, NUM_ACTIVE_CHANNELS>& num_levels,
                               size_t                                       num_row_pixels,
                               size_t                                       num_rows,
                               size_t                                       row_stride_samples)
    {
        static_assert(NUM_ACTIVE_CHANNELS <= NUM_CHANNELS);
        for(size_t channel = 0u; channel < NUM_ACTIVE_CHANNELS; ++channel)
        {
            histogram_range(cmdlist,
                            samples,
                            histograms[channel],
                            levels[channel],
                            num_levels[channel],
                            num_row_pixels,
                            num_rows,
                            row_stride_samples,
                            NUM_CHANNELS,
                            channel);
        }
    }

    template <typename... T>
    [[nodiscard]] static luisa::string shader_key(luisa::string_view algorithm)
    {
        luisa::string key{algorithm};
        ((key += "+", key += Type::of<T>()->description()), ...);
        return key;
    }

    [[nodiscard]] static size_t checked_multiply(size_t lhs, size_t rhs, luisa::string_view message)
    {
        LUISA_ASSERT(lhs == 0u || rhs <= std::numeric_limits<size_t>::max() / lhs, "{}", message);
        return lhs * rhs;
    }

    [[nodiscard]] static size_t checked_add(size_t lhs, size_t rhs, luisa::string_view message)
    {
        LUISA_ASSERT(rhs <= std::numeric_limits<size_t>::max() - lhs, "{}", message);
        return lhs + rhs;
    }

    template <typename SampleT>
    [[nodiscard]] static size_t row_stride_in_samples(size_t row_stride_bytes)
    {
        LUISA_ASSERT(row_stride_bytes % sizeof(SampleT) == 0u,
                     "Histogram row stride must be a multiple of the sample size.");
        return row_stride_bytes / sizeof(SampleT);
    }

    static size_t validate_sample_layout(size_t sample_count,
                                         size_t num_row_samples,
                                         size_t num_rows,
                                         size_t row_stride_samples,
                                         size_t sample_stride,
                                         size_t channel)
    {
        LUISA_ASSERT(channel < sample_stride, "Histogram channel exceeds sample stride.");
        size_t row_payload = checked_multiply(num_row_samples, sample_stride, "Histogram row width overflows.");
        LUISA_ASSERT(num_rows <= std::numeric_limits<uint>::max()
                         && num_row_samples <= std::numeric_limits<uint>::max()
                         && row_stride_samples <= std::numeric_limits<uint>::max()
                         && sample_stride <= std::numeric_limits<uint>::max()
                         && channel <= std::numeric_limits<uint>::max(),
                     "Histogram dimensions exceed the shader index range.");
        LUISA_ASSERT(row_stride_samples >= row_payload, "Histogram row stride is smaller than the row payload.");
        size_t logical_samples =
            checked_multiply(num_row_samples, num_rows, "Histogram sample count overflows.");
        LUISA_ASSERT(logical_samples <= std::numeric_limits<uint>::max(),
                     "Histogram dispatch exceeds the shader index range.");
        if(logical_samples == 0u)
        {
            return 0u;
        }
        size_t prior_rows =
            checked_multiply(num_rows - 1u, row_stride_samples, "Histogram row layout overflows.");
        size_t required_samples = checked_add(prior_rows, row_payload, "Histogram sample layout overflows.");
        LUISA_ASSERT(sample_count >= required_samples, "Histogram sample view is too small.");
        return logical_samples;
    }

    template <NumericT CounterT>
    void reset_histogram(CommandList& cmdlist, BufferView<CounterT> histogram, size_t num_bins)
    {
        if(num_bins == 0u)
        {
            return;
        }
        using ResetShader = Shader<1, Buffer<CounterT>>;
        auto key          = shader_key<CounterT>("histogram_reset");
        auto shader_it    = m_reset_shaders.find(key);
        if(shader_it == m_reset_shaders.end())
        {
            U<ResetShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<CounterT> output) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             output.write(dispatch_id().x, CounterT{});
                         });
            m_reset_shaders.try_emplace(key, std::move(shader));
            shader_it = m_reset_shaders.find(key);
        }
        auto shader = reinterpret_cast<ResetShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile histogram reset shader.");
        cmdlist << (*shader)(histogram).dispatch(num_bins);
    }

    template <NumericT CommonT>
    static UInt integral_even_bin(Var<CommonT> sample, Var<CommonT> lower, Var<CommonT> upper, UInt num_bins)
    {
        ULong delta      = cast<ulong>(sample) - cast<ulong>(lower);
        ULong range      = cast<ulong>(upper) - cast<ulong>(lower);
        ULong multiplier = cast<ulong>(num_bins);
        ULong quotient   = 0ull;
        ULong remainder  = 0ull;
        for(int bit = 31; bit >= 0; --bit)
        {
            Bool double_wraps = remainder >= range - remainder;
            ULong doubled = select(remainder + remainder, remainder - (range - remainder), double_wraps);
            quotient  = quotient * 2ull + select(0ull, 1ull, double_wraps);
            remainder = doubled;

            Bool  bit_is_set = ((multiplier >> static_cast<uint>(bit)) & 1ull) != 0ull;
            Bool  add_wraps  = remainder >= range - delta;
            ULong added      = select(remainder + delta, remainder - (range - delta), add_wraps);
            quotient  = select(quotient, quotient + select(0ull, 1ull, add_wraps), bit_is_set);
            remainder = select(remainder, added, bit_is_set);
        }
        return cast<uint>(quotient);
    }

    template <size_t BINS, size_t FAST_BLOCK_SIZE, NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void histogram_even_contiguous(CommandList&         cmdlist,
                                   BufferView<SampleT>  samples,
                                   BufferView<CounterT> histogram,
                                   LevelT               lower_level,
                                   LevelT               upper_level,
                                   size_t               num_samples)
    {
        // Aggregate samples in shared memory before the global histogram update.
        using CommonT = std::common_type_t<SampleT, LevelT>;
        static_assert(std::is_integral_v<CommonT>);
        using FastEvenShader = Shader<1, Buffer<SampleT>, Buffer<CounterT>, uint, LevelT, LevelT>;
        auto key             = shader_key<SampleT, CounterT, LevelT>("histogram_even_contiguous");
        auto shader_it       = m_even_shaders.find(key);
        if(shader_it == m_even_shaders.end())
        {
            U<FastEvenShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<SampleT>  samples_buffer,
                            BufferVar<CounterT> histogram_buffer,
                            UInt                num_samples_var,
                            Var<LevelT>         lower_level_var,
                            Var<LevelT>         upper_level_var) noexcept
                         {
                             constexpr size_t fast_items_per_thread =
                                 ITEMS_PER_THREAD == details::ITEMS_PER_THREAD ? 16u : ITEMS_PER_THREAD;
                             set_block_size(FAST_BLOCK_SIZE);
                             SmemTypePtr<CounterT> local_histogram = new SmemType<CounterT>{BINS};
                             UInt                  thread          = thread_id().x;
                             $for(init_bin, thread, UInt(BINS), UInt(FAST_BLOCK_SIZE))
                             {
                                 local_histogram->write(init_bin, CounterT{});
                             };
                             sync_block();

                             UInt tile_start = block_id().x * UInt(FAST_BLOCK_SIZE * fast_items_per_thread);
                             UInt valid_items = min(UInt(FAST_BLOCK_SIZE * fast_items_per_thread),
                                                    num_samples_var - tile_start);
                             for(size_t item = 0u; item < fast_items_per_thread; ++item)
                             {
                                 UInt local = thread + UInt(item * FAST_BLOCK_SIZE);
                                 $if(local < valid_items)
                                 {
                                     UInt index = tile_start + local;
                                     Var<CommonT> sample = cast<CommonT>(samples_buffer.read(index));
                                     Var<CommonT> lower = cast<CommonT>(lower_level_var);
                                     Var<CommonT> upper = cast<CommonT>(upper_level_var);
                                     $if(sample >= lower & sample < upper)
                                     {
                                         UInt sample_bin = cast<uint>(sample - lower);
                                         local_histogram->atomic(sample_bin).fetch_add(CounterT{1});
                                     };
                                 };
                             }
                             sync_block();
                             $for(flush_bin, thread, UInt(BINS), UInt(FAST_BLOCK_SIZE))
                             {
                                 Var<CounterT> count = local_histogram->read(flush_bin);
                                 $if(count != CounterT{})
                                 {
                                     histogram_buffer.atomic(flush_bin).fetch_add(count);
                                 };
                             };
                         });
            m_even_shaders.try_emplace(key, std::move(shader));
            shader_it = m_even_shaders.find(key);
        }
        auto shader = reinterpret_cast<FastEvenShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile contiguous even-histogram shader.");
        constexpr size_t fast_items_per_thread =
            ITEMS_PER_THREAD == details::ITEMS_PER_THREAD ? 16u : ITEMS_PER_THREAD;
        size_t tile_items =
            checked_multiply(FAST_BLOCK_SIZE, fast_items_per_thread, "Histogram tile size overflows.");
        size_t blocks = num_samples / tile_items + static_cast<size_t>(num_samples % tile_items != 0u);
        size_t dispatch_threads =
            checked_multiply(blocks, FAST_BLOCK_SIZE, "Histogram dispatch size overflows.");
        cmdlist << (*shader)(samples, histogram, static_cast<uint>(num_samples), lower_level, upper_level)
                       .dispatch(dispatch_threads);
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void histogram_even(CommandList&         cmdlist,
                        BufferView<SampleT>  samples,
                        BufferView<CounterT> histogram,
                        size_t               num_levels,
                        LevelT               lower_level,
                        LevelT               upper_level,
                        size_t               num_row_samples,
                        size_t               num_rows,
                        size_t               row_stride_samples,
                        size_t               sample_stride,
                        size_t               channel)
    {
        using CommonT = std::common_type_t<SampleT, LevelT>;
        static_assert(NumericT<CommonT>);
        LUISA_ASSERT(m_created, "DeviceHistogram::create must be called before dispatch.");
        LUISA_ASSERT(num_levels >= 2u, "A histogram needs at least two levels.");
        LUISA_ASSERT(num_levels <= std::numeric_limits<uint>::max(), "Too many histogram levels.");
        LUISA_ASSERT(static_cast<CommonT>(lower_level) < static_cast<CommonT>(upper_level),
                     "Histogram upper level must exceed lower level.");
        LUISA_ASSERT(histogram.size() >= num_levels - 1u, "Histogram output is too small.");
        size_t logical_samples = validate_sample_layout(
            samples.size(), num_row_samples, num_rows, row_stride_samples, sample_stride, channel);

        reset_histogram(cmdlist, histogram, num_levels - 1u);
        if(logical_samples == 0u)
        {
            return;
        }

        if constexpr(std::is_integral_v<CommonT>
                     && (std::is_same_v<CounterT, uint> || std::is_same_v<CounterT, int>))
        {
            const CommonT lower = static_cast<CommonT>(lower_level);
            const CommonT upper = static_cast<CommonT>(upper_level);
            // Unsigned subtraction is exact even when signed bounds straddle zero.
            const bool unit_width = static_cast<ulong>(upper) - static_cast<ulong>(lower) == FAST_EVEN_BINS;
            if(m_device.backend_name() == "cuda" && num_levels == FAST_EVEN_BINS + 1u && sample_stride == 1u
               && channel == 0u && row_stride_samples == num_row_samples && unit_width)
            {
                constexpr size_t fast_block_size = BLOCK_SIZE == details::BLOCK_SIZE ? 512u : BLOCK_SIZE;
                histogram_even_contiguous<FAST_EVEN_BINS, fast_block_size>(
                    cmdlist, samples, histogram, lower_level, upper_level, logical_samples);
                return;
            }
        }

        using EvenShader =
            Shader<1, Buffer<SampleT>, Buffer<CounterT>, uint, LevelT, LevelT, uint, uint, uint, uint>;
        auto key       = shader_key<SampleT, CounterT, LevelT>("histogram_even");
        auto shader_it = m_even_shaders.find(key);
        if(shader_it == m_even_shaders.end())
        {
            U<EvenShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<SampleT>  samples_buffer,
                            BufferVar<CounterT> histogram_buffer,
                            UInt                num_levels_var,
                            Var<LevelT>         lower_level_var,
                            Var<LevelT>         upper_level_var,
                            UInt                num_row_samples_var,
                            UInt                row_stride_samples_var,
                            UInt                sample_stride_var,
                            UInt                channel_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt logical_index = dispatch_id().x;
                             UInt row           = logical_index / num_row_samples_var;
                             UInt column        = logical_index % num_row_samples_var;
                             UInt sample_index =
                                 row * row_stride_samples_var + column * sample_stride_var + channel_var;
                             Var<CommonT> sample = cast<CommonT>(samples_buffer.read(sample_index));
                             Var<CommonT> lower  = cast<CommonT>(lower_level_var);
                             Var<CommonT> upper  = cast<CommonT>(upper_level_var);
                             UInt         num_bins = num_levels_var - 1u;
                             $if(sample >= lower & sample < upper)
                             {
                                 UInt bin;
                                 if constexpr(std::is_integral_v<CommonT>)
                                 {
                                     bin = integral_even_bin(sample, lower, upper, num_bins);
                                 }
                                 else
                                 {
                                     bin = cast<uint>((sample - lower) * cast<CommonT>(num_bins)
                                                      / (upper - lower));
                                 }
                                 bin = min(bin, num_bins - 1u);
                                 histogram_buffer.atomic(bin).fetch_add(CounterT{1});
                             };
                         });
            m_even_shaders.try_emplace(key, std::move(shader));
            shader_it = m_even_shaders.find(key);
        }
        auto shader = reinterpret_cast<EvenShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile even-histogram shader.");
        cmdlist << (*shader)(samples,
                             histogram,
                             static_cast<uint>(num_levels),
                             lower_level,
                             upper_level,
                             static_cast<uint>(num_row_samples),
                             static_cast<uint>(row_stride_samples),
                             static_cast<uint>(sample_stride),
                             static_cast<uint>(channel))
                       .dispatch(logical_samples);
    }

    template <NumericT SampleT, NumericT CounterT, NumericT LevelT>
    void histogram_range(CommandList&         cmdlist,
                         BufferView<SampleT>  samples,
                         BufferView<CounterT> histogram,
                         BufferView<LevelT>   levels,
                         size_t               num_levels,
                         size_t               num_row_samples,
                         size_t               num_rows,
                         size_t               row_stride_samples,
                         size_t               sample_stride,
                         size_t               channel)
    {
        using CommonT = std::common_type_t<SampleT, LevelT>;
        static_assert(NumericT<CommonT>);
        LUISA_ASSERT(m_created, "DeviceHistogram::create must be called before dispatch.");
        LUISA_ASSERT(num_levels >= 2u, "A histogram needs at least two levels.");
        LUISA_ASSERT(num_levels <= std::numeric_limits<uint>::max(), "Too many histogram levels.");
        LUISA_ASSERT(histogram.size() >= num_levels - 1u, "Histogram output is too small.");
        LUISA_ASSERT(levels.size() >= num_levels, "Histogram level view is too small.");
        size_t logical_samples = validate_sample_layout(
            samples.size(), num_row_samples, num_rows, row_stride_samples, sample_stride, channel);

        reset_histogram(cmdlist, histogram, num_levels - 1u);
        if(logical_samples == 0u)
        {
            return;
        }

        using RangeShader =
            Shader<1, Buffer<SampleT>, Buffer<CounterT>, Buffer<LevelT>, uint, uint, uint, uint, uint>;
        auto key       = shader_key<SampleT, CounterT, LevelT>("histogram_range");
        auto shader_it = m_range_shaders.find(key);
        if(shader_it == m_range_shaders.end())
        {
            U<RangeShader> shader = nullptr;
            lazy_compile(m_device,
                         shader,
                         [](BufferVar<SampleT>  samples_buffer,
                            BufferVar<CounterT> histogram_buffer,
                            BufferVar<LevelT>   levels_buffer,
                            UInt                num_levels_var,
                            UInt                num_row_samples_var,
                            UInt                row_stride_samples_var,
                            UInt                sample_stride_var,
                            UInt                channel_var) noexcept
                         {
                             set_block_size(BLOCK_SIZE);
                             UInt logical_index = dispatch_id().x;
                             UInt row           = logical_index / num_row_samples_var;
                             UInt column        = logical_index % num_row_samples_var;
                             UInt sample_index =
                                 row * row_stride_samples_var + column * sample_stride_var + channel_var;
                             Var<CommonT> sample = cast<CommonT>(samples_buffer.read(sample_index));
                             Var<CommonT> lower  = cast<CommonT>(levels_buffer.read(0u));
                             Var<CommonT> upper = cast<CommonT>(levels_buffer.read(num_levels_var - 1u));
                             $if(sample >= lower & sample < upper)
                             {
                                 UInt low  = 0u;
                                 UInt high = num_levels_var - 1u;
                                 $while(low + 1u < high)
                                 {
                                     UInt middle = (low + high) / 2u;
                                     Var<CommonT> middle_level = cast<CommonT>(levels_buffer.read(middle));
                                     $if(sample < middle_level)
                                     {
                                         high = middle;
                                     }
                                     $else
                                     {
                                         low = middle;
                                     };
                                 };
                                 histogram_buffer.atomic(low).fetch_add(CounterT{1});
                             };
                         });
            m_range_shaders.try_emplace(key, std::move(shader));
            shader_it = m_range_shaders.find(key);
        }
        auto shader = reinterpret_cast<RangeShader*>(&(*shader_it->second));
        LUISA_ASSERT(shader != nullptr, "Failed to compile range-histogram shader.");
        cmdlist << (*shader)(samples,
                             histogram,
                             levels,
                             static_cast<uint>(num_levels),
                             static_cast<uint>(num_row_samples),
                             static_cast<uint>(row_stride_samples),
                             static_cast<uint>(sample_stride),
                             static_cast<uint>(channel))
                       .dispatch(logical_samples);
    }
};

template <size_t BLOCK_SIZE = details::BLOCK_SIZE, size_t WARP_NUMS = details::WARP_SIZE, size_t ITEMS_PER_THREAD = details::ITEMS_PER_THREAD>
using DeviceDeviceHistogram = DeviceHistogram<BLOCK_SIZE, WARP_NUMS, ITEMS_PER_THREAD>;
}  // namespace luisa::parallel_primitive
