#pragma once

#include <luisa/ast/external_function.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/resource.h>

#include <type_traits>

namespace luisa::parallel_primitive::details
{

// Only for naturally aligned, self-contained status/value words. These relaxed
// accesses do not publish payload stored elsewhere. Explicit resource usages
// keep the temporary buffer visible to Luisa's dependency tracking.
template <typename Word>
[[nodiscard]] compute::Var<Word> cuda_tile_state_load(const compute::BufferVar<Word>& buffer,
                                                    compute::UInt                  index)
{
    static_assert(std::is_same_v<Word, uint> || std::is_same_v<Word, ulong>);
    static const auto function = luisa::make_shared<compute::ExternalFunction>(
        R"cuda(([](auto buffer, lc_uint index) {
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + index);
    if constexpr (sizeof(*buffer.ptr) == 4) {
        lc_uint value;
#if __CUDA_ARCH__ >= 700
        asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
#else
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
#endif
        return value;
    } else {
        lc_ulong value;
#if __CUDA_ARCH__ >= 700
        asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
#else
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
#endif
        return value;
    }
}))cuda",
        compute::Type::of<Word>(),
        luisa::vector<const compute::Type*>{compute::Type::of<compute::Buffer<Word>>(), compute::Type::of<uint>()},
        luisa::vector<compute::Usage>{compute::Usage::READ, compute::Usage::READ});
    const compute::Expression* arguments[]{buffer.expression(), index.expression()};
    return compute::def<Word>(compute::detail::FunctionBuilder::current()->call(
        compute::Type::of<Word>(), function, luisa::span{arguments}));
}

template <typename Word>
void cuda_tile_state_store(const compute::BufferVar<Word>& buffer,
                           compute::UInt                  index,
                           compute::Expr<Word>            value)
{
    static_assert(std::is_same_v<Word, uint> || std::is_same_v<Word, ulong>);
    static const auto function = luisa::make_shared<compute::ExternalFunction>(
        R"cuda(([](auto buffer, lc_uint index, auto value) {
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + index);
    if constexpr (sizeof(*buffer.ptr) == 4) {
        lc_uint word = value;
#if __CUDA_ARCH__ >= 700
        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(address), "r"(word) : "memory");
#else
        asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(address), "r"(word) : "memory");
#endif
    } else {
        lc_ulong word = value;
#if __CUDA_ARCH__ >= 700
        asm volatile("st.relaxed.gpu.u64 [%0], %1;" :: "l"(address), "l"(word) : "memory");
#else
        asm volatile("st.volatile.global.u64 [%0], %1;" :: "l"(address), "l"(word) : "memory");
#endif
    }
}))cuda",
        nullptr,
        luisa::vector<const compute::Type*>{
            compute::Type::of<compute::Buffer<Word>>(), compute::Type::of<uint>(), compute::Type::of<Word>()},
        luisa::vector<compute::Usage>{compute::Usage::WRITE, compute::Usage::READ, compute::Usage::READ});
    const compute::Expression* arguments[]{buffer.expression(), index.expression(), value.expression()};
    compute::detail::FunctionBuilder::current()->call(function, luisa::span{arguments});
}

}  // namespace luisa::parallel_primitive::details
