#pragma once

#include <luisa/dsl/resource.h>

#include <utility>

namespace luisa::parallel_primitive
{

/// Cache behavior requested for a thread-level buffer store.
///
/// LuisaCompute exposes portable ordinary and volatile buffer writes. CUDA-specific cache
/// placement hints therefore use an ordinary write on backends that do not expose the requested
/// cache policy, while STORE_VOLATILE retains volatile access semantics.
enum class CacheStoreModifier
{
    STORE_DEFAULT,
    STORE_WB,
    STORE_CG,
    STORE_CS,
    STORE_WT,
    STORE_VOLATILE,
};

inline constexpr auto STORE_DEFAULT  = CacheStoreModifier::STORE_DEFAULT;
inline constexpr auto STORE_WB       = CacheStoreModifier::STORE_WB;
inline constexpr auto STORE_CG       = CacheStoreModifier::STORE_CG;
inline constexpr auto STORE_CS       = CacheStoreModifier::STORE_CS;
inline constexpr auto STORE_WT       = CacheStoreModifier::STORE_WT;
inline constexpr auto STORE_VOLATILE = CacheStoreModifier::STORE_VOLATILE;

template <CacheStoreModifier MODIFIER = CacheStoreModifier::STORE_DEFAULT, typename T, typename Index, typename Value>
inline void ThreadStore(const compute::BufferVar<T>& output, Index&& index, Value&& value) noexcept
{
    if constexpr(MODIFIER == CacheStoreModifier::STORE_VOLATILE)
    {
        output.volatile_write(std::forward<Index>(index), std::forward<Value>(value));
    }
    else
    {
        output.write(std::forward<Index>(index), std::forward<Value>(value));
    }
}

}  // namespace luisa::parallel_primitive
