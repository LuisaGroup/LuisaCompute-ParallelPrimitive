#pragma once

#include <luisa/dsl/resource.h>

#include <utility>

namespace luisa::parallel_primitive
{

/// Cache behavior requested for a thread-level buffer load.
///
/// LuisaCompute exposes portable ordinary and volatile buffer reads. CUDA-specific cache
/// placement hints therefore use an ordinary read on backends that do not expose the requested
/// cache policy, while LOAD_CV and LOAD_VOLATILE retain volatile access semantics.
enum class CacheLoadModifier
{
    LOAD_DEFAULT,
    LOAD_CA,
    LOAD_CG,
    LOAD_CS,
    LOAD_CV,
    LOAD_LDG,
    LOAD_VOLATILE,
};

inline constexpr auto LOAD_DEFAULT  = CacheLoadModifier::LOAD_DEFAULT;
inline constexpr auto LOAD_CA       = CacheLoadModifier::LOAD_CA;
inline constexpr auto LOAD_CG       = CacheLoadModifier::LOAD_CG;
inline constexpr auto LOAD_CS       = CacheLoadModifier::LOAD_CS;
inline constexpr auto LOAD_CV       = CacheLoadModifier::LOAD_CV;
inline constexpr auto LOAD_LDG      = CacheLoadModifier::LOAD_LDG;
inline constexpr auto LOAD_VOLATILE = CacheLoadModifier::LOAD_VOLATILE;

template <CacheLoadModifier MODIFIER = CacheLoadModifier::LOAD_DEFAULT, typename T, typename Index>
[[nodiscard]] inline auto ThreadLoad(const compute::BufferVar<T>& input, Index&& index) noexcept
{
    if constexpr(MODIFIER == CacheLoadModifier::LOAD_CV || MODIFIER == CacheLoadModifier::LOAD_VOLATILE)
    {
        return input.volatile_read(std::forward<Index>(index));
    }
    else
    {
        return input.read(std::forward<Index>(index));
    }
}

}  // namespace luisa::parallel_primitive
