/*
 * Fixed-size keys for the per-module shader caches. Data types and boolean
 * options use stable process-wide addresses. Operators use the hash of their
 * generated callable, since objects of the same C++ type can emit different
 * code. The key representation itself requires no string allocation.
 */

#pragma once

  #include <array>
  #include <cstddef>
  #include <cstdint>
  #include <cstdio>

  #include <luisa/ast/type.h>
  #include <luisa/core/stl/format.h>
  #include <luisa/core/stl/string.h>
  #include <luisa/core/stl/unordered_map.h>
  #include <luisa/dsl/resource.h>

namespace luisa::parallel_primitive::detail
{

    /// Pointer identities and generated-function hashes occupy separate slot
    /// kinds, even when their numeric values happen to match.
    struct ShaderCacheKey {
        static constexpr int MAX_SLOTS = 6;
        std::array<uint64_t, MAX_SLOTS> values{};
        uint8_t count = 0;
        uint8_t hash_mask = 0;

        [[nodiscard]] bool operator==(const ShaderCacheKey& rhs) const noexcept
        {
            if(count != rhs.count || hash_mask != rhs.hash_mask) {
                return false;
            }
            for(int i = 0; i < count; ++i) {
                if(values[i] != rhs.values[i]) {
                    return false;
                }
            }
            return true;
        }
    };

    struct ShaderCacheKeyHash {
        [[nodiscard]] size_t operator()(const ShaderCacheKey& k) const noexcept
        {
            // splitmix64 avalanche over the slot values and kinds.
            auto mix = [](uint64_t x) noexcept {
                x += 0x9E3779B97F4A7C15ull;
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
                x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
                return x ^ (x >> 31);
            };
            uint64_t h = static_cast<uint64_t>(k.count)
                         | (static_cast<uint64_t>(k.hash_mask) << 8u);
            for(int i = 0; i < k.count; ++i) {
                h = mix(h ^ k.values[i]);
            }
            return static_cast<size_t>(h);
        }
    };

    struct ShaderCacheKeyComponent {
        uint64_t value;
        bool is_function_hash;
    };

    [[nodiscard]] inline ShaderCacheKeyComponent shader_cache_key_component(const void* identity) noexcept
    {
        static_assert(sizeof(uintptr_t) <= sizeof(uint64_t));
        return {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(identity)), false};
    }

    [[nodiscard]] inline ShaderCacheKeyComponent shader_cache_key_component(uint64_t function_hash) noexcept
    {
        return {function_hash, true};
    }

    /// Build a key from identity pointers and full-width generated-function hashes.
    template <typename... Ts>
    [[nodiscard]] inline ShaderCacheKey make_shader_cache_key(Ts... components) noexcept
    {
        static_assert(sizeof...(Ts) <= ShaderCacheKey::MAX_SLOTS);
        ShaderCacheKey key;
        std::array<ShaderCacheKeyComponent, sizeof...(Ts)> raw{shader_cache_key_component(components)...};
        key.count = static_cast<uint8_t>(sizeof...(Ts));
        for(uint8_t i = 0; i < key.count; ++i) {
            key.values[i] = raw[i].value;
            if(raw[i].is_function_hash) {
                key.hash_mask |= static_cast<uint8_t>(1u << i);
            }
        }
        return key;
    }

    /// Identity pointer for a data element type (the process-wide type-registry
    /// singleton returned by Type::of<T>()).  Distinct descriptions give
    /// distinct pointers, so this discriminates exactly the same set of types the
    /// previous description-based string key did -- without building any string.
    template <typename T>
    [[nodiscard]] inline const void* type_ptr() noexcept
    {
        return static_cast<const void*>(luisa::compute::Type::of<T>());
    }

    /// One of two distinct, stable addresses used to fold a boolean template
    /// parameter (e.g. KEY_ONLY / IS_DESCENDING) into a cache key.  Inline
    /// function-local statics have a single instance across TUs, so each value
    /// always maps to the same pointer.
    [[nodiscard]] inline const void* bool_ptr(bool b) noexcept
    {
        static const char k_false = 0;
        static const char k_true  = 1;
        return b ? static_cast<const void*>(&k_true) : static_cast<const void*>(&k_false);
    }

    /// Value type stored in every shader cache: a type-erased compiled shader
    /// Resource (reinterpreted back to its concrete Shader<K> at the call site).
    using ShaderCacheMap =
        luisa::unordered_map<ShaderCacheKey,
                             luisa::shared_ptr<luisa::compute::Resource>,
                             ShaderCacheKeyHash>;

    /// Stable string form of a key, for the few call sites (inherited from the
    /// string-keyed design) that embed the key in a larger luisa::format key.
    [[nodiscard]] inline luisa::string to_string(const ShaderCacheKey& k)
    {
        auto out = luisa::string{"ShaderCacheKey("};
        for(int i = 0; i < k.count; ++i)
        {
            if(i != 0) { out += ','; }
            char buf[2 + 2 * sizeof(uint64_t) + 1];
            std::snprintf(buf, sizeof(buf), "%c:%016llx",
                          (k.hash_mask & (1u << i)) != 0 ? 'h' : 'p',
                          static_cast<unsigned long long>(k.values[i]));
            out += buf;
        }
        out += ')';
        return out;
    }

}  // namespace luisa::parallel_primitive::detail

// fmt support so luisa::format can embed a ShaderCacheKey directly; keeps the
// string-keyed call sites compiling against the fixed-size key type.
template <>
struct fmt::formatter<luisa::parallel_primitive::detail::ShaderCacheKey>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const luisa::parallel_primitive::detail::ShaderCacheKey& k,
                FormatContext& ctx) const
    {
        auto out = ctx.out();
        auto s   = luisa::parallel_primitive::detail::to_string(k);
        for(char c : s) { *out++ = c; }
        return out;
    }
};
