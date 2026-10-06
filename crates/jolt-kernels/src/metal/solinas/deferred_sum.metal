#define SOLINAS_DEFERRED_SUM_WORDS 5u

inline void solinas_deferred_atomic_add_5(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * SOLINAS_DEFERRED_SUM_WORDS;
    uint carry = 0u;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0u) {
        atomic_fetch_add_explicit(
            &sums[base + 4u],
            carry,
            memory_order_relaxed);
    }
}

inline SolinasFp128 solinas_deferred_atomic_reduce_5(
    threadgroup atomic_uint* sums,
    uint field)
{
    uint base = field * SOLINAS_DEFERRED_SUM_WORDS;
    SolinasFp128 low;
    for (uint limb = 0u; limb < 4u; limb++) {
        low.limb[limb] = atomic_load_explicit(
            &sums[base + limb],
            memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(
        &sums[base + 4u],
        memory_order_relaxed);

    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0u, canonical.value, low);

    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32);
    return solinas_add(low, correction);
}

// A thread-local sum of 128-bit values, exactly low + overflow * 2^128.
struct SolinasLazySum {
    SolinasFp128 low;
    uint overflow;
};

inline SolinasLazySum solinas_lazy_zero() {
    SolinasLazySum sum = {};
    return sum;
}

inline void solinas_lazy_add(thread SolinasLazySum& sum, SolinasFp128 value) {
    ulong carry = 0ul;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong word = (ulong)sum.low.limb[limb] + (ulong)value.limb[limb] + carry;
        sum.low.limb[limb] = (uint)word;
        carry = word >> 32;
    }
    sum.overflow += (uint)carry;
}

// Every lane of a full SIMD group must call this together (callers run it
// after their row loops; tile widths are SIMD multiples, see
// resolve_threadgroup_width). Lane 0 adds the shuffle-combined total, so a
// field most rows hit costs one threadgroup update per SIMD group.
inline void solinas_deferred_atomic_flush_simd(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasLazySum value,
    uint lane)
{
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasLazySum other;
        other.low.limb = simd_shuffle_down(value.low.limb, offset);
        other.overflow = simd_shuffle_down(value.overflow, offset);
        solinas_lazy_add(value, other.low);
        value.overflow += other.overflow;
    }
    if (lane == 0u) {
        solinas_deferred_atomic_add_5(sums, field, value.low);
        if (value.overflow != 0u) {
            atomic_fetch_add_explicit(
                &sums[field * SOLINAS_DEFERRED_SUM_WORDS + 4u],
                value.overflow,
                memory_order_relaxed);
        }
    }
}
