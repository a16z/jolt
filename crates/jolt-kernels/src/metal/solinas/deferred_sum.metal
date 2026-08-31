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
