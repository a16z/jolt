struct BytecodeAddressParams { uint prefix, addresses, stages, base_stages, in_len, out_len, group_rows, first, stripes; };
struct BytecodeFinishParams { uint addresses, stages, stripes; };

// The fifth limb retains integer overflow. At most 2^32-1 canonical
// contributions reach any bin, so it cannot wrap. The carry propagation is
// exact even with interleaved writers; each overflowing atomic add contributes
// exactly one carry to the next limb.
#define BYTECODE_ATOMICS(SPACE) \
template <typename F> inline void bytecode_add(SPACE metal::atomic_uint* bin, F value) { \
    uint carry = 0; \
    for (uint limb = 0; limb < 4; limb++) { \
        ulong addend = ulong(value.limb[limb]) + carry; uint low = uint(addend); \
        uint old = metal::atomic_fetch_add_explicit(&bin[limb], low, metal::memory_order_relaxed); \
        carry = uint(addend >> 32) | uint(old > 0xffffffffu - low); \
    } \
    if (carry) metal::atomic_fetch_add_explicit(&bin[4], carry, metal::memory_order_relaxed); \
} \
template <uint C> inline jolt::Fp128<C> bytecode_reduce(SPACE metal::atomic_uint* bin, jolt::Fp128<C>) { \
    using F = jolt::Fp128<C>; uint4 low; \
    for (uint limb = 0; limb < 4; limb++) low[limb] = metal::atomic_load_explicit(&bin[limb], metal::memory_order_relaxed); \
    uint overflow = metal::atomic_load_explicit(&bin[4], metal::memory_order_relaxed); \
    return F{jolt::fp128_detail::fold2_canonicalize<C>(low, 0)} + F::from_u64(ulong(overflow) * C); \
}
BYTECODE_ATOMICS(threadgroup)
BYTECODE_ATOMICS(device)
#undef BYTECODE_ATOMICS

template <typename F> kernel void bytecode_pushforward(
    device const ulong4* rows [[buffer(0)]], device const ulong* magnitude [[buffer(1)]],
    device const ulong* signs [[buffer(2)]], device const F* e_lo [[buffer(3)]],
    device const F* e_hi [[buffer(4)]], device metal::atomic_uint* output [[buffer(5)]],
    constant BytecodeAddressParams& p [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup metal::atomic_uint keys[512];
    threadgroup metal::atomic_uint sums[2 * 512 * 5];
    for (uint i = lane; i < 512; i += 256) metal::atomic_store_explicit(&keys[i], 0xffffffffu, metal::memory_order_relaxed);
    for (uint i = lane; i < 2 * 512 * 5; i += 256) metal::atomic_store_explicit(&sums[i], 0, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint row = group * p.group_rows + lane;
    if (lane < p.group_rows && row < p.prefix) {
        uint pc = uint(rows[row].z & 0x00fffffffffffffful);
        uint slot = (pc * 2654435761u) & 511;
        while (true) {
            uint expected = 0xffffffffu;
            if (metal::atomic_compare_exchange_weak_explicit(&keys[slot], &expected, pc, metal::memory_order_relaxed, metal::memory_order_relaxed) || expected == pc) break;
            // Weak CAS may fail spuriously on an empty slot; probing still
            // terminates because at least half of the table remains empty.
            slot = (slot + 1) & 511;
        }
        F increment = F::from_u64(magnitude[row]);
        if ((signs[row >> 6] >> (row & 63)) & 1ul) increment = F::zero() - increment;
        for (uint offset = 0; offset < 2 && p.first + offset < p.stages; offset++) {
            uint stage = p.first + offset;
            F value = e_lo[ulong(stage) * p.in_len + row % p.in_len];
            if (stage >= p.base_stages) value = value * increment;
            bytecode_add(&sums[(offset * 512 + slot) * 5], value);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint high = (group * p.group_rows) / p.in_len;
    for (uint slot = lane; slot < 512; slot += 256) {
        uint pc = metal::atomic_load_explicit(&keys[slot], metal::memory_order_relaxed);
        if (pc == 0xffffffffu) continue;
        for (uint offset = 0; offset < 2 && p.first + offset < p.stages; offset++) {
            uint stage = p.first + offset;
            F value = bytecode_reduce(&sums[(offset * 512 + slot) * 5], F::zero()) * e_hi[ulong(stage) * p.out_len + high];
            ulong bin = ((ulong(group % p.stripes) * p.stages + stage) * p.addresses + pc) * 5;
            bytecode_add(&output[bin], value);
        }
    }
}

template <typename F> kernel void bytecode_pushforward_finish(
    device metal::atomic_uint* bins [[buffer(0)]], device F* output [[buffer(1)]],
    constant BytecodeFinishParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ulong stride = ulong(p.addresses) * p.stages;
    if (i >= stride) return;
    F total = F::zero();
    for (uint stripe = 0; stripe < p.stripes; stripe++) total = total + bytecode_reduce(&bins[(ulong(stripe) * stride + i) * 5], F::zero());
    output[i] = total;
}
