// Host-generated geometry and explicit suffix opcodes; protocol enum ordinals
// are never part of this interface. Every histogram lane has 256 bins.
struct AddressParams {
    uint records, prefix, suffix_bits, upper_bits, condense, previous_shift;
    uint caches, output_lanes, canonical;
};
struct AddressJob {
    uint start, end, raf_flag, emit_raf, suffix_count;
    uint opcodes[3];
    uint outputs[6];
};
struct AddressWindow { uint jobs, first; };
struct AddressCacheOutputs { uint lanes[5]; };

inline ulong address_mask(uint bits) { return bits >= 64 ? ~0ul : (bits == 0 ? 0ul : (1ul << bits) - 1); }
inline ulong2 address_suffix(ulong2 key, uint bits) {
    return ulong2(key.x & address_mask(bits), key.y & address_mask(bits > 64 ? bits - 64 : 0));
}
inline uint address_chunk(ulong2 key, uint shift) {
    return uint(shift < 64 ? key.x >> shift : key.y >> (shift - 64)) & (ADDRESS_BINS - 1);
}
inline ulong address_even(ulong x) {
    x &= 0x5555555555555555ul;
    x = (x | (x >> 1)) & 0x3333333333333333ul;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00fful;
    x = (x | (x >> 8)) & 0x0000ffff0000fffful;
    return (x | (x >> 16)) & 0xfffffffful;
}
inline ulong2 address_operands(ulong2 key) {
    return ulong2(address_even(key.x >> 1) | (address_even(key.y >> 1) << 32),
                  address_even(key.x) | (address_even(key.y) << 32));
}
inline bool address_cached(uint bits, constant AddressParams& p) {
    return bits < p.suffix_bits || (bits <= p.suffix_bits && p.upper_bits == 0);
}
inline uint address_leading_ones(ulong value, uint bits) {
    return bits == 0 ? 0u : uint(metal::clz(~(value << (64 - bits))));
}
inline ulong address_pext(ulong x, ulong y) {
    if (!y) return 0;
    uint trailing = uint(metal::ctz(y));
    ulong normalized = y >> trailing;
    if ((normalized & (normalized + 1)) == 0) return (x >> trailing) & normalized;
    ulong out = 0;
    for (uint k = 0; y; k++, y &= y - 1) out |= ((x >> uint(metal::ctz(y))) & 1ul) << k;
    return out;
}
inline ulong address_window_sign(ulong x, ulong y) {
    return y == 0 ? 0 : (x >> (63 - uint(metal::clz(y)))) & 1ul;
}
inline ulong address_scalar(uint opcode, ulong2 suffix, ulong2 operands, uint bits) {
    uint operand_bits = bits / 2;
    uint trailing = metal::min(uint(metal::ctz(operands.y)), operand_bits);
    uint leading = address_leading_ones(operands.y, operand_bits);
    switch (opcode) {
        case 0: return 1;
        case 1: return operands.x & operands.y;
        case 2: return operands.x & ~operands.y;
        case 3: return operands.x ^ operands.y;
        case 4: return operands.x | operands.y;
        case 5: return operands.y;
        case 6: return suffix.y;
        case 7: return suffix.x;
        case 8: return suffix.x & 0xfffffffful;
        case 9: return ulong(operands.x < operands.y);
        case 10: return ulong(operands.x > operands.y);
        case 11: return ulong(operands.x == operands.y);
        case 12: return ulong(operands.x == 0);
        case 13: return ulong(operands.y == 0);
        case 14: return bits == 0 ? 1 : suffix.x & 1ul;
        case 15: return ulong(operands.x == 0 && operands.y == address_mask(operand_bits));
        case 16: return 1ul << (suffix.x & 63ul);
        case 17: return 1ul << (suffix.x & 31ul);
        case 18: return ulong(uint(operands.y));
        case 19: return ulong(suffix.y == 0);
        case 20: return bits < 32 ? 1 : ((suffix.x >> 31) & 1ul) * 0xffffffff00000000ul;
        case 21: return bits < 64 ? 1 : ((suffix.x >> 62) & 1ul) * 0xffffffff00000000ul;
        case 22: return trailing == 0 ? 0 : ~address_mask(64 - trailing);
        case 23: return bits == 0 ? 1 : 1ul << (63 - uint(suffix.x & 63ul));
        case 24: return trailing >= 64 ? 0 : operands.x >> trailing;
        case 25: return 1ul << leading;
        case 26: return leading >= 64 ? 0 : (operands.x & ~operands.y) << leading;
        case 27: return trailing >= 32 ? 0 : ulong(uint(operands.x) >> trailing);
        case 28: return 1ul << address_leading_ones(operands.y & 0xfffffffful, metal::min(operand_bits, 32u));
        case 29: {
            uint shift = address_leading_ones(operands.y & 0xfffffffful, metal::min(operand_bits, 32u));
            return shift >= 32 ? 0 : ulong((uint(operands.x) & ~uint(operands.y)) << shift);
        }
        // Rust's u32 shift uses the low five shift bits in release builds;
        // valid W-mask witnesses keep this exponent below 32.
        case 30: return ulong(1u << (leading & 31));
        case 31: {
            if (bits == 0) return 0;
            ulong fill = 0;
            if (bits >= 64) {
                if (((operands.x >> 31) & 1ul) == 0) return 0;
                fill = 0xffffffff00000000ul;
            }
            uint count = metal::min(operand_bits, 32u);
            for (uint offset = 0; offset < count; offset++) {
                uint position = 32 - count + offset;
                if (position) fill += (1ul - ((operands.y >> (count - 1 - offset)) & 1ul)) << position;
            }
            return fill;
        }
        case 32: return bits < 64 ? 0 : ((operands.x >> 31) & 1ul) * (operands.y & 1ul);
        case 33: return 1ul << (8 * uint(suffix.x & 7ul));
        case 34: return 1ul << (8 * uint(suffix.x & 6ul));
        case 35: return 1ul << (8 * uint(suffix.x & 4ul));
        case 36: return (operands.x & 0xfful) << (8 * uint(operands.y & 7ul));
        case 37: return (operands.x & 0xfffful) << (8 * uint(operands.y & 6ul));
        case 38: return (operands.x & 0xfffffffful) << (8 * uint(operands.y & 4ul));
        case 39: return 1ul << (8 * uint(operands.y & 7ul));
        case 40: return 1ul << (8 * uint(operands.y & 6ul));
        case 41: return 1ul << (8 * uint(operands.y & 4ul));
        case 42: return suffix.x & 7ul;
        case 43: return suffix.x & 6ul;
        case 44: return suffix.x & 4ul;
        case 45: return address_pext(operands.x, operands.y);
        case 46: return 1ul << uint(metal::popcount(operands.y));
        case 47: return address_window_sign(operands.x, operands.y);
        case 48: return address_window_sign(operands.x, operands.y) << uint(metal::popcount(operands.y));
        case 49: {
            ulong x = suffix.x;
            x = ((x & 0x00ff00ff00ff00fful) << 8) | ((x >> 8) & 0x00ff00ff00ff00fful);
            return ((x & 0x0000ffff0000fffful) << 16) | ((x >> 16) & 0x0000ffff0000fffful);
        }
        case 50: case 51: case 52: case 53: {
            uint shift = opcode == 50 ? 16 : (opcode == 51 ? 24 : (opcode == 52 ? 32 : 63));
            ulong x = operands.x ^ operands.y;
            return (x >> shift) | (x << (64 - shift));
        }
        case 54: case 55: case 56: case 57: case 58: case 59: case 60: {
            uint shift = opcode == 54 ? 16 : (opcode == 55 ? 12 : (opcode == 56 ? 8 : (opcode == 57 ? 7 : (opcode == 58 ? 22 : (opcode == 59 ? 19 : 6)))));
            uint x = uint(operands.x ^ operands.y);
            return ulong((x >> shift) | (x << (32 - shift)));
        }
        case 61: return (operands.x ^ (operands.y << 1)) & address_mask(operand_bits) & ~1ul;
        case 62: return operand_bits == 0 ? 0 : (operands.y >> (operand_bits - 1)) & 1ul;
        case 63: return bits == 0 ? 0 : operands.x & 1ul;
    }
    return 0; // Host admission rejects every unsupported opcode.
}
template <typename F> inline F address_from_u128(ulong2 value) {
    // A suffix has at most 120 bits, so this input is already canonical.
    return F{uint4(uint(value.x), uint(value.x >> 32), uint(value.y), uint(value.y >> 32))};
}

// Instantiated only by the existing Metal parity test, against the canonical
// Rust scalar definitions at every address-phase boundary.
template <typename F>
kernel void address_suffix_oracle(device const ulong4* input [[buffer(0)]],
    device ulong* output [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    ulong4 row = input[i];
    ulong2 suffix = address_suffix(row.xy, uint(row.z));
    output[i] = address_scalar(uint(row.w), suffix, address_operands(suffix), uint(row.z));
}

template <typename F>
kernel void address_condense(
    device const uint2* records [[buffer(0)]], device F* weights [[buffer(1)]],
    device const ulong4* rows [[buffer(2)]], constant ulong4& tail [[buffer(3)]],
    device const F* previous [[buffer(4)]], constant AddressParams& p [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.records || !p.condense) return;
    uint row = records[i].x;
    ulong2 key = (row < p.prefix ? rows[row] : tail).xy;
    weights[i] = weights[i] * previous[address_chunk(key, p.previous_shift)];
}

template <typename F>
kernel void address_cache(
    device const uint2* records [[buffer(0)]], device const F* weights [[buffer(1)]],
    device const ulong4* rows [[buffer(2)]], constant ulong4& tail [[buffer(3)]],
    device const F* previous [[buffer(4)]], constant AddressParams& p [[buffer(5)]],
    device const uint4* metadata [[buffer(6)]], device F* cache [[buffer(7)]],
    device uint* valid [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint4 meta = metadata[group];
    if (!address_cached(meta.z, p)) return;
    if (valid[group]) {
        if (tid < 3 && p.condense) cache[3 * group + tid] = cache[3 * group + tid] * previous[0];
        return;
    }
    using Acc = jolt::Accumulator<F>;
    threadgroup Acc scratch[3][32];
    Acc sums[3];
    for (uint k = 0; k < 3; k++) sums[k] = Acc::zero();
    for (uint i = meta.x + tid; i < meta.y; i += 256) {
        uint row = records[i].x;
        ulong2 key = address_suffix((row < p.prefix ? rows[row] : tail).xy, p.suffix_bits);
        F weight = weights[i];
        sums[0].add(weight);
        sums[1].fmadd(weight, address_from_u128<F>(key));
        sums[2].fmadd(weight, F::from_u64(key.x));
    }
    for (uint k = 0; k < 3; k++) {
        Acc sum = jolt::threadgroup_merge(sums[k], scratch[k], lane, simdgroup, 8, 32);
        if (tid == 0) cache[3 * group + k] = sum.reduce();
    }
    if (tid == 0) valid[group] = 1;
}

template <typename F>
inline void address_atomic_add(threadgroup metal::atomic_uint* bins, uint field, F value) {
    uint carry = 0;
    for (uint limb = 0; limb < 4; limb++) {
        ulong addend = ulong(value.limb[limb]) + carry;
        uint low = uint(addend);
        uint old = metal::atomic_fetch_add_explicit(&bins[5 * field + limb], low, metal::memory_order_relaxed);
        carry = uint(addend >> 32) | uint(old > 0xffffffffu - low);
    }
    if (carry) metal::atomic_fetch_add_explicit(&bins[5 * field + 4], carry, metal::memory_order_relaxed);
}
template <uint C>
inline jolt::Fp128<C> address_atomic_reduce(threadgroup metal::atomic_uint* bins, uint field, jolt::Fp128<C>) {
    using F = jolt::Fp128<C>;
    uint4 low;
    for (uint limb = 0; limb < 4; limb++) low[limb] = metal::atomic_load_explicit(&bins[5 * field + limb], metal::memory_order_relaxed);
    uint overflow = metal::atomic_load_explicit(&bins[5 * field + 4], metal::memory_order_relaxed);
    // At most 65,536 canonical contributions: the exact sum is <2^144.
    // overflow <=65,535, hence overflow*C fits u64 for the current field.
    return F{jolt::fp128_detail::fold2_canonicalize<C>(low, 0)} + F::from_u64(ulong(overflow) * C);
}

template <typename F>
kernel void address_histogram(
    device const uint2* records [[buffer(0)]], device const F* weights [[buffer(1)]],
    device const ulong4* rows [[buffer(2)]], constant ulong4& tail [[buffer(3)]],
    device const uint4* metadata [[buffer(4)]], device const AddressJob* jobs [[buffer(5)]],
    device F* partials [[buffer(6)]], constant AddressParams& p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    // This is the only threadgroup allocation: 6*256*5*4=30,720 bytes.
    // Pipeline static storage is checked against the device limit at admission.
    threadgroup metal::atomic_uint bins[6 * ADDRESS_BINS * 5];
    for (uint i = tid; i < 6 * ADDRESS_BINS * 5; i += 256) metal::atomic_store_explicit(&bins[i], 0, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    AddressJob job = jobs[group];
    for (uint i = job.start + tid; i < job.end; i += 256) {
        uint2 record = records[i];
        if (record.y != 0xffffffffu && address_cached(metadata[record.y].z, p)) continue;
        ulong2 key = (record.x < p.prefix ? rows[record.x] : tail).xy;
        uint chunk = address_chunk(key, p.suffix_bits);
        ulong2 suffix = address_suffix(key, p.suffix_bits);
        ulong2 operands = address_operands(suffix);
        F weight = weights[i];
        if (job.emit_raf) {
            address_atomic_add(bins, chunk, weight);
            if (!job.raf_flag) {
                if (operands.x) address_atomic_add(bins, ADDRESS_BINS + chunk, mul_u64(weight, operands.x));
                if (operands.y) address_atomic_add(bins, 2 * ADDRESS_BINS + chunk, mul_u64(weight, operands.y));
            } else {
                if (suffix.x || suffix.y) address_atomic_add(bins, ADDRESS_BINS + chunk, weight * address_from_u128<F>(suffix));
                if (p.canonical && (p.upper_bits == 0 || suffix.y == address_mask(p.upper_bits)))
                    address_atomic_add(bins, 2 * ADDRESS_BINS + chunk, weight);
            }
        }
        for (uint k = 0; k < job.suffix_count; k++) {
            ulong scalar = address_scalar(job.opcodes[k], suffix, operands, p.suffix_bits);
            if (scalar) address_atomic_add(bins, (3 + k) * ADDRESS_BINS + chunk, scalar == 1 ? weight : mul_u64(weight, scalar));
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint i = tid; i < 6 * ADDRESS_BINS; i += 256)
        partials[group * 6 * ADDRESS_BINS + i] = address_atomic_reduce(bins, i, F::zero());
}

template <typename F>
kernel void address_combine(
    device const AddressJob* jobs [[buffer(0)]], device const F* partials [[buffer(1)]],
    device F* output [[buffer(2)]], constant AddressWindow& window [[buffer(3)]],
    constant AddressParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.output_lanes * ADDRESS_BINS) return;
    uint lane = i / ADDRESS_BINS, bin = i % ADDRESS_BINS;
    F sum = window.first ? F::zero() : output[i];
    for (uint job = 0; job < window.jobs; job++)
        for (uint k = 0; k < 6; k++)
            if (jobs[job].outputs[k] == lane) sum = sum + partials[(job * 6 + k) * ADDRESS_BINS + bin];
    output[i] = sum;
}

template <typename F>
kernel void address_cache_finish(
    device const uint4* metadata [[buffer(0)]], device const F* cache [[buffer(1)]],
    device F* output [[buffer(2)]], constant AddressCacheOutputs& outputs [[buffer(3)]],
    constant AddressParams& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (outputs.lanes[group] == 0xffffffffu || (group == 2 && (!p.canonical || p.upper_bits != 0))) return;
    using Acc = jolt::Accumulator<F>;
    threadgroup Acc scratch[32];
    Acc sum = Acc::zero();
    uint value = group == 1 ? 1 : (group == 4 ? 2 : 0);
    for (uint i = tid; i < p.caches; i += 256)
        if (address_cached(metadata[i].z, p)) sum.add(cache[3 * i + value]);
    Acc total = jolt::threadgroup_merge(sum, scratch, lane, simdgroup, 8, 32);
    if (tid == 0) {
        uint i = outputs.lanes[group] * ADDRESS_BINS;
        output[i] = output[i] + total.reduce();
    }
}
