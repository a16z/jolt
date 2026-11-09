struct BytecodeCycleParams {
    uint source_prefix, previous_prefix, prefix, columns, ra_width, inc_width, radix;
    uint ra_mode, inc_mode, coefficient_mode, in_len, out_len, base_terms, fused_terms, degree;
};

template <typename F>
F bytecode_ra_gather(device const ulong4* rows, device const F* tables,
                     device const uint* shifts, constant ulong4& source_tail,
                     constant BytecodeCycleParams& p, ulong row, uint column) {
    F value = F::zero();
    for (uint branch = 0; branch < p.ra_width; branch++) {
        ulong index = row * p.ra_width + branch;
        ulong pc = index < p.source_prefix ? rows[index].z & 0x00fffffffffffffful : source_tail.x;
        uint hot = uint(pc >> shifts[column]) & (p.radix - 1);
        value = value + tables[(ulong(column) * p.ra_width + branch) * p.radix + hot];
    }
    return value;
}

template <typename F>
F bytecode_inc_gather(device const ulong* magnitude, device const ulong* signs,
                      device const F* weights, constant ulong4& source_tail,
                      constant BytecodeCycleParams& p, ulong row) {
    F value = F::zero();
    for (uint branch = 0; branch < p.inc_width; branch++) {
        ulong index = row * p.inc_width + branch;
        ulong mag = index < p.source_prefix ? magnitude[index] : source_tail.y;
        bool negative = index < p.source_prefix ? ((signs[index >> 6] >> (index & 63)) & 1ul) != 0 : source_tail.z != 0;
        F term = mul_u64(weights[branch], mag);
        value = negative ? value - term : value + term;
    }
    return value;
}

template <typename F>
F bytecode_coefficient(device const F* e_lo, device const F* e_hi,
                        constant BytecodeCycleParams& p, ulong row, uint column) {
    F value = F::zero();
    uint first = column == 0 ? 0 : p.base_terms;
    uint end = column == 0 ? p.base_terms : p.base_terms + p.fused_terms;
    for (uint stage = first; stage < end; stage++)
        value = value + e_lo[ulong(stage) * p.in_len + row % p.in_len] * e_hi[ulong(stage) * p.out_len + row / p.in_len];
    return value;
}

// The increment stays a separate linear factor. Multiplying it into C before
// interpolation would drop the quadratic coefficient and change the relation.
template <typename F>
void bytecode_product(thread F* ra_lo, thread F* ra_hi, F c0, F c1, F c2,
                      constant BytecodeCycleParams& p, thread F* output) {
    F scale = F::one(), values[8], slopes[8]; uint degree = 0;
    for (uint i = 0; i < p.columns; i++) {
        F slope = ra_hi[i] - ra_lo[i];
        if (metal::all(slope.limb == uint4(0))) scale = scale * ra_lo[i];
        else { values[degree] = ra_lo[i]; slopes[degree] = slope; degree++; }
    }
    F second = c2 + c2;
    F coefficient = c0, next = c0 + c1 + c1 + second + second;
    F delta = c1 + second + second + c2;
    for (uint slot = 0; slot < p.degree; slot++) {
        F product = scale;
        for (uint i = 0; i < degree; i++) product = product * values[i];
        output[slot] = product * coefficient;
        for (uint i = 0; i < degree; i++) values[i] = values[i] + slopes[i] + (slot == 0 ? slopes[i] : F::zero());
        coefficient = next; next = next + delta; delta = delta + second;
    }
}

inline ulong bytecode_pc(device const ulong4* rows, constant ulong4& tail,
                         constant BytecodeCycleParams& p, ulong index) {
    return index < p.source_prefix ? rows[index].z & 0x00fffffffffffffful : tail.x;
}
inline uint bytecode_pair_hash(device const ulong4* rows, constant ulong4& tail,
                               constant BytecodeCycleParams& p, uint pair) {
    uint hash = 2166136261u;
    for (uint branch = 0; branch < 2 * p.ra_width; branch++) {
        ulong pc = bytecode_pc(rows, tail, p, ulong(pair) * 2 * p.ra_width + branch);
        hash = (hash ^ uint(pc)) * 16777619u;
        hash = (hash ^ uint(pc >> 32)) * 16777619u;
    }
    return hash & 127;
}
inline bool bytecode_same_pair(device const ulong4* rows, constant ulong4& tail,
                               constant BytecodeCycleParams& p, uint a, uint b) {
    for (uint branch = 0; branch < 2 * p.ra_width; branch++)
        if (bytecode_pc(rows, tail, p, ulong(a) * 2 * p.ra_width + branch) !=
            bytecode_pc(rows, tail, p, ulong(b) * 2 * p.ra_width + branch)) return false;
    return true;
}

template <typename F> kernel void bytecode_cycle_round(
    device const ulong4* rows [[buffer(0)]], device const ulong* magnitude [[buffer(1)]],
    device const ulong* signs [[buffer(2)]], device const F* ra_tables [[buffer(3)]],
    device const uint* shifts [[buffer(4)]], device const F* weights [[buffer(5)]],
    device const F* previous_ra [[buffer(6)]], device const F* previous_inc [[buffer(7)]],
    device const F* previous_coefficients [[buffer(8)]], device F* next_ra [[buffer(9)]],
    device F* next_inc [[buffer(10)]], device F* next_coefficients [[buffer(11)]],
    device const F* e_lo [[buffer(12)]], device const F* e_hi [[buffer(13)]],
    device const F* tails [[buffer(14)]], device F* partials [[buffer(15)]],
    constant BytecodeCycleParams& p [[buffer(16)]], constant ulong4& source_tail [[buffer(17)]],
    constant F& challenge [[buffer(18)]], uint group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    threadgroup Acc scratch[10][2];
    threadgroup metal::atomic_uint representatives[128];
    threadgroup metal::atomic_uint coefficients[3 * 128 * 5];
    // Equal ordered PC tuples have equal RA factors; only their independent
    // quadratic coefficients are added. A complete tuple comparison handles
    // hash collisions. No assumption is made about increment equality.
    if (p.ra_mode == 0) {
        for (uint i = local; i < 128; i += 64) metal::atomic_store_explicit(&representatives[i], 0xffffffffu, metal::memory_order_relaxed);
        for (uint i = local; i < 3 * 128 * 5; i += 64) metal::atomic_store_explicit(&coefficients[i], 0, metal::memory_order_relaxed);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    Acc sums[10]; for (uint i = 0; i < p.degree; i++) sums[i] = Acc::zero();
    for (uint offset = local; offset < 64; offset += 64) {
        ulong pair = ulong(group) * 64 + offset;
        if (pair >= (ulong(p.prefix) + 1) / 2) break;
        F ra_lo[8], ra_hi[8], coeff_lo[2], coeff_hi[2], inc[2];
        for (uint side = 0; side < 2; side++) {
            ulong row = 2 * pair + side;
            if (p.ra_mode != 0) for (uint column = 0; column < p.columns; column++) {
                F value = tails[column];
                if (row < p.prefix) {
                    if (p.ra_mode == 0) value = bytecode_ra_gather(rows, ra_tables, shifts, source_tail, p, row, column);
                    else {
                        F a, b;
                        if (p.ra_mode == 1) {
                            a = bytecode_ra_gather(rows, ra_tables, shifts, source_tail, p, 2 * row, column);
                            b = bytecode_ra_gather(rows, ra_tables, shifts, source_tail, p, 2 * row + 1, column);
                        } else {
                            a = previous_ra[2 * row * p.columns + column];
                            b = 2 * row + 1 < p.previous_prefix ? previous_ra[(2 * row + 1) * p.columns + column] : tails[column];
                        }
                        value = a + challenge * (b - a);
                        next_ra[row * p.columns + column] = value;
                    }
                }
                if (side == 0) ra_lo[column] = value; else ra_hi[column] = value;
            }
            F increment = tails[p.columns];
            if (row < p.prefix) {
                if (p.inc_mode == 0) increment = bytecode_inc_gather(magnitude, signs, weights, source_tail, p, row);
                else {
                    F a, b;
                    if (p.inc_mode == 1) {
                        a = bytecode_inc_gather(magnitude, signs, weights, source_tail, p, 2 * row);
                        b = bytecode_inc_gather(magnitude, signs, weights, source_tail, p, 2 * row + 1);
                    } else {
                        a = previous_inc[2 * row];
                        b = 2 * row + 1 < p.previous_prefix ? previous_inc[2 * row + 1] : tails[p.columns];
                    }
                    increment = a + challenge * (b - a); next_inc[row] = increment;
                }
            }
            inc[side] = increment;
            for (uint column = 0; column < 2; column++) {
                F value;
                if (p.coefficient_mode < 2) {
                    value = bytecode_coefficient(e_lo, e_hi, p, row, column);
                    if (column == 0 && row == 0) value = value + tails[p.columns + 5];
                } else if (row >= p.prefix) value = tails[p.columns + 3 + column];
                else {
                    F a = previous_coefficients[4 * row + column];
                    F b = 2 * row + 1 < p.previous_prefix ? previous_coefficients[4 * row + 2 + column] : tails[p.columns + 1 + column];
                    value = a + challenge * (b - a);
                }
                if (p.coefficient_mode > 0 && row < p.prefix) next_coefficients[2 * row + column] = value;
                if (side == 0) coeff_lo[column] = value; else coeff_hi[column] = value;
            }
        }
        F df = coeff_hi[1] - coeff_lo[1], di = inc[1] - inc[0];
        F c0 = coeff_lo[0] + inc[0] * coeff_lo[1];
        F c1 = coeff_hi[0] - coeff_lo[0] + inc[0] * df + di * coeff_lo[1];
        F c2 = di * df;
        if (p.ra_mode == 0) {
            uint slot = bytecode_pair_hash(rows, source_tail, p, uint(pair));
            while (true) {
                uint old = 0xffffffffu;
                if (metal::atomic_compare_exchange_weak_explicit(&representatives[slot], &old, uint(pair), metal::memory_order_relaxed, metal::memory_order_relaxed) ||
                    (old != 0xffffffffu && bytecode_same_pair(rows, source_tail, p, old, uint(pair)))) break;
                slot = (slot + 1) & 127;
            }
            bytecode_add(&coefficients[(3 * slot) * 5], c0);
            bytecode_add(&coefficients[(3 * slot + 1) * 5], c1);
            bytecode_add(&coefficients[(3 * slot + 2) * 5], c2);
        } else {
            F output[10]; bytecode_product(ra_lo, ra_hi, c0, c1, c2, p, output);
            for (uint i = 0; i < p.degree; i++) sums[i].add(output[i]);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (p.ra_mode == 0) {
        for (uint slot = local; slot < 128; slot += 64) {
            uint pair = metal::atomic_load_explicit(&representatives[slot], metal::memory_order_relaxed);
            if (pair == 0xffffffffu) continue;
            F ra_lo[8], ra_hi[8];
            for (uint column = 0; column < p.columns; column++) {
                ra_lo[column] = bytecode_ra_gather(rows, ra_tables, shifts, source_tail, p, ulong(pair) * 2, column);
                ra_hi[column] = bytecode_ra_gather(rows, ra_tables, shifts, source_tail, p, ulong(pair) * 2 + 1, column);
            }
            F c0 = bytecode_reduce(&coefficients[(3 * slot) * 5], F::zero());
            F c1 = bytecode_reduce(&coefficients[(3 * slot + 1) * 5], F::zero());
            F c2 = bytecode_reduce(&coefficients[(3 * slot + 2) * 5], F::zero());
            F output[10]; bytecode_product(ra_lo, ra_hi, c0, c1, c2, p, output);
            for (uint i = 0; i < p.degree; i++) sums[i].add(output[i]);
        }
    }
    for (uint i = 0; i < p.degree; i++) {
        Acc sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, simdgroup, 2, 32);
        if (local == 0) partials[ulong(group) * p.degree + i] = sum.reduce();
    }
}
