struct LazyRaParams {
    uint prefix, pairs, width, columns, chunk_bits, active_virtuals, in_len, need_zero, dense;
};

uint ra_chunk(ulong2 index, uint column, constant LazyRaParams& p) {
    uint shift = (p.columns - 1 - column) * p.chunk_bits;
    ulong limb = shift < 64 ? index.x : index.y;
    return uint(limb >> (shift & 63)) & ((1u << p.chunk_bits) - 1);
}

// A group owns one E_out block, so its weight is applied after reduction.
template <typename F>
kernel void lazy_ra_product(
    device const ulong4* indices [[buffer(0)]],
    device const F* tables [[buffer(1)]],
    device const F* e_in [[buffer(2)]],
    device const F* e_out [[buffer(3)]],
    device F* partials [[buffer(4)]],
    constant LazyRaParams& p [[buffer(5)]],
    constant ulong4& tail [[buffer(6)]],
    device const F* dense [[buffer(7)]],
    device const F* tails [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "group sums fit the accumulator");
    threadgroup Acc scratch[5][32];
    Acc sums[5];
    for (uint k = 0; k < 5; k++) sums[k] = Acc::zero();
    uint radix = 1u << p.chunk_bits;
    for (uint inner = local; inner < p.in_len; inner += threads) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F row[5];
        for (uint k = 0; k < 5; k++) row[k] = F::zero();
        for (uint batch = 0; batch < p.active_virtuals; batch++) {
            F low[4], high[4];
            for (uint f = 0; f < 4; f++) { low[f] = F::zero(); high[f] = F::zero(); }
            if (p.dense) {
                for (uint f = 0; f < 4; f++) {
                    uint column = 4 * batch + f;
                    low[f] = 2 * pair < p.prefix ? dense[2 * pair * p.columns + column] : tails[column];
                    high[f] = 2 * pair + 1 < p.prefix ? dense[(2 * pair + 1) * p.columns + column] : tails[column];
                }
            } else {
            for (uint branch = 0; branch < p.width; branch++) {
                ulong lo_row = 2 * pair * p.width + branch;
                ulong hi_row = lo_row + p.width;
                ulong2 lo = lo_row < p.prefix ? indices[lo_row].xy : tail.xy;
                ulong2 hi = hi_row < p.prefix ? indices[hi_row].xy : tail.xy;
                #pragma unroll
                for (uint f = 0; f < 4; f++) {
                    uint column = 4 * batch + f;
                    uint base = (column * p.width + branch) * radix;
                    low[f] = low[f] + tables[base + ra_chunk(lo, column, p)];
                    high[f] = high[f] + tables[base + ra_chunk(hi, column, p)];
                }
            }
            }
            F delta[4];
            for (uint f = 0; f < 4; f++) delta[f] = high[f] - low[f];
            F l1 = high[0] * high[1], r1 = high[2] * high[3];
            F l2 = (high[0] + delta[0]) * (high[1] + delta[1]);
            F r2 = (high[2] + delta[2]) * (high[3] + delta[3]);
            F li = delta[0] * delta[1], ri = delta[2] * delta[3];
            F l3 = l2 + li, r3 = r2 + ri;
            l3 = l3 + l3 - l1; r3 = r3 + r3 - r1;
            row[0] = row[0] + l1 * r1;
            row[1] = row[1] + l2 * r2;
            row[2] = row[2] + l3 * r3;
            row[3] = row[3] + li * ri;
            if (p.need_zero) row[4] = row[4] + (low[0] * low[1]) * (low[2] * low[3]);
        }
        for (uint k = 0; k < (p.need_zero ? 5u : 4u); k++) sums[k].fmadd(e_in[inner], row[k]);
    }
    for (uint k = 0; k < 5; k++) {
        Acc sum = jolt::threadgroup_merge(sums[k], scratch[k], lane, simdgroup, simdgroups, simdwidth);
        if (simdgroup == 0 && lane == 0) partials[5 * group + k] = e_out[group] * sum.reduce();
    }
}
