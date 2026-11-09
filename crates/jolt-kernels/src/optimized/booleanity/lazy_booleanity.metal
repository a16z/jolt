struct BooleanityParams {
    uint prefix, pairs, width, radix, columns, in_len, need_one;
};

uint booleanity_index(ulong4 row, uint2 selector, uint radix) {
    ulong value = row[selector.x];
    if (selector.x == 2) value &= 0x00FFFFFFFFFFFFFFul;
    if (selector.x == 3) {
        if (value == 0) return radix;
        value--;
    }
    return uint(value >> selector.y) & (radix - 1);
}

template <typename F>
kernel void lazy_booleanity(
    device const ulong4* rows [[buffer(0)]],
    device const F* tables [[buffer(1)]],
    device const F* rho [[buffer(2)]],
    device const uint2* selectors [[buffer(3)]],
    device const F* e_in [[buffer(4)]],
    device const F* e_out [[buffer(5)]],
    device F* partials [[buffer(6)]],
    constant BooleanityParams& p [[buffer(7)]],
    constant ulong4& tail [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "group sums fit the accumulator");
    threadgroup Acc scratch[3][32];
    Acc sums[3];
    for (uint k = 0; k < 3; k++) sums[k] = Acc::zero();
    // A SIMD group evaluates one pair, with columns distributed across lanes.
    for (uint inner = simdgroup; inner < p.in_len; inner += simdgroups) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        Acc values[3];
        for (uint k = 0; k < 3; k++) values[k] = Acc::zero();
        for (uint column = lane; column < p.columns; column += simdwidth) {
            F lo = F::zero(), hi = F::zero();
            for (uint branch = 0; branch < p.width; branch++) {
                ulong lo_row = 2 * pair * p.width + branch, hi_row = lo_row + p.width;
                uint l = booleanity_index(lo_row < p.prefix ? rows[lo_row] : tail, selectors[column], p.radix);
                uint h = booleanity_index(hi_row < p.prefix ? rows[hi_row] : tail, selectors[column], p.radix);
                uint base = (column * p.width + branch) * p.radix;
                if (l < p.radix) lo = lo + tables[base + l];
                if (h < p.radix) hi = hi + tables[base + h];
            }
            F delta = hi - lo;
            values[0].fmadd(lo, lo - rho[column]);
            values[1].fmadd(delta, delta);
            if (p.need_one) values[2].fmadd(hi, hi - rho[column]);
        }
        for (uint k = 0; k < 3; k++) {
            Acc value = jolt::simd_merge(values[k], simdwidth);
            if (lane == 0) sums[k].fmadd(e_in[inner], value.reduce());
        }
    }
    for (uint k = 0; k < 3; k++) {
        Acc sum = jolt::threadgroup_merge(sums[k], scratch[k], lane, simdgroup, simdgroups, simdwidth);
        if (simdgroup == 0 && lane == 0) partials[3 * group + k] = e_out[group] * sum.reduce();
    }
}
