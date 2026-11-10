// The CPU owns table values. This kernel only decodes canonical categories and
// performs the same split-equality reduction as lazy_booleanity.
struct CachedBooleanityParams {
    BooleanityParams base;
    uint mode, states, square_stride;
};

template <typename F>
kernel void lazy_booleanity_cached(
    device const ulong4* rows [[buffer(0)]],
    device const F* rho [[buffer(1)]],
    device const uint2* selectors [[buffer(2)]],
    device const F* e_in [[buffer(3)]],
    device const F* e_out [[buffer(4)]],
    device F* partials [[buffer(5)]],
    constant CachedBooleanityParams& params [[buffer(6)]],
    constant ulong4& tail [[buffer(7)]],
    device const F* values [[buffer(8)]],
    device const F* squares [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "group sums fit the accumulator");
    constant BooleanityParams& p = params.base;
    threadgroup Acc scratch[3][32];
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    uint radix = p.radix + 1;
    for (uint inner = simdgroup; inner < p.in_len; inner += simdgroups) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        Acc terms[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
        for (uint column = lane; column < p.columns; column += simdwidth) {
            if (params.mode <= 2) {
                uint lo = 0, hi = 0, power = 1;
                for (uint branch = 0; branch < p.width; branch++) {
                    ulong lrow = 2 * pair * p.width + branch, hrow = lrow + p.width;
                    lo += power * booleanity_index(lrow < p.prefix ? rows[lrow] : tail, selectors[column], p.radix);
                    hi += power * booleanity_index(hrow < p.prefix ? rows[hrow] : tail, selectors[column], p.radix);
                    power *= radix;
                }
                // CategoricalProducts uses first + radix * second.
                uint base = column * params.states;
                uint square = params.mode == 1 ? lo * params.states + hi
                    : metal::max(lo, hi) * (metal::max(lo, hi) + 1) / 2 + metal::min(lo, hi);
                terms[0].add(values[base + lo]);
                terms[1].add(squares[column * params.square_stride + square]);
                if (p.need_one) terms[2].add(values[base + hi]);
            } else {
                F lo = F::zero(), hi = F::zero();
                for (uint branch = 0; branch < p.width; branch += 2) {
                    ulong lrow = 2 * pair * p.width + branch, hrow = lrow + p.width;
                    uint l0 = booleanity_index(lrow < p.prefix ? rows[lrow] : tail, selectors[column], p.radix);
                    uint l1 = booleanity_index(lrow + 1 < p.prefix ? rows[lrow + 1] : tail, selectors[column], p.radix);
                    uint h0 = booleanity_index(hrow < p.prefix ? rows[hrow] : tail, selectors[column], p.radix);
                    uint h1 = booleanity_index(hrow + 1 < p.prefix ? rows[hrow + 1] : tail, selectors[column], p.radix);
                    // PairedBranchTables uses first * radix + second.
                    uint base = column * params.states + (branch / 2) * radix * radix;
                    lo = lo + values[base + l0 * radix + l1];
                    hi = hi + values[base + h0 * radix + h1];
                }
                F delta = hi - lo;
                terms[0].fmadd(lo, lo - rho[column]);
                terms[1].fmadd(delta, delta);
                if (p.need_one) terms[2].fmadd(hi, hi - rho[column]);
            }
        }
        for (uint k = 0; k < 3; k++) {
            Acc value = jolt::simd_merge(terms[k], simdwidth);
            if (lane == 0) sums[k].fmadd(e_in[inner], value.reduce());
        }
    }
    for (uint k = 0; k < 3; k++) {
        Acc sum = jolt::threadgroup_merge(sums[k], scratch[k], lane, simdgroup, simdgroups, simdwidth);
        if (simdgroup == 0 && lane == 0) partials[3 * group + k] = e_out[group] * sum.reduce();
    }
}
