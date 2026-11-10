// The CPU owns table values. This kernel only decodes canonical categories and
// performs the same split-equality reduction as lazy_booleanity.
struct CachedBooleanityParams {
    BooleanityParams base;
    uint mode, states, square_stride;
};

template <typename F, bool Weighted>
inline void cached_booleanity_pair(
    device const ulong4* rows, device const F* rho, device const uint2* selectors,
    constant CachedBooleanityParams& params, constant ulong4& tail,
    device const F* values, device const F* squares, device const F* weighted,
    ulong pair, uint phase, ushort lane, ushort simdwidth,
    thread jolt::Accumulator<F>* terms) {
    constant BooleanityParams& p = params.base;
    uint radix = p.radix + 1;
    uint values_len = p.columns * params.states;
    uint weighted_base = phase * (values_len + p.columns * params.square_stride);
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
            terms[0].add(Weighted ? weighted[weighted_base + base + lo] : values[base + lo]);
            uint square_index = column * params.square_stride + square;
            terms[1].add(Weighted ? weighted[weighted_base + values_len + square_index] : squares[square_index]);
            if (p.need_one) terms[2].add(Weighted ? weighted[weighted_base + base + hi] : values[base + hi]);
        } else {
            F lo = F::zero(), hi = F::zero();
            F weighted_lo = F::zero(), weighted_hi = F::zero();
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
                if (Weighted) {
                    weighted_lo = weighted_lo + weighted[weighted_base + base + l0 * radix + l1];
                    weighted_hi = weighted_hi + weighted[weighted_base + base + h0 * radix + h1];
                }
            }
            F delta = hi - lo;
            if (!Weighted) { weighted_lo = lo; weighted_hi = hi; }
            terms[0].fmadd(weighted_lo, lo - rho[column]);
            terms[1].fmadd(weighted_hi - weighted_lo, delta);
            if (p.need_one) terms[2].fmadd(weighted_hi, hi - rho[column]);
        }
    }
}

template <typename F, uint BlockPairs>
inline void cached_booleanity_reduce(
    device const ulong4* rows, device const F* rho, device const uint2* selectors,
    device const F* e_in, device const F* e_out, device F* partials,
    constant CachedBooleanityParams& params, constant ulong4& tail,
    device const F* values, device const F* squares, device const F* weighted,
    uint group, ushort lane, ushort simdgroup, ushort simdgroups, ushort simdwidth,
    threadgroup jolt::Accumulator<F> (*scratch)[32]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "group sums fit the accumulator");
    constant BooleanityParams& p = params.base;
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    for (uint block = simdgroup * BlockPairs; block < p.in_len; block += simdgroups * BlockPairs) {
        ulong first_pair = ulong(group) * p.in_len + block;
        if (first_pair >= p.pairs) break;
        Acc terms[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
        for (uint phase = 0; phase < BlockPairs; phase++) {
            ulong pair = first_pair + phase;
            if (pair >= p.pairs) break;
            cached_booleanity_pair<F, (BlockPairs != 1)>(rows, rho, selectors, params,
                tail, values, squares, weighted, pair, phase, lane, simdwidth, terms);
        }
        for (uint k = 0; k < 3; k++) {
            Acc value = jolt::simd_merge(terms[k], simdwidth);
            if (lane == 0) sums[k].fmadd(e_in[block / BlockPairs], value.reduce());
        }
    }
    for (uint k = 0; k < 3; k++) {
        Acc sum = jolt::threadgroup_merge(sums[k], scratch[k], lane, simdgroup, simdgroups, simdwidth);
        if (simdgroup == 0 && lane == 0) partials[3 * group + k] = e_out[group] * sum.reduce();
    }
}
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
    threadgroup jolt::Accumulator<F> scratch[3][32];
    cached_booleanity_reduce<F, 1>(rows, rho, selectors, e_in, e_out, partials,
        params, tail, values, squares, values,
        group, lane, simdgroup, simdgroups, simdwidth, scratch);
}

template <typename F>
kernel void lazy_booleanity_factored(
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
    device const F* weighted [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    threadgroup jolt::Accumulator<F> scratch[3][32];
    cached_booleanity_reduce<F, 16>(rows, rho, selectors, e_in, e_out, partials,
        params, tail, values, squares, weighted,
        group, lane, simdgroup, simdgroups, simdwidth, scratch);
}

struct BooleanityWeightParams { uint values, squares, phases; };

template <typename F>
kernel void booleanity_weight_tables(
    device const F* values [[buffer(0)]], device const F* squares [[buffer(1)]],
    device const F* active [[buffer(2)]], device F* weighted [[buffer(3)]],
    constant BooleanityWeightParams& p [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint entries = p.values + p.squares;
    if (index >= entries * p.phases) return;
    uint phase = index / entries, entry = index % entries;
    F value = F::zero();
    if (entry < p.values) value = values[entry];
    else value = squares[entry - p.values];
    weighted[index] = active[phase] * value;
}
