// Columns have a constant suffix after the stream bind. Only their explicit
// prefixes are stored; the host adds the suffix's equality mass.
template <typename F>
F outer_value(device const F* values, uint n, ulong index, F tail) {
    return index < n ? values[index] : tail;
}

template <typename F>
kernel void outer_product(
    device const F* az [[buffer(0)]],
    device const F* bz [[buffer(1)]],
    device F* next_az [[buffer(2)]],
    device F* next_bz [[buffer(3)]],
    device const F* e_in [[buffer(4)]],
    device const F* e_out [[buffer(5)]],
    device F* partials [[buffer(6)]],
    constant uint& old_len [[buffer(7)]],
    constant uint& in_len [[buffer(8)]],
    constant uint& bind [[buffer(9)]],
    constant uint& need_one [[buffer(10)]],
    constant F& challenge [[buffer(11)]],
    constant F& az_tail [[buffer(12)]],
    constant F& bz_tail [[buffer(13)]],
    uint local [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort width [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul,
                  "a threadgroup can accumulate every prefix pair");
    threadgroup Acc scratch[3][32];
    Acc q0 = Acc::zero(), qi = Acc::zero(), q1 = Acc::zero();
    ulong n = bind ? (ulong(old_len) + 1) / 2 : old_len;
    ulong pairs = (n + 1) / 2;
    // Each group owns one outer-equality block; apply its weight once below.
    for (uint inner = local; inner < in_len; inner += threads) {
        ulong j = ulong(group) * in_len + inner;
        if (j >= pairs) break;
        ulong index = (bind ? 4 : 2) * j;
        F a0 = outer_value(az, old_len, index, az_tail);
        F a1 = outer_value(az, old_len, index + 1, az_tail);
        F b0 = outer_value(bz, old_len, index, bz_tail);
        F b1 = outer_value(bz, old_len, index + 1, bz_tail);
        if (bind) {
            a0 = a0 + challenge * (a1 - a0);
            b0 = b0 + challenge * (b1 - b0);
            a1 = outer_value(az, old_len, index + 2, az_tail);
            b1 = outer_value(bz, old_len, index + 2, bz_tail);
            a1 = a1 + challenge * (outer_value(az, old_len, index + 3, az_tail) - a1);
            b1 = b1 + challenge * (outer_value(bz, old_len, index + 3, bz_tail) - b1);
            next_az[2 * j] = a0;
            next_bz[2 * j] = b0;
            if (2 * j + 1 < n) {
                next_az[2 * j + 1] = a1;
                next_bz[2 * j + 1] = b1;
            }
        }
        F weight = e_in[inner];
        q0.fmadd(weight * a0, b0);
        qi.fmadd(weight * (a1 - a0), b1 - b0);
        if (need_one) q1.fmadd(weight * a1, b1);
    }
    q0 = jolt::threadgroup_merge(q0, scratch[0], lane, simdgroup, simdgroups, width);
    qi = jolt::threadgroup_merge(qi, scratch[1], lane, simdgroup, simdgroups, width);
    q1 = jolt::threadgroup_merge(q1, scratch[2], lane, simdgroup, simdgroups, width);
    if (simdgroup == 0 && lane == 0) {
        F weight = e_out[group];
        partials[3 * group] = weight * q0.reduce();
        partials[3 * group + 1] = weight * qi.reduce();
        partials[3 * group + 2] = weight * q1.reduce();
    }
}
