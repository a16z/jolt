struct ReadRafParams {
    uint prefix, pairs, width, columns, bits, in_len, mode, output_rows, need_zero;
};

template <typename F>
F read_raf_gather(device const ulong4* rows, device const F* ra, device const F* claims,
                 constant ulong4& tail, constant ReadRafParams& p, ulong row, uint factor) {
    F value = F::zero();
    for (uint branch = 0; branch < p.width; branch++) {
        ulong index = row * p.width + branch;
        ulong4 packed = index < p.prefix ? rows[index] : tail;
        if (factor == 0) {
            value = value + claims[branch * 256u + uint(packed.z >> 56)];
        } else {
            uint column = factor - 1, shift = (p.columns - 1 - column) * p.bits;
            ulong limb = shift < 64 ? packed.x : packed.y;
            uint radix = 1u << p.bits;
            uint hot = uint(limb >> (shift & 63)) & (radix - 1);
            value = value + ra[(column * p.width + branch) * radix + hot];
        }
    }
    return value;
}

template <typename F, uint N>
F read_raf_dense(device const F* values, device const F* tails, uint prefix, ulong row, uint factor) {
    return row < prefix ? values[row * N + factor] : tails[factor];
}

// Constant factors are removed before sampling. Low-degree products need only
// degree samples and their leading coefficient; finite differences extend them.
template <typename F, uint N>
void read_raf_product(thread F* lo, thread F* hi, device const F* tails,
                      device const F* cached, bool need_zero, thread F* out) {
    F values[N], slopes[N], constant = F::one();
    uint degree = 0, mask = 0;
    for (uint i = 0; i < N; i++) {
        F delta = hi[i] - lo[i];
        if (all(delta.limb == uint4(0))) {
            if (N == 9 && i != 0 && all(hi[i].limb == tails[i].limb)) mask |= 1u << (i - 1);
            else constant = constant * hi[i];
        } else {
            values[degree] = hi[i]; slopes[degree] = delta; degree++;
        }
    }
    if (N == 9 && mask != 0) constant = constant * cached[mask];
    for (uint i = 0; i <= N; i++) out[i] = F::zero();
    if (need_zero) {
        F value = F::one();
        for (uint i = 0; i < N; i++) value = value * lo[i];
        out[N] = value;
    }
    if (all(constant.limb == uint4(0))) return;
    if (degree == 0) {
        for (uint i = 0; i < N - 1; i++) out[i] = constant;
        return;
    }
    values[0] = values[0] * constant; slopes[0] = slopes[0] * constant;
    F lead = slopes[0];
    for (uint i = 1; i < degree; i++) lead = lead * slopes[i];
    uint samples = degree <= 6 ? degree : N - 1;
    for (uint point = 0; point < samples; point++) {
        F value = values[0];
        for (uint i = 1; i < degree; i++) value = value * values[i];
        out[point] = value;
        for (uint i = 0; i < degree; i++) values[i] = values[i] + slopes[i];
    }
    if (degree <= 6) {
        F work[6], back[6];
        for (uint i = 0; i < degree; i++) work[i] = out[i];
        back[0] = work[degree - 1];
        for (uint order = 1; order < degree; order++) {
            for (uint i = degree - 1; i >= order; i--) work[i] = work[i] - work[i - 1];
            back[order] = work[degree - 1];
        }
        ulong factorial = 1;
        for (uint i = 2; i <= degree; i++) factorial *= i;
        F top = mul_u64(lead, factorial);
        for (uint point = degree; point < N - 1; point++) {
            back[degree - 1] = back[degree - 1] + top;
            for (uint i = degree - 1; i > 0; i--) back[i - 1] = back[i - 1] + back[i];
            out[point] = back[0];
        }
    }
    if (degree == N) out[N - 1] = lead;
}

template <typename F, uint N>
void read_raf_round_impl(
    device const ulong4* rows, device const F* ra, device const F* claims,
    device const F* previous, device F* next, device const F* tails, device const F* cached,
    device const F* e_in, device const F* e_out, device F* partials,
    constant ReadRafParams& p, constant ulong4& tail, constant F& challenge,
    uint group, uint local, uint threads, ushort lane, ushort simdgroup,
    ushort simdgroups, ushort simdwidth, threadgroup jolt::Accumulator<F> (*scratch)[32]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xFFFFFFFFul, "one equality block fits");
    Acc sums[N + 1];
    for (uint i = 0; i <= N; i++) sums[i] = Acc::zero();
    for (uint inner = local; inner < p.in_len; inner += threads) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F lo[N], hi[N], samples[N + 1];
        for (uint side = 0; side < 2; side++) {
            ulong row = 2 * pair + side;
            for (uint factor = 0; factor < N; factor++) {
                F value;
                if (p.mode == 0) {
                    value = read_raf_gather(rows, ra, claims, tail, p, row, factor);
                } else {
                    F a, b;
                    if (p.mode == 1) {
                        a = read_raf_gather(rows, ra, claims, tail, p, 2 * row, factor);
                        b = read_raf_gather(rows, ra, claims, tail, p, 2 * row + 1, factor);
                    } else {
                        a = read_raf_dense<F, N>(previous, tails, p.prefix, 2 * row, factor);
                        b = read_raf_dense<F, N>(previous, tails, p.prefix, 2 * row + 1, factor);
                    }
                    value = a + challenge * (b - a);
                    if (row < p.output_rows) next[row * N + factor] = value;
                }
                if (side == 0) lo[factor] = value; else hi[factor] = value;
            }
        }
        lo[0] = e_in[inner] * lo[0]; hi[0] = e_in[inner] * hi[0];
        read_raf_product<F, N>(lo, hi, tails, cached, p.need_zero != 0, samples);
        for (uint i = 0; i < N + uint(p.need_zero); i++) sums[i].add(samples[i]);
    }
    for (uint i = 0; i <= N; i++) {
        Acc sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, simdgroup, simdgroups, simdwidth);
        if (simdgroup == 0 && lane == 0) partials[ulong(group) * (N + 1) + i] = e_out[group] * sum.reduce();
    }
}

#define READ_RAF_KERNEL(NAME, N) \
template <typename F> kernel void NAME( \
    device const ulong4* rows [[buffer(0)]], device const F* ra [[buffer(1)]], \
    device const F* claims [[buffer(2)]], device const F* previous [[buffer(3)]], \
    device F* next [[buffer(4)]], device const F* tails [[buffer(5)]], \
    device const F* cached [[buffer(6)]], device const F* e_in [[buffer(7)]], \
    device const F* e_out [[buffer(8)]], device F* partials [[buffer(9)]], \
    constant ReadRafParams& p [[buffer(10)]], constant ulong4& tail [[buffer(11)]], \
    constant F& challenge [[buffer(12)]], uint group [[threadgroup_position_in_grid]], \
    uint local [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], ushort simdgroup [[simdgroup_index_in_threadgroup]], \
    ushort simdgroups [[simdgroups_per_threadgroup]], ushort simdwidth [[threads_per_simdgroup]]) { \
    threadgroup jolt::Accumulator<F> scratch[N + 1][32]; \
    read_raf_round_impl<F, N>(rows, ra, claims, previous, next, tails, cached, e_in, e_out, \
        partials, p, tail, challenge, group, local, threads, lane, simdgroup, simdgroups, simdwidth, scratch); \
}
READ_RAF_KERNEL(read_raf_round9, 9)
READ_RAF_KERNEL(read_raf_round17, 17)
#undef READ_RAF_KERNEL
