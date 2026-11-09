struct ProductNativeRow {
    ulong left, lookup, right_lo, right_hi;
    uint flags, reserved;
};
template <typename F>
struct ProductTails { F values[2]; };
struct ProductParams {
    uint first, count, in_len, outputs, mode, prefix, pairs, need_one;
};

template <typename F>
F product_right(ProductNativeRow row) {
    F lo = F::from_u64(row.right_lo);
    if (row.right_hi == 0) return lo;
    F radix = F::from_u64(1ul << 63) + F::from_u64(1ul << 63);
    if (row.right_hi == ~0ul) return lo - radix;
    // Signed high limb times2^64 plus unsigned low limb represents every i128,
    // including MIN; no negation of a signed minimum or signed overflow.
    return lo + F::from_i64(long(row.right_hi)) * radix;
}

template <typename F>
void product_factors(ProductNativeRow row, device const F* weights, thread F& left, thread F& right) {
    left = mul_u64(weights[0], row.left) + mul_u64(weights[1], row.lookup);
    if (row.flags & 1u) left = left + weights[2];
    right = weights[0] * product_right<F>(row);
    if (row.flags & 4u) right = right + weights[1];
    if (!(row.flags & 8u)) right = right + weights[2];
}

// mode0: canonical extension rows; mode1: direct prefix-field production plus
// first message; mode2: eight original opening columns. Each threadgroup owns
// one split-equality inner block. Source tiles start on these block boundaries.
template <typename F>
kernel void spartan_product_native(
    device const ProductNativeRow* rows [[buffer(0)]], device const F* weights [[buffer(1)]],
    device const F* e_in [[buffer(2)]], device const F* e_out [[buffer(3)]],
    device F* values [[buffer(4)]], device F* partials [[buffer(5)]],
    constant ProductParams& p [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0x100000ul, "a source tile fits the accumulator");
    threadgroup Acc scratch[8][8];
    Acc sums[8];
    for (uint i = 0; i < 8; i++) sums[i] = Acc::zero();
    uint first = p.mode == 1 ? p.first / 2 : p.first;
    uint count = p.mode == 1 ? p.count / 2 : p.count;
    for (uint inner = local; inner < p.in_len; inner += 256) {
        uint index = group * p.in_len + inner;
        if (index >= count) break;
        F e = e_in[inner];
        if (p.mode == 1) {
            F a0, b0, a1, b1;
            product_factors(rows[2 * index], weights, a0, b0);
            product_factors(rows[2 * index + 1], weights, a1, b1);
            ulong out = ulong(p.first + 2 * index) * 2;
            values[out] = a0; values[out + 1] = b0;
            values[out + 2] = a1; values[out + 3] = b1;
            sums[0].fmadd(e * a0, b0);
            sums[1].fmadd(e * (a1 - a0), b1 - b0);
            if (p.need_one) sums[2].fmadd(e * a1, b1);
        } else {
            ProductNativeRow row = rows[index];
            if (p.mode == 0) {
                for (uint i = 0; i < p.outputs; i++) {
                    F a, b;
                    product_factors(row, weights + 3 * i, a, b);
                    sums[i].fmadd(e * a, b);
                }
            } else {
                sums[0].fmadd(e, F::from_u64(row.left));
                sums[1].fmadd(e, product_right<F>(row));
                sums[4].fmadd(e, F::from_u64(row.lookup));
                constexpr uint slots[5] = {2, 3, 5, 6, 7};
                for (uint i = 0; i < 5; i++) if (row.flags & (1u << i)) sums[slots[i]].add(e);
            }
        }
    }
    for (uint i = 0; i < p.outputs; i++) {
        Acc sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, sg, 8, 32);
        if (sg == 0 && lane == 0) partials[ulong(group) * p.outputs + i] = e_out[first / p.in_len + group] * sum.reduce();
    }
}

template <typename F>
F product_dense(device const F* values, constant F* tails, uint prefix, ulong row, uint column) {
    if (row < prefix) return values[row * 2 + column];
    return tails[column];
}

template <typename F>
kernel void spartan_product_bind(
    device const F* previous [[buffer(0)]], device F* next [[buffer(1)]],
    constant ProductTails<F>& tails [[buffer(2)]], constant F& challenge [[buffer(3)]],
    constant uint& prefix [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= (prefix + 1) / 2) return;
    for (uint column = 0; column < 2; column++) {
        F a = product_dense(previous, tails.values, prefix, ulong(row) * 2, column);
        F b = product_dense(previous, tails.values, prefix, ulong(row) * 2 + 1, column);
        next[ulong(row) * 2 + column] = a + challenge * (b - a);
    }
}

template <typename F>
kernel void spartan_product_round(
    device const F* values [[buffer(0)]], device const F* e_in [[buffer(1)]],
    device const F* e_out [[buffer(2)]], device F* partials [[buffer(3)]],
    constant ProductTails<F>& tails [[buffer(4)]], constant ProductParams& p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one equality block fits");
    threadgroup Acc scratch[3][8];
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    for (uint inner = local; inner < p.in_len; inner += 256) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F a0 = product_dense(values, tails.values, p.prefix, 2 * pair, 0);
        F a1 = product_dense(values, tails.values, p.prefix, 2 * pair + 1, 0);
        F b0 = product_dense(values, tails.values, p.prefix, 2 * pair, 1);
        F b1 = product_dense(values, tails.values, p.prefix, 2 * pair + 1, 1);
        F e = e_in[inner];
        sums[0].fmadd(e * a0, b0);
        sums[1].fmadd(e * (a1 - a0), b1 - b0);
        if (p.need_one) sums[2].fmadd(e * a1, b1);
    }
    for (uint i = 0; i < 3; i++) {
        Acc sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, sg, 8, 32);
        if (sg == 0 && lane == 0) partials[ulong(group) * 3 + i] = e_out[group] * sum.reduce();
    }
}
