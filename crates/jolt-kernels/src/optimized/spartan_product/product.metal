template <typename F>
struct ProductTails { F values[2]; };
struct ProductParams {
    uint first, count, in_len, outputs, mode, prefix, pairs, need_one;
};

template <typename F>
void product_factors(NativeCore row, device const F* weights, constant InputTerms& terms, thread F& left, thread F& right) {
    F native_left, native_right;
    native_inputs(row, terms, native_left, native_right);
    left = weights[0] * native_left + mul_u64(weights[1], row.lookup);
    if (row.flags & (1u << 5)) left = left + weights[2];
    right = weights[0] * native_right;
    if (row.flags & (1u << 7)) right = right + weights[1];
    if (!(row.flags & (1u << 8))) right = right + weights[2];
}

// mode0: canonical extension rows; mode1: direct prefix-field production plus
// first message. Each threadgroup owns one split-equality inner block.
template <typename F>
kernel void spartan_product_native(
    device const NativeCore* rows [[buffer(0)]], device const F* weights [[buffer(1)]],
    device const F* e_in [[buffer(2)]], device const F* e_out [[buffer(3)]],
    device F* values [[buffer(4)]], device F* partials [[buffer(5)]],
    constant ProductParams& p [[buffer(6)]], constant InputTerms& terms [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one split equality block fits");
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
            product_factors(rows[2 * index], weights, terms, a0, b0);
            product_factors(rows[2 * index + 1], weights, terms, a1, b1);
            ulong out = ulong(p.first + 2 * index) * 2;
            values[out] = a0; values[out + 1] = b0;
            values[out + 2] = a1; values[out + 3] = b1;
            sums[0].fmadd(e * a0, b0);
            sums[1].fmadd(e * (a1 - a0), b1 - b0);
            if (p.need_one) sums[2].fmadd(e * a1, b1);
        } else {
            NativeCore row = rows[index];
            for (uint i = 0; i < p.outputs; i++) {
                F a, b;
                product_factors(row, weights + 3 * i, terms, a, b);
                sums[i].fmadd(e * a, b);
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

template <typename F>
kernel void spartan_operands_openings(
    device const NativeCore* rows [[buffer(0)]], device const NativeLookup* lookup [[buffer(1)]],
    device const F* e_in [[buffer(2)]], device const F* e_out [[buffer(3)]],
    device F* partials [[buffer(4)]], constant ProductParams& p [[buffer(5)]],
    constant InputTerms& terms [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one equality block fits");
    threadgroup Acc scratch[10][8];
    Acc sums[10]; for (uint i = 0; i < 10; i++) sums[i] = Acc::zero();
    for (uint inner = local; inner < p.in_len; inner += 256) {
        ulong index = ulong(group) * p.in_len + inner;
        if (index >= p.count) break;
        NativeCore row = rows[index];
        F left, right; native_inputs(row, terms, left, right);
        F e = e_in[inner];
        sums[0].fmadd(e, left); sums[1].fmadd(e, right);
        sums[4].fmadd(e, F::from_u64(row.lookup));
        constexpr uint slots[5] = {2, 3, 5, 6, 7};
        for (uint i = 0; i < 5; i++) if (row.flags & (1u << (i + 5))) sums[slots[i]].add(e);
        sums[8].fmadd(e, F::from_u64(lookup[index].left));
        sums[9].fmadd(e, native_lookup_right<F>(lookup[index]));
    }
    for (uint i = 0; i < 10; i++) {
        auto sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, sg, 8, 32);
        if (lane == 0 && sg == 0) partials[ulong(group) * 10 + i] = e_out[group] * sum.reduce();
    }
}

template <typename F>
F operand_combined(NativeCore row, NativeLookup lookup, device const F* weights, constant InputTerms& terms) {
    F left, right; native_inputs(row, terms, left, right);
    return weights[0] * F::from_u64(row.lookup) + weights[1] * F::from_u64(lookup.left)
        + weights[2] * native_lookup_right<F>(lookup) + weights[3] * left + weights[4] * right;
}
template <typename F>
kernel void spartan_operand_combined(
    device const NativeCore* rows [[buffer(0)]], device const NativeLookup* lookups [[buffer(1)]],
    device const F* weights [[buffer(2)]], device const F* previous [[buffer(3)]],
    device F* output [[buffer(4)]], device const F* e_in [[buffer(5)]], device const F* e_out [[buffer(6)]],
    device F* partials [[buffer(7)]], constant ProductParams& p [[buffer(8)]],
    constant InputTerms& terms [[buffer(9)]], constant F& tail [[buffer(10)]], constant F& challenge [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one equality block fits");
    threadgroup Acc scratch[3][8];
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    for (uint inner = local; inner < p.in_len; inner += 256) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F values[2] = {tail, tail};
        for (uint side = 0; side < 2; side++) {
            ulong row = 2 * pair + side;
            if (row < p.count) {
                F value = tail;
                if (p.mode == 0) {
                    if (row < p.prefix) value = operand_combined<F>(rows[row], lookups[row], weights, terms);
                } else {
                    F low = tail;
                    F high = tail;
                    if (2 * row < p.prefix) low = previous[2 * row];
                    if (2 * row + 1 < p.prefix) high = previous[2 * row + 1];
                    value = low + challenge * (high - low);
                }
                values[side] = value;
                output[row] = value;
            }
        }
        sums[0].fmadd(e_in[inner], values[0]);
        sums[1].fmadd(e_in[inner], values[1]);
        sums[2].fmadd(e_in[inner], values[1] + values[1] - values[0]);
    }
    for (uint i = 0; i < 3; i++) {
        auto sum = jolt::threadgroup_merge(sums[i], scratch[i], lane, sg, 8, 32);
        if (lane == 0 && sg == 0) partials[ulong(group) * 3 + i] = e_out[group] * sum.reduce();
    }
}
