struct InputNative { ulong rs1, pc, rs2, rd, imm_lo, imm_hi; };
struct InputParams {
    uint prefix, output_len, first_pair, pairs, in_len, old_bits, new_bits, need_one;
};
template <typename F> struct InputConstants {
    F tails[10];
    uint4 tail_word;
    F challenge, gamma;
};
inline uint input_pattern(uint4 word, uint column, uint bits) {
    uint offset = column * bits;
    return (word[offset / 32] >> (offset % 32)) & ((1u << bits) - 1u);
}
inline uint4 input_merge(uint4 low, uint4 high, uint bits) {
    uint4 word = uint4(0);
    for (uint c = 0; c < 5; c++) {
        uint pattern = input_pattern(low, c, bits) | (input_pattern(high, c, bits) << bits);
        uint offset = c * bits * 2;
        word[offset / 32] |= pattern << (offset % 32);
    }
    return word;
}
template <typename F> F input_operand(InputNative row, uint column) {
    switch (column) {
        case 0: return F::from_u64(row.rs1);
        case 1: return F::from_u64(row.pc);
        case 2: return F::from_u64(row.rs2);
        case 3: return jolt::from_i128<F>(ulong2(row.imm_lo, row.imm_hi));
        default: return F::from_u64(row.rd);
    }
}
template <typename F>
void input_accumulate(thread jolt::Accumulator<F>* sums, thread F (&flags)[2][5],
    thread F (&values)[2][5], F equality, bool need_one) {
    F left[3] = {F::zero(), F::zero(), F::zero()};
    F right[3] = {F::zero(), F::zero(), F::zero()};
    for (uint c = 0; c < 5; c++) {
        F q0 = flags[0][c] * values[0][c];
        F q2 = (flags[1][c] - flags[0][c]) * (values[1][c] - values[0][c]);
        F q1 = need_one ? flags[1][c] * values[1][c] : F::zero();
        if (c < 2) { left[0] = left[0] + q0; left[1] = left[1] + q2; left[2] = left[2] + q1; }
        else { right[0] = right[0] + q0; right[1] = right[1] + q2; right[2] = right[2] + q1; }
    }
    for (uint term = 0; term < 3; term++) {
        sums[term].fmadd(equality, left[term]);
        sums[term + 3].fmadd(equality, right[term]);
    }
}
template <typename F>
void input_reduce(thread jolt::Accumulator<F>* sums, threadgroup jolt::Accumulator<F> (&scratch)[6][8],
    device F* partials, F outer, F gamma, uint group, ushort lane, ushort sg) {
    for (uint term = 0; term < 3; term++) {
        auto left = jolt::threadgroup_merge(sums[term], scratch[term], lane, sg, 8, 32);
        auto right = jolt::threadgroup_merge(sums[term + 3], scratch[term + 3], lane, sg, 8, 32);
        if (lane == 0 && sg == 0)
            partials[ulong(group) * 3 + term] = outer * (right.reduce() + gamma * left.reduce());
    }
}
// Source tiles contain two native rows per materialized output row. Each
// invocation creates two output rows and immediately evaluates their pair.
template <typename F>
kernel void instruction_input_native(
    device const InputNative* source [[buffer(0)]], device const ushort* codes [[buffer(1)]],
    device const F* lut [[buffer(2)]], device const F* e_in [[buffer(3)]], device const F* e_out [[buffer(4)]],
    device F* output [[buffer(5)]], device uint4* words [[buffer(6)]], device F* partials [[buffer(7)]],
    constant InputConstants<F>& c [[buffer(8)]], constant InputParams& p [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one equality inner block fits");
    threadgroup Acc scratch[6][8];
    Acc sums[6];
    for (uint i = 0; i < 6; i++) sums[i] = Acc::zero();
    for (uint inner = local; inner < p.in_len; inner += 256) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F flags[2][5], values[2][5];
        for (uint j = 0; j < 2; j++) {
            ulong row = (ulong(p.first_pair) + pair) * 2 + j;
            uint4 word = c.tail_word;
            if (row < p.output_len) {
                word = uint4(uint(codes[pair * 2 + j]), 0, 0, 0);
                words[row] = word;
            }
            for (uint column = 0; column < 5; column++) {
                F value = c.tails[column * 2 + 1];
                if (row < p.output_len) {
                    F lo = input_operand<F>(source[pair * 4 + j * 2], column);
                    F hi = input_operand<F>(source[pair * 4 + j * 2 + 1], column);
                    value = lo + c.challenge * (hi - lo);
                    output[ulong(column) * p.output_len + row] = value;
                }
                values[j][column] = value;
                flags[j][column] = lut[input_pattern(word, column, 2)];
            }
        }
        input_accumulate(sums, flags, values, e_in[inner], p.need_one != 0);
    }
    input_reduce(sums, scratch, partials, e_out[p.first_pair / p.in_len + group], c.gamma, group, lane, sg);
}

template <typename F>
F input_value(device const F* values, uint prefix, ulong row, uint column, F tail) {
    if (row < prefix) return values[ulong(column) * prefix + row];
    return tail;
}
// The output never aliases the input. Packed patterns remain 2/4/8/16 bits;
// at saturation, lookup and bind flags directly into dense field columns.
template <typename F>
kernel void instruction_input_bind_round(
    device const F* previous [[buffer(0)]], device const uint4* old_words [[buffer(1)]],
    device const F* lut [[buffer(2)]], device const F* e_in [[buffer(3)]], device const F* e_out [[buffer(4)]],
    device F* output [[buffer(5)]], device uint4* words [[buffer(6)]], device F* partials [[buffer(7)]],
    constant InputConstants<F>& c [[buffer(8)]], constant InputParams& p [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 0xfffffffful, "one equality inner block fits");
    threadgroup Acc scratch[6][8];
    Acc sums[6];
    for (uint i = 0; i < 6; i++) sums[i] = Acc::zero();
    for (uint inner = local; inner < p.in_len; inner += 256) {
        ulong pair = ulong(group) * p.in_len + inner;
        if (pair >= p.pairs) break;
        F flags[2][5], values[2][5];
        for (uint j = 0; j < 2; j++) {
            ulong row = pair * 2 + j, low = row * 2, high = low + 1;
            uint4 lo_word = c.tail_word, hi_word = c.tail_word, merged = uint4(0);
            if (p.old_bits) {
                if (low < p.prefix) lo_word = old_words[low];
                if (high < p.prefix) hi_word = old_words[high];
                if (p.new_bits) merged = input_merge(lo_word, hi_word, p.old_bits);
            }
            if (p.new_bits && row < p.output_len) words[row] = merged;
            for (uint column = 0; column < 5; column++) {
                uint value_column = p.old_bits ? column : 2 * column + 1;
                F vlo = input_value(previous, p.prefix, low, value_column, c.tails[column * 2 + 1]);
                F vhi = input_value(previous, p.prefix, high, value_column, c.tails[column * 2 + 1]);
                F value = vlo + c.challenge * (vhi - vlo);
                F flag;
                if (p.new_bits) flag = lut[input_pattern(merged, column, p.new_bits)];
                else {
                    F flo = p.old_bits ? lut[input_pattern(lo_word, column, p.old_bits)]
                        : input_value(previous, p.prefix, low, 2 * column, c.tails[2 * column]);
                    F fhi = p.old_bits ? lut[input_pattern(hi_word, column, p.old_bits)]
                        : input_value(previous, p.prefix, high, 2 * column, c.tails[2 * column]);
                    flag = flo + c.challenge * (fhi - flo);
                }
                if (row < p.output_len) {
                    uint destination = p.new_bits ? column : 2 * column + 1;
                    output[ulong(destination) * p.output_len + row] = value;
                    if (!p.new_bits) output[ulong(2 * column) * p.output_len + row] = flag;
                }
                values[j][column] = value;
                flags[j][column] = flag;
            }
        }
        input_accumulate(sums, flags, values, e_in[inner], p.need_one != 0);
    }
    input_reduce(sums, scratch, partials, e_out[group], c.gamma, group, lane, sg);
}
