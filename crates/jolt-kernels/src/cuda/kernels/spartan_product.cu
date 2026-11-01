#define SP_NARROW 2
#define SP_WIDE 1
#define SP_LEFT_SLOT 0
#define SP_LOOKUP_SLOT 1
#define SP_CLAIM_LANES 4
#define SP_KIND_NARROW 0u
#define SP_KIND_WIDE 1u
#define SP_KIND_FLAG 2u

#define SP_GATHER_BITS 5

extern "C" __global__ void sp_gather_kernel(
    const unsigned int *__restrict__ canonical, const unsigned int *__restrict__ bit_sources,
    unsigned int sign_base, const u64 *__restrict__ left_input,
    const u64 *__restrict__ right_input, const u64 *__restrict__ lookup_output,
    u64 *__restrict__ narrow, u64 *__restrict__ wide, unsigned int *__restrict__ flags,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    u64 *row = narrow + (size_t)t * SP_NARROW;
    row[SP_LEFT_SLOT] = left_input[t];
    row[SP_LOOKUP_SLOT] = lookup_output[t];

    unsigned int mask = 0u;
    u128 right = ((u128)right_input[2 * (size_t)t + 1] << 64) | (u128)right_input[2 * (size_t)t];
    bool negative = ((__int128)right) < 0;
    u128 magnitude = negative ? (~right + (u128)1) : right;
    u64 *limbs = wide + (size_t)t * (SP_WIDE * 2);
    limbs[0] = (u64)magnitude;
    limbs[1] = (u64)(magnitude >> 64);
    if (negative) mask |= 1u << sign_base;

    unsigned int source = canonical[t];
    for (unsigned int bit = 0u; bit < SP_GATHER_BITS; ++bit) {
        mask |= ((source >> bit_sources[bit]) & 1u) << bit;
    }
    flags[t] = mask;
}


__device__ __forceinline__ void sp_accumulate_word(u64 *folded, const u64 *coefficient,
                                                   unsigned long long word) {
    if (word == 0ULL) return;
    if (word == 1ULL) {
        unr_add_field(folded, coefficient);
        return;
    }
    unr_mul_words(coefficient, &word, 1, folded);
}

__device__ __forceinline__ void sp_negate(const u64 *value, u64 *out) {
    const u64 zero[LIMBS] = {0, 0, 0, 0};
    fr_sub(zero, value, out);
}

// Coefficients are checked on the host to have magnitude at most three.
// Thus |left| < 2^67, |right| < 2^130 and |product| < 2^197.
__device__ __forceinline__ void sp_extended_product(
    const u64 *row, const u64 *wide, unsigned int mask, const int *coefficients,
    unsigned int jump_bit, unsigned int branch_bit, unsigned int noop_bit,
    unsigned int sign_bit, u64 *product, bool &negative) {
    __int128 left = (__int128)coefficients[0] * row[SP_LEFT_SLOT]
        + (__int128)coefficients[1] * row[SP_LOOKUP_SLOT]
        + coefficients[2] * (int)((mask >> jump_bit) & 1u);
    u128 left_abs = left < 0 ? (u128)-left : (u128)left;
    int c = coefficients[0];
    u64 magnitude = c < 0 ? -c : c;
    u128 lo = (u128)wide[0] * magnitude;
    u128 hi = (u128)wide[1] * magnitude + (lo >> 64);
    u64 right[3] = {(u64)lo, (u64)hi, (u64)(hi >> 64)};
    bool right_negative = ((mask >> sign_bit) & 1u) != (c < 0);
    if (right_negative) {
        u64 carry = 1;
        for (int i = 0; i < 3; i++) {
            u128 value = (u128)~right[i] + carry;
            right[i] = (u64)value;
            carry = (u64)(value >> 64);
        }
    }
    int flags = coefficients[1] * (int)((mask >> branch_bit) & 1u)
        + coefficients[2] * (int)(1u - ((mask >> noop_bit) & 1u));
    u64 carry = 0;
    for (int i = 0; i < 3; i++) {
        u64 word = i == 0 ? (u64)(long long)flags : (flags < 0 ? ~0ULL : 0ULL);
        u128 value = (u128)right[i] + word + carry;
        right[i] = (u64)value;
        carry = (u64)(value >> 64);
    }
    right_negative = (right[2] >> 63) != 0;
    if (right_negative) {
        carry = 1;
        for (int i = 0; i < 3; i++) {
            u128 value = (u128)~right[i] + carry;
            right[i] = (u64)value;
            carry = (u64)(value >> 64);
        }
    }
    negative = (left < 0) != right_negative;
    u64 left_words[2] = {(u64)left_abs, (u64)(left_abs >> 64)};
    u64 words[5] = {};
    for (int i = 0; i < 2; i++) {
        carry = 0;
        for (int j = 0; j < 3; j++) {
            u128 value = (u128)left_words[i] * right[j] + words[i + j] + carry;
            words[i + j] = (u64)value;
            carry = (u64)(value >> 64);
        }
        words[i + 3] = carry;
    }
    for (int i = 0; i < 4; i++) product[i] = words[i];
}

extern "C" __global__ void sp_uniskip_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const int *__restrict__ coefficients,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out,
    unsigned int in_bits, unsigned int jump_bit, unsigned int branch_bit,
    unsigned int noop_bit, unsigned int sign_bit, unsigned int cycles,
    unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    const int *node = coefficients + 3 * blockIdx.y;
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;
    // Four-word integer products require one more slot than the shared
    // three-word accumulator. The strip bound keeps each split slot below u64.
    u64 folded[16] = {};
    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long t = base + step;
        if (t >= cycles) break;
        u64 product[4];
        bool negative;
        sp_extended_product(narrow + t * SP_NARROW, wide + t * (SP_WIDE * 2),
                            flags[t], node, jump_bit, branch_bit, noop_bit, sign_bit,
                            product, negative);
        if ((product[0] | product[1] | product[2] | product[3]) == 0) continue;
        u64 weight[LIMBS];
        so_split_eq(e_in, e_out, in_bits, t, weight);
        if (negative) sp_negate(weight, weight);
        unr_mul_words(weight, product, 4, folded);
    }
    // A strip sums at most four 451-bit weighted products. Its upper half
    // is below 2^199, so the shared Montgomery high-half reduction is valid.
    u64 words[8], carry = 0;
    for (int i = 0; i < 8; i++) {
        u128 value = (u128)folded[2 * i] + ((u128)folded[2 * i + 1] << 32) + carry;
        words[i] = (u64)value;
        carry = (u64)(value >> 64);
    }
    u64 low[LIMBS], high[LIMBS], acc[1][LIMBS];
    fr_reduce_256(words, low);
    fr_mul(words + LIMBS, FR_R2, high);
    fr_add(low, high, acc[0]);
    so_block_reduce(scratch, 1, acc, partials + (unsigned long long)blockIdx.y * gridDim.x * LIMBS);
}

__device__ __forceinline__ void sp_factors_at(
    const u64 *narrow, const u64 *wide, const unsigned int *flags, const u64 *weights,
    unsigned int jump_bit, unsigned int branch_bit, unsigned int noop_bit,
    unsigned int sign_bit, unsigned long long t, u64 *left, u64 *right) {
    unsigned int mask = flags[t];
    const u64 *row = narrow + t * SP_NARROW;
    const u64 *row_wide = wide + t * (SP_WIDE * 2);

    u64 w0[LIMBS], w1[LIMBS], w2[LIMBS];
    load4(weights, w0);
    load4(weights + LIMBS, w1);
    load4(weights + 2 * LIMBS, w2);

    u64 folded[2 * UNR_SLOTS];

    unr_zero(folded);
    sp_accumulate_word(folded, w0, row[SP_LEFT_SLOT]);
    sp_accumulate_word(folded, w1, row[SP_LOOKUP_SLOT]);
    if ((mask >> jump_bit) & 1u) unr_add_field(folded, w2);
    unr_finalize(folded, left);

    unr_zero(folded);
    unsigned long long pair[2];
    pair[0] = row_wide[0];
    pair[1] = row_wide[1];
    if (pair[0] != 0ULL || pair[1] != 0ULL) {
        if ((mask >> sign_bit) & 1u) {
            u64 flipped[LIMBS];
            sp_negate(w0, flipped);
            unr_mul_words(flipped, pair, 2, folded);
        } else {
            unr_mul_words(w0, pair, 2, folded);
        }
    }
    if ((mask >> branch_bit) & 1u) unr_add_field(folded, w1);
    if (((mask >> noop_bit) & 1u) == 0u) unr_add_field(folded, w2);
    unr_finalize(folded, right);
}

extern "C" __global__ void sp_prepare_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const u64 *__restrict__ weights,
    unsigned int jump_bit, unsigned int branch_bit, unsigned int noop_bit,
    unsigned int sign_bit, unsigned int cycles, const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out, unsigned int e_in_len, unsigned int in_bits,
    u64 *__restrict__ left_out, u64 *__restrict__ right_out,
    u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    u64 acc[3][LIMBS] = {};
    if (pair < cycles / 2) {
        u64 left0[LIMBS], right0[LIMBS], left1[LIMBS], right1[LIMBS];
        sp_factors_at(narrow, wide, flags, weights, jump_bit, branch_bit, noop_bit,
                      sign_bit, 2 * pair, left0, right0);
        sp_factors_at(narrow, wide, flags, weights, jump_bit, branch_bit, noop_bit,
                      sign_bit, 2 * pair + 1, left1, right1);
        store4(left_out + (2 * pair) * LIMBS, left0);
        store4(right_out + (2 * pair) * LIMBS, right0);
        store4(left_out + (2 * pair + 1) * LIMBS, left1);
        store4(right_out + (2 * pair + 1) * LIMBS, right1);

        u64 weight[LIMBS];
        if (e_in_len <= 1) {
            load4(e_out + pair * LIMBS, weight);
        } else {
            so_split_eq(e_in, e_out, in_bits, pair, weight);
        }
        gruen_pair_accumulate(left0, left1, right0, right1, weight, acc);
    }
    so_block_reduce(scratch, 3, acc, partials);
}

extern "C" __global__ void sp_claims_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const unsigned int *__restrict__ layout,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out, unsigned int in_bits,
    unsigned int sign_base, unsigned int first, unsigned int lanes, unsigned int cycles,
    unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 folded[SP_CLAIM_LANES][2 * UNR_SLOTS];
    for (unsigned int lane = 0; lane < lanes; lane++) unr_zero(folded[lane]);

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long t = base + step;
        if (t >= cycles) break;

        unsigned int mask = flags[t];
        const u64 *row = narrow + t * SP_NARROW;
        const u64 *row_wide = wide + t * (SP_WIDE * 2);
        u64 weight[LIMBS];
        so_split_eq(e_in, e_out, in_bits, t, weight);

        // NVCC 12.4 miscompiles this accumulator loop with a dynamic upper bound.
        for (unsigned int lane = 0; lane < SP_CLAIM_LANES; lane++) {
            if (lane >= lanes) continue;
            unsigned int term = layout[first + lane];
            unsigned int slot = term & 0xFFu;
            unsigned int kind = (term >> 8) & 3u;
            if (kind == SP_KIND_FLAG) {
                if ((mask >> slot) & 1u) unr_add_field(folded[lane], weight);
                continue;
            }
            if (kind == SP_KIND_NARROW) {
                sp_accumulate_word(folded[lane], weight, row[slot]);
                continue;
            }
            unsigned long long pair[2];
            pair[0] = row_wide[2 * slot];
            pair[1] = row_wide[2 * slot + 1];
            if (pair[0] == 0ULL && pair[1] == 0ULL) continue;
            if ((mask >> (sign_base + slot)) & 1u) {
                u64 flipped[LIMBS];
                sp_negate(weight, flipped);
                unr_mul_words(flipped, pair, 2, folded[lane]);
            } else {
                unr_mul_words(weight, pair, 2, folded[lane]);
            }
        }
    }

    u64 acc[SP_CLAIM_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++) unr_finalize(folded[lane], acc[lane]);
    for (unsigned int lane = lanes; lane < SP_CLAIM_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    so_block_reduce(scratch, lanes, acc, partials);
}
