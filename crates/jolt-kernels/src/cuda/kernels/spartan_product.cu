#define SP_NARROW 2
#define SP_WIDE 1
#define SP_LEFT_SLOT 0
#define SP_LOOKUP_SLOT 1
#define SP_MATRIX_LANES 3

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

__device__ __forceinline__ void sp_accumulate_pair(u64 *folded, const u64 *coefficient,
                                                   unsigned long long left,
                                                   unsigned long long lo, unsigned long long hi) {
    if (left == 0ULL || (lo == 0ULL && hi == 0ULL)) return;
    if (left == 1ULL) {
        unsigned long long pair[2] = {lo, hi};
        unr_mul_words(coefficient, pair, 2, folded);
        return;
    }
    u128 low = (u128)left * (u128)lo;
    u128 high = (u128)left * (u128)hi + (u128)(unsigned long long)(low >> 64);
    unsigned long long words[3];
    words[0] = (unsigned long long)low;
    words[1] = (unsigned long long)high;
    words[2] = (unsigned long long)(high >> 64);
    unr_mul_words(coefficient, words, 3, folded);
}

__device__ __forceinline__ void sp_negate(const u64 *value, u64 *out) {
    const u64 zero[LIMBS] = {0, 0, 0, 0};
    fr_sub(zero, value, out);
}

extern "C" __global__ void sp_matrix_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out, unsigned int in_bits, unsigned int jump_bit,
    unsigned int branch_bit, unsigned int noop_bit, unsigned int sign_bit, unsigned int cycles,
    unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int left_index = blockIdx.y;
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 folded[SP_MATRIX_LANES][2 * UNR_SLOTS];
    for (unsigned int lane = 0; lane < SP_MATRIX_LANES; lane++) unr_zero(folded[lane]);

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long t = base + step;
        if (t >= cycles) break;

        unsigned int mask = flags[t];
        const u64 *row = narrow + t * SP_NARROW;
        unsigned long long left;
        if (left_index == 0) {
            left = row[SP_LEFT_SLOT];
        } else if (left_index == 1) {
            left = row[SP_LOOKUP_SLOT];
        } else {
            left = (unsigned long long)((mask >> jump_bit) & 1u);
        }
        if (left == 0ULL) continue;

        u64 weight[LIMBS];
        so_split_eq(e_in, e_out, in_bits, t, weight);

        const u64 *row_wide = wide + t * (SP_WIDE * 2);
        unsigned long long lo = row_wide[0];
        unsigned long long hi = row_wide[1];
        if (lo != 0ULL || hi != 0ULL) {
            if ((mask >> sign_bit) & 1u) {
                u64 flipped[LIMBS];
                sp_negate(weight, flipped);
                sp_accumulate_pair(folded[0], flipped, left, lo, hi);
            } else {
                sp_accumulate_pair(folded[0], weight, left, lo, hi);
            }
        }
        if ((mask >> branch_bit) & 1u) sp_accumulate_word(folded[1], weight, left);
        if (((mask >> noop_bit) & 1u) == 0u) sp_accumulate_word(folded[2], weight, left);
    }

    u64 acc[SP_MATRIX_LANES][LIMBS];
    for (unsigned int lane = 0; lane < SP_MATRIX_LANES; lane++)
        unr_finalize(folded[lane], acc[lane]);

    u64 *lane_partials =
        partials + (unsigned long long)left_index * SP_MATRIX_LANES * gridDim.x * LIMBS;
    so_block_reduce(scratch, SP_MATRIX_LANES, acc, lane_partials);
}
