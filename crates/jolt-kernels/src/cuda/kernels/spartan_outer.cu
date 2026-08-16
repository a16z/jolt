#define SO_NARROW 13
#define SO_CLAIM_LANES 4
#define SO_WIDE 4
#define SO_SIGN_BASE 24
#define SO_KIND_NARROW 0u
#define SO_KIND_WIDE 1u
#define SO_KIND_FLAG 2u

extern "C" __global__ void so_shift_kernel(const unsigned int *__restrict__ raw,
                                          u64 *__restrict__ narrow,
                                          unsigned int *__restrict__ flags,
                                          unsigned int pc_slot, unsigned int unexpanded_pc_slot,
                                          unsigned int next_pc_slot,
                                          unsigned int next_unexpanded_pc_slot,
                                          unsigned int virtual_bit, unsigned int first_bit,
                                          unsigned int next_virtual_bit, unsigned int next_first_bit,
                                          unsigned int cycles) {
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int mask = raw[t];
    if (t + 1 < cycles) {
        unsigned int next = raw[t + 1];
        mask |= ((next >> virtual_bit) & 1u) << next_virtual_bit;
        mask |= ((next >> first_bit) & 1u) << next_first_bit;
        narrow[t * SO_NARROW + next_unexpanded_pc_slot] =
            narrow[(t + 1) * SO_NARROW + unexpanded_pc_slot];
        narrow[t * SO_NARROW + next_pc_slot] = narrow[(t + 1) * SO_NARROW + pc_slot];
    } else {
        narrow[t * SO_NARROW + next_unexpanded_pc_slot] = 0;
        narrow[t * SO_NARROW + next_pc_slot] = 0;
    }
    flags[t] = mask;
}

__device__ __forceinline__ void so_linear_form(const u64 *__restrict__ narrow,
                                               const u64 *__restrict__ wide, unsigned int mask,
                                               unsigned long long t,
                                               const unsigned int *__restrict__ terms,
                                               const u64 *__restrict__ coeffs, unsigned int count,
                                               const u64 *__restrict__ constant, u64 *out) {
    u64 pos[2 * UNR_SLOTS];
    unr_zero(pos);
    const u64 zero[LIMBS] = {0, 0, 0, 0};

    const u64 *row = narrow + t * SO_NARROW;
    const u64 *row_wide = wide + t * (SO_WIDE * 2);

    for (unsigned int k = 0; k < count; k++) {
        unsigned int term = terms[k];
        unsigned int slot = term & 0xFFu;
        unsigned int kind = (term >> 8) & 3u;
        const u64 *coefficient = coeffs + (unsigned long long)k * LIMBS;

        if (kind == SO_KIND_FLAG) {
            if ((mask >> slot) & 1u) unr_add_field(pos, coefficient);
            continue;
        }
        if (kind == SO_KIND_NARROW) {
            unsigned long long word = row[slot];
            if (word != 0ULL) unr_mul_words(coefficient, &word, 1, pos);
            continue;
        }
        unsigned long long words[2];
        words[0] = row_wide[2 * slot];
        words[1] = row_wide[2 * slot + 1];
        if (words[0] == 0ULL && words[1] == 0ULL) continue;
        if ((mask >> (SO_SIGN_BASE + slot)) & 1u) {
            u64 flipped[LIMBS];
            fr_sub(zero, coefficient, flipped);
            unr_mul_words(flipped, words, 2, pos);
        } else {
            unr_mul_words(coefficient, words, 2, pos);
        }
    }

    u64 total[LIMBS];
    unr_finalize(pos, total);
    u64 shift[LIMBS];
    load4(constant, shift);
    fr_add(total, shift, out);
}

__device__ __forceinline__ void so_form(const u64 *__restrict__ narrow,
                                        const u64 *__restrict__ wide, unsigned int mask,
                                        unsigned long long t,
                                        const unsigned int *__restrict__ offsets,
                                        const unsigned int *__restrict__ counts,
                                        const u64 *__restrict__ constants,
                                        const unsigned int *__restrict__ terms,
                                        const u64 *__restrict__ coeffs, unsigned int form,
                                        u64 *out) {
    so_linear_form(narrow, wide, mask, t, terms + offsets[form],
                   coeffs + (unsigned long long)offsets[form] * LIMBS, counts[form],
                   constants + (unsigned long long)form * LIMBS, out);
}

__device__ __forceinline__ void so_split_eq(const u64 *__restrict__ e_in,
                                            const u64 *__restrict__ e_out, unsigned int in_bits,
                                            unsigned long long index, u64 *out) {
    u64 inner[LIMBS], outer[LIMBS];
    load4(e_in + (index & ((1ull << in_bits) - 1ull)) * LIMBS, inner);
    load4(e_out + (index >> in_bits) * LIMBS, outer);
    fr_mul(inner, outer, out);
}

__device__ __forceinline__ void so_block_reduce(u64 *scratch, unsigned int lanes,
                                                const u64 acc[][LIMBS], u64 *partials) {
    unsigned int tid = threadIdx.x;
    for (unsigned int lane = 0; lane < lanes; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], s[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, s);
                store4(scratch + tid * LIMBS, s);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(partials + ((unsigned long long)lane * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void so_uniskip_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ counts, const u64 *__restrict__ constants,
    const unsigned int *__restrict__ terms, const u64 *__restrict__ coeffs,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out, unsigned int in_bits,
    unsigned int cycles, unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int node = blockIdx.y;
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 acc[1][LIMBS];
    for (int l = 0; l < LIMBS; l++) acc[0][l] = 0;

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long t = base + step;
        if (t >= cycles) break;
        unsigned int mask = flags[t];

        for (unsigned int stream = 0; stream < 2; stream++) {
            unsigned int form = (node * 2 + stream) * 2;
            u64 az[LIMBS], bz[LIMBS];
            so_form(narrow, wide, mask, t, offsets, counts, constants, terms, coeffs, form, az);
            so_form(narrow, wide, mask, t, offsets, counts, constants, terms, coeffs, form + 1, bz);

            u64 product[LIMBS], weight[LIMBS], weighted[LIMBS], sum[LIMBS];
            fr_mul(az, bz, product);
            so_split_eq(e_in, e_out, in_bits, (t << 1) | stream, weight);
            fr_mul(product, weight, weighted);
            fr_add(acc[0], weighted, sum);
            for (int l = 0; l < LIMBS; l++) acc[0][l] = sum[l];
        }
    }

    u64 *lane_partials = partials + (unsigned long long)node * gridDim.x * LIMBS;
    so_block_reduce(scratch, 1, acc, lane_partials);
}

extern "C" __global__ void so_factors_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ counts, const u64 *__restrict__ constants,
    const unsigned int *__restrict__ terms, const u64 *__restrict__ coeffs, unsigned int cycles,
    u64 *__restrict__ az, u64 *__restrict__ bz) {
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int mask = flags[t];
    for (unsigned int stream = 0; stream < 2; stream++) {
        u64 value[LIMBS];
        so_form(narrow, wide, mask, t, offsets, counts, constants, terms, coeffs, stream * 2, value);
        store4(az + ((t << 1) | stream) * LIMBS, value);
        so_form(narrow, wide, mask, t, offsets, counts, constants, terms, coeffs, stream * 2 + 1,
                value);
        store4(bz + ((t << 1) | stream) * LIMBS, value);
    }
}

extern "C" __global__ void gruen_pair_message_kernel(const u64 *__restrict__ az,
                                             const u64 *__restrict__ bz,
                                             const u64 *__restrict__ e_in,
                                             const u64 *__restrict__ e_out, unsigned int e_in_len,
                                             unsigned int in_bits, unsigned int half,
                                             unsigned int strip, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 acc[2][LIMBS];
    for (int lane = 0; lane < 2; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long pair = base + step;
        if (pair >= half) break;

        u64 az0[LIMBS], az1[LIMBS], bz0[LIMBS], bz1[LIMBS];
        load4(az + (2 * pair) * LIMBS, az0);
        load4(az + (2 * pair + 1) * LIMBS, az1);
        load4(bz + (2 * pair) * LIMBS, bz0);
        load4(bz + (2 * pair + 1) * LIMBS, bz1);

        u64 weight[LIMBS];
        if (e_in_len <= 1) {
            load4(e_out + pair * LIMBS, weight);
        } else {
            so_split_eq(e_in, e_out, in_bits, pair, weight);
        }

        u64 product[LIMBS], weighted[LIMBS], sum[LIMBS];
        fr_mul(az0, bz0, product);
        fr_mul(product, weight, weighted);
        fr_add(acc[0], weighted, sum);
        for (int l = 0; l < LIMBS; l++) acc[0][l] = sum[l];

        u64 az_inf[LIMBS], bz_inf[LIMBS];
        fr_sub(az1, az0, az_inf);
        fr_sub(bz1, bz0, bz_inf);
        fr_mul(az_inf, bz_inf, product);
        fr_mul(product, weight, weighted);
        fr_add(acc[1], weighted, sum);
        for (int l = 0; l < LIMBS; l++) acc[1][l] = sum[l];
    }

    so_block_reduce(scratch, 2, acc, partials);
}

extern "C" __global__ void so_claims_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const unsigned int *__restrict__ layout,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out, unsigned int in_bits,
    unsigned int first, unsigned int lanes, unsigned int cycles, unsigned int strip,
    u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * (unsigned long long)strip;

    u64 folded[SO_CLAIM_LANES][2 * UNR_SLOTS];
    for (unsigned int lane = 0; lane < lanes; lane++) unr_zero(folded[lane]);
    const u64 zero[LIMBS] = {0, 0, 0, 0};

    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long t = base + step;
        if (t >= cycles) break;

        unsigned int mask = flags[t];
        const u64 *row = narrow + t * SO_NARROW;
        const u64 *row_wide = wide + t * (SO_WIDE * 2);
        u64 weight[LIMBS];
        so_split_eq(e_in, e_out, in_bits, t, weight);

        for (unsigned int lane = 0; lane < lanes; lane++) {
            unsigned int term = layout[first + lane];
            unsigned int slot = term & 0xFFu;
            unsigned int kind = (term >> 8) & 3u;
            if (kind == SO_KIND_FLAG) {
                if ((mask >> slot) & 1u) unr_add_field(folded[lane], weight);
                continue;
            }
            if (kind == SO_KIND_NARROW) {
                unsigned long long word = row[slot];
                if (word != 0ULL) unr_mul_words(weight, &word, 1, folded[lane]);
                continue;
            }
            unsigned long long words[2];
            words[0] = row_wide[2 * slot];
            words[1] = row_wide[2 * slot + 1];
            if (words[0] == 0ULL && words[1] == 0ULL) continue;
            if ((mask >> (SO_SIGN_BASE + slot)) & 1u) {
                u64 flipped[LIMBS];
                fr_sub(zero, weight, flipped);
                unr_mul_words(flipped, words, 2, folded[lane]);
            } else {
                unr_mul_words(weight, words, 2, folded[lane]);
            }
        }
    }

    u64 acc[SO_CLAIM_LANES][LIMBS];
    for (unsigned int lane = 0; lane < lanes; lane++) unr_finalize(folded[lane], acc[lane]);
    for (unsigned int lane = lanes; lane < SO_CLAIM_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    so_block_reduce(scratch, lanes, acc, partials);
}
