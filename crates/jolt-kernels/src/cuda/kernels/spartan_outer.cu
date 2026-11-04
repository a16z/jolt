#define SO_NARROW 13
#define SO_CLAIM_LANES 4
#define SO_WIDE 4
#define SO_SIGN_BASE 24
#define SO_KIND_NARROW 0u
#define SO_KIND_WIDE 1u
#define SO_KIND_FLAG 2u
#define SO_EXTRA_WORDS 11
#define SO_EXTRA_RS1 0
#define SO_EXTRA_RS2 1
#define SO_EXTRA_RD_POST 2
#define SO_EXTRA_RAM_READ 3
#define SO_EXTRA_RAM_WRITE 4
#define SO_EXTRA_IMM_LO 5
#define SO_EXTRA_IMM_HI 6
#define SO_GATHER_BITS 16

extern "C" __global__ void so_gather_kernel(
    const u64 *__restrict__ extras, const u64 *__restrict__ address,
    const u64 *__restrict__ ram_address, const unsigned int *__restrict__ pc_words,
    const unsigned int *__restrict__ canonical, const unsigned int *__restrict__ bit_sources,
    unsigned int product_sign_source, const u64 *__restrict__ left_input,
    const u64 *__restrict__ right_input, const u64 *__restrict__ left_operand,
    const u64 *__restrict__ right_operand, const u64 *__restrict__ lookup_output,
    const u64 *__restrict__ product, unsigned int sign_base, u64 *__restrict__ narrow,
    u64 *__restrict__ wide,
    unsigned int *__restrict__ raw, unsigned long long *__restrict__ unmapped,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int word = pc_words[t];
    if (word == 0xFFFFFFFFu) {
        atomicMin(unmapped, (unsigned long long)t);
        return;
    }

    const u64 *words = extras + (size_t)t * SO_EXTRA_WORDS;
    u64 *row = narrow + (size_t)t * SO_NARROW;
    u64 ram = ram_address[t];
    row[0] = left_input[t];
    row[1] = (u64)word;
    row[2] = address[t];
    row[3] = (ram == 0xFFFFFFFFFFFFFFFFull) ? 0ull : ram;
    row[4] = words[SO_EXTRA_RS1];
    row[5] = words[SO_EXTRA_RS2];
    row[6] = words[SO_EXTRA_RD_POST];
    row[7] = words[SO_EXTRA_RAM_READ];
    row[8] = words[SO_EXTRA_RAM_WRITE];
    row[9] = left_operand[t];
    row[10] = lookup_output[t];
    row[11] = 0ull;
    row[12] = 0ull;

    unsigned int mask = 0u;
    u64 *limbs = wide + (size_t)t * (SO_WIDE * 2);

    u128 right = ((u128)right_input[2 * (size_t)t + 1] << 64) | (u128)right_input[2 * (size_t)t];
    bool right_negative = ((__int128)right) < 0;
    u128 magnitude = right_negative ? (~right + (u128)1) : right;
    limbs[0] = (u64)magnitude;
    limbs[1] = (u64)(magnitude >> 64);
    if (right_negative) mask |= 1u << (sign_base + 0u);

    limbs[2] = product[2 * (size_t)t];
    limbs[3] = product[2 * (size_t)t + 1];
    if (((canonical[t] >> product_sign_source) & 1u) != 0u) mask |= 1u << (sign_base + 1u);

    u128 imm = ((u128)words[SO_EXTRA_IMM_HI] << 64) | (u128)words[SO_EXTRA_IMM_LO];
    bool imm_negative = ((__int128)imm) < 0;
    u128 imm_magnitude = imm_negative ? (~imm + (u128)1) : imm;
    limbs[4] = (u64)imm_magnitude;
    limbs[5] = (u64)(imm_magnitude >> 64);
    if (imm_negative) mask |= 1u << (sign_base + 2u);

    limbs[6] = right_operand[2 * (size_t)t];
    limbs[7] = right_operand[2 * (size_t)t + 1];

    unsigned int source = canonical[t];
    for (unsigned int bit = 0u; bit < SO_GATHER_BITS; ++bit) {
        mask |= ((source >> bit_sources[bit]) & 1u) << bit;
    }
    raw[t] = mask;
}

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

#define SO_GROUPS 32
#define SO_DOMAIN 10
#define SO_NODES (SO_DOMAIN - 1)
#define SO_LEFT_NODES (SO_DOMAIN / 2)

__device__ __forceinline__ unsigned int so_guard_key(
    unsigned int mask, unsigned int stream, const unsigned int *offsets,
    const unsigned int *terms) {
    unsigned int key = 0x80000000u;
    for (unsigned int row = 0; row < SO_DOMAIN; row++) {
        unsigned int form = 2 * (stream * SO_DOMAIN + row);
        int guard = 0;
        for (unsigned int j = offsets[form]; j < offsets[form + 1]; j++) {
            unsigned int term = terms[j];
            int value = ((term >> 8) & 3u) == 3u ? 1 : (mask >> (term & 255u)) & 1u;
            value *= 1 << (term >> 24);
            guard += (term & (1u << 16)) ? -value : value;
        }
        key |= (unsigned int)(guard + 3) << (3 * row);
    }
    return key;
}

__device__ __forceinline__ void so_integer_form(
    const u64 *narrow, const u64 *wide, unsigned int mask, unsigned long long t,
    const unsigned int *offsets, const unsigned int *terms, unsigned int form,
    __int128 &low, __int128 &high) {
    low = high = 0;
    for (unsigned int j = offsets[form]; j < offsets[form + 1]; j++) {
        unsigned int term = terms[j], slot = term & 255u, kind = (term >> 8) & 3u;
        unsigned int shift = term >> 24;
        bool negative = (term & (1u << 16)) != 0;
        u64 lo = 0, hi = 0;
        if (kind == SO_KIND_NARROW) {
            lo = narrow[t * SO_NARROW + slot];
        } else if (kind == SO_KIND_WIDE) {
            lo = wide[t * SO_WIDE * 2 + 2 * slot];
            hi = wide[t * SO_WIDE * 2 + 2 * slot + 1];
            negative ^= ((mask >> (SO_SIGN_BASE + slot)) & 1u) != 0;
        } else if (kind == SO_KIND_FLAG) {
            lo = (mask >> slot) & 1u;
        } else {
            lo = 1;
        }
        __int128 a, b;
        if (shift == 64) {
            a = 0;
            b = lo;
        } else {
            a = (__int128)lo * ((__int128)1 << shift);
            b = (__int128)hi * ((__int128)1 << shift);
        }
        low += negative ? -a : a;
        high += negative ? -b : b;
    }
}

__device__ __forceinline__ void so_weight_integer(
    const u64 *weight, __int128 low, __int128 high, u64 *folded) {
    // GuardForms bounds each row's low/high part below 2^72, extension gain
    // below 2^20, and guards to [-3, 3]. Extended products stay below 2^114
    // per part, so their combined magnitude fits three words. At most 1024
    // rows accumulate in seven unreduced limbs, including carry headroom.
    __int128 top = high + (low >> 64);
    u64 bottom = (u64)low;
    const u64 zero[LIMBS] = {0, 0, 0, 0};
    u64 signed_weight[LIMBS];
    if (top < 0) {
        top = -top - (bottom != 0);
        bottom = 0ull - bottom;
        fr_sub(zero, weight, signed_weight);
    } else {
        load4(weight, signed_weight);
    }
    unsigned long long words[3] = {bottom, (u64)top, (u64)((u128)top >> 64)};
    unr_zero(folded);
    unr_mul_words(signed_weight, words, 3, folded);
}

__device__ __noinline__ void so_ungrouped_row(
    const u64 *narrow, const u64 *wide, unsigned int mask, unsigned long long t,
    unsigned int key, unsigned int stream, const unsigned int *offsets,
    const unsigned int *terms, const int *extension, const u64 *weight, u64 *overflow) {
    for (unsigned int node = 0; node < SO_NODES; node++) {
        long long guard = 0;
        __int128 low = 0, high = 0;
        for (unsigned int row = 0; row < SO_DOMAIN; row++) {
            long long coefficient = extension[node * SO_DOMAIN + row];
            guard += coefficient * ((int)((key >> (3 * row)) & 7u) - 3);
            __int128 a, b;
            so_integer_form(narrow, wide, mask, t, offsets, terms,
                            2 * (stream * SO_DOMAIN + row) + 1, a, b);
            low += coefficient * a;
            high += coefficient * b;
        }
        u64 folded[2 * UNR_SLOTS];
        so_weight_integer(weight, guard * low, guard * high, folded);
        unr_scatter_add(overflow + node * 2 * UNR_SLOTS, folded, UNR_SLOTS);
    }
}

__device__ __forceinline__ void so_extend_group(
    unsigned int key, unsigned int group, const u64 *groups, u64 *nodes) {
    unsigned int lane = threadIdx.x & 31u;
    if (key == 0) {
        if (lane < SO_NODES) {
            const u64 zero[LIMBS] = {0, 0, 0, 0};
            store4(nodes + (lane * SO_GROUPS + group) * LIMBS, zero);
        }
        return;
    }
    // A warp holds one difference-table diagonal across its first ten lanes.
    // GuardForms' gain bound keeps guard differences and scan partials below 2^34.
    u64 row[LIMBS] = {0, 0, 0, 0};
    u64 forward[LIMBS] = {0, 0, 0, 0}, backward[LIMBS] = {0, 0, 0, 0};
    long long guard = 0, forward_guard = 0, backward_guard = 0;
    if (lane < SO_DOMAIN) {
        load4(groups + (group * SO_DOMAIN + lane) * 2 * UNR_SLOTS, row);
        guard = (int)((key >> (3 * lane)) & 7u) - 3;
    }
    for (int limb = 0; limb < LIMBS; limb++) {
        u64 last = __shfl_sync(0xffffffffu, row[limb], SO_DOMAIN - 1);
        if (lane == 0) {
            forward[limb] = row[limb];
            backward[limb] = last;
        }
    }
    long long last_guard = __shfl_sync(0xffffffffu, guard, SO_DOMAIN - 1);
    if (lane == 0) { forward_guard = guard; backward_guard = last_guard; }
    for (unsigned int order = 1; order < SO_DOMAIN; order++) {
        u64 next[LIMBS], difference[LIMBS];
        for (int limb = 0; limb < LIMBS; limb++)
            next[limb] = __shfl_down_sync(0xffffffffu, row[limb], 1);
        long long next_guard = __shfl_down_sync(0xffffffffu, guard, 1);
        if (lane < SO_DOMAIN - order) {
            fr_sub(next, row, difference);
            load4(difference, row);
            guard = next_guard - guard;
        }
        for (int limb = 0; limb < LIMBS; limb++) {
            u64 first = __shfl_sync(0xffffffffu, row[limb], 0);
            u64 last = __shfl_sync(0xffffffffu, row[limb], SO_DOMAIN - order - 1);
            if (lane == order) { forward[limb] = first; backward[limb] = last; }
        }
        long long first = __shfl_sync(0xffffffffu, guard, 0);
        long long last = __shfl_sync(0xffffffffu, guard, SO_DOMAIN - order - 1);
        if (lane == order) { forward_guard = first; backward_guard = last; }
    }
    // Alternating signs turn a backward extension step into a suffix sum,
    // so both directions use the same parallel scan.
    if ((lane & 1u) != 0) {
        const u64 zero[LIMBS] = {0, 0, 0, 0};
        u64 negative[LIMBS];
        fr_sub(zero, forward, negative);
        load4(negative, forward);
        forward_guard = -forward_guard;
    }
    for (unsigned int side = 0; side < 2; side++) {
        unsigned int steps = side == 0 ? SO_LEFT_NODES : SO_NODES - SO_LEFT_NODES;
        for (unsigned int step = 0; step < steps; step++) {
            for (unsigned int offset = 1; offset < SO_DOMAIN; offset *= 2) {
                u64 next[LIMBS], value[LIMBS];
                for (int limb = 0; limb < LIMBS; limb++)
                    next[limb] = __shfl_down_sync(0xffffffffu,
                        side == 0 ? forward[limb] : backward[limb], offset);
                long long next_guard = __shfl_down_sync(0xffffffffu,
                    side == 0 ? forward_guard : backward_guard, offset);
                if (lane + offset < SO_DOMAIN) {
                    if (side == 0) {
                        fr_add(forward, next, value);
                        load4(value, forward);
                        forward_guard += next_guard;
                    } else {
                        fr_add(backward, next, value);
                        load4(value, backward);
                        backward_guard += next_guard;
                    }
                }
            }
            if (lane == 0) {
                long long a = side == 0 ? forward_guard : backward_guard;
                u64 value[LIMBS], folded[2 * UNR_SLOTS];
                load4(side == 0 ? forward : backward, value);
                if (a < 0) {
                    const u64 zero[LIMBS] = {0, 0, 0, 0};
                    u64 negative[LIMBS];
                    fr_sub(zero, value, negative);
                    load4(negative, value);
                }
                unsigned long long magnitude = a < 0 ? -a : a;
                unr_zero(folded);
                unr_mul_words(value, &magnitude, 1, folded);
                unr_finalize(folded, value);
                unsigned int node = side == 0 ? SO_LEFT_NODES - 1 - step : SO_LEFT_NODES + step;
                store4(nodes + (node * SO_GROUPS + group) * LIMBS, value);
            }
        }
    }
}

extern "C" __global__ void so_uniskip_kernel(
    const u64 *__restrict__ narrow, const u64 *__restrict__ wide,
    const unsigned int *__restrict__ flags, const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ terms, const int *__restrict__ extension,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out, unsigned int in_bits,
    unsigned int cycles, unsigned int tile, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int *keys = (unsigned int *)scratch;
    u64 *groups = scratch + SO_GROUPS / 2;
    u64 *overflow = groups + SO_GROUPS * SO_DOMAIN * 2 * UNR_SLOTS;
    u64 *nodes = overflow + SO_NODES * 2 * UNR_SLOTS;
    if (threadIdx.x < SO_GROUPS) keys[threadIdx.x] = 0;
    for (unsigned int i = threadIdx.x;
         i < (SO_GROUPS * SO_DOMAIN + SO_NODES) * 2 * UNR_SLOTS; i += blockDim.x)
        groups[i] = 0;
    __syncthreads();

    unsigned int stream = blockIdx.y;
    unsigned long long base = (unsigned long long)blockIdx.x * tile;
    for (unsigned int step = threadIdx.x; step < tile; step += blockDim.x) {
        unsigned long long t = base + step;
        if (t >= cycles) continue;
        unsigned int mask = flags[t];
        unsigned int key = so_guard_key(mask, stream, offsets, terms);
        unsigned int initial = (key * 0x9e3779b9u) >> 27;
        unsigned int group = SO_GROUPS;
        for (unsigned int probe = 0; probe < SO_GROUPS; probe++) {
            unsigned int slot = (initial + probe) & (SO_GROUPS - 1);
            unsigned int previous = atomicCAS(keys + slot, 0u, key);
            if (previous == 0u || previous == key) {
                group = slot;
                break;
            }
        }
        u64 weight[LIMBS];
        load4(e_in + (((t << 1) | stream) & ((1ull << in_bits) - 1)) * LIMBS, weight);
        if (group == SO_GROUPS) {
            so_ungrouped_row(narrow, wide, mask, t, key, stream, offsets, terms,
                             extension, weight, overflow);
            continue;
        }
        for (unsigned int row = 0; row < SO_DOMAIN; row++) {
            __int128 low, high;
            so_integer_form(narrow, wide, mask, t, offsets, terms,
                            2 * (stream * SO_DOMAIN + row) + 1, low, high);
            u64 folded[2 * UNR_SLOTS];
            so_weight_integer(weight, low, high, folded);
            unr_scatter_add(groups + (group * SO_DOMAIN + row) * 2 * UNR_SLOTS,
                            folded, UNR_SLOTS);
        }
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < SO_GROUPS * SO_DOMAIN; i += blockDim.x) {
        if (keys[i / SO_DOMAIN] == 0) continue;
        u64 value[LIMBS];
        unr_finalize(groups + i * 2 * UNR_SLOTS, value);
        store4(groups + i * 2 * UNR_SLOTS, value);
    }
    __syncthreads();

    for (unsigned int group = threadIdx.x / 32; group < SO_GROUPS;
         group += blockDim.x / 32)
        so_extend_group(keys[group], group, groups, nodes);
    __syncthreads();

    if (threadIdx.x < SO_NODES) {
        unsigned int node = threadIdx.x;
        u64 value[LIMBS], sum[LIMBS], next[LIMBS], outer[LIMBS];
        unr_finalize(overflow + node * 2 * UNR_SLOTS, sum);
        for (unsigned int group = 0; group < SO_GROUPS; group++) {
            load4(nodes + (node * SO_GROUPS + group) * LIMBS, value);
            fr_add(sum, value, next);
            load4(next, sum);
        }
        load4(e_out + ((base << 1) >> in_bits) * LIMBS, outer);
        fr_mul(sum, outer, value);
        store4(partials + ((unsigned long long)(stream * SO_NODES + node) * gridDim.x
                          + blockIdx.x) * LIMBS, value);
    }
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

__device__ __forceinline__ void gruen_pair_accumulate(
    const u64 *az0, const u64 *az1, const u64 *bz0, const u64 *bz1,
    const u64 *weight, u64 acc[3][LIMBS]) {
    u64 product[LIMBS], weighted[LIMBS], sum[LIMBS];
    fr_mul(az0, bz0, product);
    fr_mul(product, weight, weighted);
    fr_add(acc[0], weighted, sum);
    for (int l = 0; l < LIMBS; l++) acc[0][l] = sum[l];

    fr_mul(az1, bz1, product);
    fr_mul(product, weight, weighted);
    fr_add(acc[2], weighted, sum);
    for (int l = 0; l < LIMBS; l++) acc[2][l] = sum[l];

    u64 az_inf[LIMBS], bz_inf[LIMBS];
    fr_sub(az1, az0, az_inf);
    fr_sub(bz1, bz0, bz_inf);
    fr_mul(az_inf, bz_inf, product);
    fr_mul(product, weight, weighted);
    fr_add(acc[1], weighted, sum);
    for (int l = 0; l < LIMBS; l++) acc[1][l] = sum[l];
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

    u64 acc[3][LIMBS];
    for (int lane = 0; lane < 3; lane++)
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

        gruen_pair_accumulate(az0, az1, bz0, bz1, weight, acc);
    }

    so_block_reduce(scratch, 3, acc, partials);
}

extern "C" __global__ void gruen_prefix_pair_message_kernel(
    const u64 *__restrict__ az, const u64 *__restrict__ bz,
    const u64 *__restrict__ e_in, const u64 *__restrict__ e_out,
    unsigned int e_in_len, unsigned int in_bits, unsigned int count,
    unsigned int strip, unsigned int prefix, unsigned int tail_width,
    u64 tail0, u64 tail1, u64 tail2, u64 tail3, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long base =
        ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * strip;
    unsigned int prefix_pairs = (prefix + 1) / 2;
    u64 acc[3][LIMBS] = {};
    for (unsigned int step = 0; step < strip; step++) {
        unsigned long long pair = base + step;
        if (pair >= count) break;
        unsigned long long lo, hi;
        u64 weight[LIMBS];
        if (pair == prefix_pairs) {
            lo = prefix;
            hi = prefix + tail_width - 1;
            weight[0] = tail0;
            weight[1] = tail1;
            weight[2] = tail2;
            weight[3] = tail3;
        } else {
            lo = min(2 * pair, (unsigned long long)prefix + tail_width - 1);
            hi = min(2 * pair + 1, (unsigned long long)prefix + tail_width - 1);
            if (e_in_len <= 1) load4(e_out + pair * LIMBS, weight);
            else so_split_eq(e_in, e_out, in_bits, pair, weight);
        }
        u64 az0[LIMBS], az1[LIMBS], bz0[LIMBS], bz1[LIMBS];
        load4(az + lo * LIMBS, az0);
        load4(az + hi * LIMBS, az1);
        load4(bz + lo * LIMBS, bz0);
        load4(bz + hi * LIMBS, bz1);
        gruen_pair_accumulate(az0, az1, bz0, bz1, weight, acc);
    }
    so_block_reduce(scratch, 3, acc, partials);
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

        // NVCC 12.4 miscompiles this accumulator loop with a dynamic upper bound.
        for (unsigned int lane = 0; lane < SO_CLAIM_LANES; lane++) {
            if (lane >= lanes) continue;
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
