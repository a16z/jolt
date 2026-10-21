#define SO_NARROW 13
#define SO_WIDE 4
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

    so_block_reduce(scratch, 3, acc, partials);
}
